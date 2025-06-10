import os
import sys


sys.path.append(os.getcwd())

from nets.layers import *
from nets.base import TrainWrapperBaseClass
# from nets.spg.faceformer import Faceformer
from nets.spg.s2g_face import Generator as s2g_face
from losses import KeypointLoss
from nets.utils import denormalize
from data_utils import get_mfcc_psf, get_mfcc_psf_min, get_mfcc_ta
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import normalize
import smplx
from vocalist.models.model import SyncTransformer
from vocalist.hparams import hparams
from torchaudio.transforms import MelScale
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftSilhouetteShader,
    TexturesVertex,
)
from pytorch3d.structures import Meshes
import torchaudio

TOP_DB = -hparams.min_level_db
MIN_LEVEL = np.exp(TOP_DB / -20 * np.log(10))


class TrainWrapper(TrainWrapperBaseClass):
    '''
    a wrapper receving a batch from data_utils and calculate loss
    '''

    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = torch.device(self.args.gpu)
        self.global_step = 0

        self.convert_to_6d = self.config.Data.pose.convert_to_6d
        self.expression = self.config.Data.pose.expression
        self.epoch = 0
        self.init_params()
        self.num_classes = 4

        self.generator = s2g_face(
            n_poses=self.config.Data.pose.generate_length,
            each_dim=self.each_dim,
            dim_list=self.dim_list,
            training=not self.args.infer,
            device=self.device,
            identity=False if self.convert_to_6d else True,
            num_classes=self.num_classes,
        ).to(self.device)

        # self.generator = Faceformer().to(self.device)

        self.discriminator = None
        self.am = None

        self.MSELoss = KeypointLoss().to(self.device)
        # Lip-sync discriminator used for sync loss
        self.sync_model = SyncTransformer().to(self.device)
        # Freeze sync model parameters
        self.sync_model.requires_grad_(False)


        # sync network
        self.sync_model = SyncTransformer().to(self.device)
        if os.path.exists('vocalist/vocalist_5f_lrs2.pth'):
            ckpt = torch.load('vocalist/vocalist_5f_lrs2.pth', map_location=self.device)
            self.sync_model.load_state_dict(ckpt['state_dict'])
        self.sync_model.eval()

        self.vshift = hparams.v_shift
        self.v_context = 5
        self.melscale = MelScale(n_mels=hparams.num_mels,
                                 sample_rate=hparams.sample_rate,
                                 f_min=hparams.fmin,
                                 f_max=hparams.fmax,
                                 n_stft=hparams.n_stft,
                                 norm='slaney',
                                 mel_scale='slaney').to(self.device)

        try:
            model_path = os.path.join(os.path.dirname(__file__), '../visualise/smplx/SMPLX_NEUTRAL.npz')
            model_data = np.load(model_path, allow_pickle=True)
            self.faces = torch.tensor(model_data['f'].astype(np.int64), device=self.device)
            cameras = FoVPerspectiveCameras(device=self.device)
            raster = MeshRasterizer(cameras=cameras,
                                   raster_settings=RasterizationSettings(image_size=96, faces_per_pixel=1))
            self.renderer = MeshRenderer(rasterizer=raster,
                                        shader=SoftSilhouetteShader(device=self.device, cameras=cameras))
            self.smplx_model = smplx.create(model_path=os.path.dirname(model_path), model_type='smplx',
                                           use_pca=False, create_expression=True, num_expression_coeffs=100,
                                           create_jaw_pose=True, create_leye_pose=True, create_reye_pose=True,
                                           create_body_pose=True, create_left_hand_pose=True,
                                           create_right_hand_pose=True, create_global_orient=True,
                                           create_transl=False).to(self.device)
        except Exception:
            self.renderer = None
            self.faces = None
            self.smplx_model = None




        super().__init__(args, config)

    def init_optimizer(self):
        self.generator_optimizer = optim.SGD(
            filter(lambda p: p.requires_grad,self.generator.parameters()),
            lr=0.001,
            momentum=0.9,
            nesterov=False,
        )

    def init_params(self):
        if self.convert_to_6d:
            scale = 2
        else:
            scale = 1

        global_orient = round(3 * scale)
        leye_pose = reye_pose = round(3 * scale)
        jaw_pose = round(3 * scale)
        body_pose = round(63 * scale)
        left_hand_pose = right_hand_pose = round(45 * scale)
        if self.expression:
            expression = 100
        else:
            expression = 0

        b_j = 0
        jaw_dim = jaw_pose
        b_e = b_j + jaw_dim
        eye_dim = leye_pose + reye_pose
        b_b = b_e + eye_dim
        body_dim = global_orient + body_pose
        b_h = b_b + body_dim
        hand_dim = left_hand_pose + right_hand_pose
        b_f = b_h + hand_dim
        face_dim = expression

        self.dim_list = [b_j, b_e, b_b, b_h, b_f]
        self.full_dim = jaw_dim + eye_dim + body_dim + hand_dim + face_dim
        self.pose = int(self.full_dim / round(3 * scale))
        self.each_dim = [jaw_dim, eye_dim + body_dim, hand_dim, face_dim]


    def audio_to_mel(self, wav_path):
        wav, sr = torchaudio.load(wav_path)
        if sr != hparams.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, hparams.sample_rate)
        spec = torch.stft(wav[0], n_fft=hparams.n_fft, hop_length=hparams.hop_size,
                          win_length=hparams.win_size,
                          window=torch.hann_window(hparams.win_size).to(wav.device),
                          return_complex=True)
        melspec = self.melscale(torch.abs(spec))
        melspec_tr1 = (20 * torch.log10(torch.clamp(melspec, min=MIN_LEVEL))) - hparams.ref_level_db
        normalized_mel = torch.clamp((2 * hparams.max_abs_value) * ((melspec_tr1 + TOP_DB) / TOP_DB) - hparams.max_abs_value,
                                     -hparams.max_abs_value, hparams.max_abs_value)
        return normalized_mel.unsqueeze(0)

    def render_mouth(self, params, betas):
        if self.renderer is None or self.smplx_model is None or self.faces is None:
            return None
        B, T, C = params.shape
        x = params.reshape(-1, C)
        out = self.smplx_model(
            betas=betas.repeat(T * B, 1),
            expression=x[:, 165:265],
            jaw_pose=x[:, 0:3],
            leye_pose=x[:, 3:6],
            reye_pose=x[:, 6:9],
            global_orient=x[:, 9:12],
            body_pose=x[:, 12:75],
            left_hand_pose=x[:, 75:120],
            right_hand_pose=x[:, 120:165],
            return_verts=True,
        )
        verts = out.vertices.float()
        textures = TexturesVertex(verts_features=torch.ones_like(verts))
        meshes = Meshes(verts=verts, faces=self.faces.unsqueeze(0).repeat(verts.shape[0],1,1), textures=textures)
        imgs = self.renderer(meshes)[:, :, :, :3].permute(0,3,1,2)
        imgs = imgs.view(B, T, 3, 96, 96)
        return imgs[:, :, :, 48:]

    def calc_pdist(self, vid_feat, mel_feat, vshift=15):
        win_size = vshift * 2 + 1
        feat2p = F.pad(mel_feat.permute(1,2,3,0), (vshift, vshift)).permute(3,0,1,2)
        dists = []
        for i in range(len(vid_feat)):
            scores = self.sync_model(vid_feat[i].unsqueeze(0).repeat(win_size,1,1,1), feat2p[i:i+win_size])
            dists.append(scores)
        return torch.stack(dists)


    def __call__(self, bat):
        # assert (not self.args.infer), "infer mode"
        self.global_step += 1

        total_loss = None
        loss_dict = {}

        aud, poses = bat['aud_feat'].to(self.device).to(torch.float32), bat['poses'].to(self.device).to(torch.float32)
        id = bat['speaker'].to(self.device) - 20
        id = F.one_hot(id, self.num_classes)

        aud = aud.permute(0, 2, 1)
        gt_poses = poses.permute(0, 2, 1)

        if self.expression:
            expression = bat['expression'].to(self.device).to(torch.float32)
            gt_poses = torch.cat([gt_poses, expression.permute(0, 2, 1)], dim=2)

        pred_poses, _ = self.generator(
            aud,
            gt_poses,
            id,
        )

        G_loss, G_loss_dict = self.get_loss(
            pred_poses=pred_poses,
            gt_poses=gt_poses,
            pre_poses=None,
            mode='training_G',
            gt_conf=None,
            aud=aud,
            aud_file=bat.get('aud_file', None),
            betas=bat.get('betas', torch.zeros([1,300], device=self.device)),
        )

        self.generator_optimizer.zero_grad()
        G_loss.backward()
        grad = torch.nn.utils.clip_grad_norm(self.generator.parameters(), self.config.Train.max_gradient_norm)
        loss_dict['grad'] = grad.item()
        self.generator_optimizer.step()

        for key in list(G_loss_dict.keys()):
            loss_dict[key] = G_loss_dict.get(key, 0).item()

        return total_loss, loss_dict



    def get_loss(self,
                 pred_poses,
                 gt_poses,
                 pre_poses,
                 aud,
                 aud_file=None,
                 betas=None,
                 mode='training_G',
                 gt_conf=None,
                 exp=1,
                 gt_nzero=None,
                 pre_nzero=None,
                 ):
        loss_dict = {}


        [b_j, b_e, b_b, b_h, b_f] = self.dim_list
        betas = torch.zeros(pred_poses.size(0), 10, device=self.device)
        imgs = self.render_mouth(pred_poses, betas.to(self.device))
        sync_out = self.sync_model(imgs, aud)
        sync_loss = sync_out.mean()

        MSELoss = torch.mean(torch.abs(pred_poses[:, :, :6] - gt_poses[:, :, :6]))
        if self.expression:
            expl = torch.mean((pred_poses[:, :, -100:] - gt_poses[:, :, -100:])**2)
        else:
            expl = 0

        sync_loss = torch.tensor(0.0, device=pred_poses.device)
        if aud_file is not None and self.renderer is not None:
            mel = self.audio_to_mel(aud_file)
            imgs = self.render_mouth(pred_poses.detach(), betas.to(self.device))
            if imgs is not None:
                lim, lcc = [], []
                mel_step_size = 16
                T = imgs.shape[1] - self.v_context
                for t in range(T):
                    lim.append(imgs[:, t:t+self.v_context].reshape(-1, self.v_context*3, 48, 96))
                    start = int(80.0 * (t / hparams.fps))
                    lcc.append(mel[:, :, :, start:start+mel_step_size])
                if lim:
                    lim = torch.cat(lim, 0)
                    lcc = torch.cat(lcc, 0)
                    dist = self.calc_pdist(lim, lcc, vshift=self.vshift)
                    target = torch.full((dist.shape[0],), self.vshift, dtype=torch.long, device=dist.device)
                    sync_loss = F.cross_entropy(dist, target)

        gen_loss = expl + MSELoss + sync_loss

        loss_dict['MSELoss'] = MSELoss
        if self.expression:
            loss_dict['exp_loss'] = expl
        loss_dict['sync_loss'] = sync_loss

        return gen_loss, loss_dict

    def infer_on_audio(self, aud_fn, id=None, initial_pose=None, norm_stats=None, w_pre=False, frame=None, am=None, am_sr=16000, **kwargs):
        '''
        initial_pose: (B, C, T), normalized
        (aud_fn, txgfile) -> generated motion (B, T, C)
        '''
        output = []

        # assert self.args.infer, "train mode"
        self.generator.eval()

        if self.config.Data.pose.normalization:
            assert norm_stats is not None
            data_mean = norm_stats[0]
            data_std = norm_stats[1]

        # assert initial_pose.shape[-1] == pre_length
        if initial_pose is not None:
            gt = initial_pose[:,:,:].permute(0, 2, 1).to(self.generator.device).to(torch.float32)
            pre_poses = initial_pose[:,:,:15].permute(0, 2, 1).to(self.generator.device).to(torch.float32)
            poses = initial_pose.permute(0, 2, 1).to(self.generator.device).to(torch.float32)
            B = pre_poses.shape[0]
        else:
            gt = None
            pre_poses=None
            B = 1

        if type(aud_fn) == torch.Tensor:
            aud_feat = torch.tensor(aud_fn, dtype=torch.float32).to(self.generator.device)
            num_poses_to_generate = aud_feat.shape[-1]
        else:
            aud_feat = get_mfcc_ta(aud_fn, am=am, am_sr=am_sr, fps=30, encoder_choice='faceformer')
            aud_feat = aud_feat[np.newaxis, ...].repeat(B, axis=0)
            aud_feat = torch.tensor(aud_feat, dtype=torch.float32).to(self.generator.device).transpose(1, 2)
        if frame is None:
            frame = aud_feat.shape[2]*30//16000
        #
        if id is None:
            id = torch.tensor([[0, 0, 0, 0]], dtype=torch.float32, device=self.generator.device)
        else:
            id = F.one_hot(id, self.num_classes).to(self.generator.device)

        with torch.no_grad():
            pred_poses = self.generator(aud_feat, pre_poses, id, time_steps=frame)[0]
            pred_poses = pred_poses.cpu().numpy()
        output = pred_poses

        if self.config.Data.pose.normalization:
            output = denormalize(output, data_mean, data_std)

        return output


    def generate(self, wv2_feat, frame):
        '''
        initial_pose: (B, C, T), normalized
        (aud_fn, txgfile) -> generated motion (B, T, C)
        '''
        output = []

        # assert self.args.infer, "train mode"
        self.generator.eval()

        B = 1

        id = torch.tensor([[0, 0, 0, 0]], dtype=torch.float32, device=self.generator.device)
        id = id.repeat(wv2_feat.shape[0], 1)

        with torch.no_grad():
            pred_poses = self.generator(wv2_feat, None, id, time_steps=frame)[0]
        return pred_poses
