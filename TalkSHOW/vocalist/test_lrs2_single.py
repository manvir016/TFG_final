import os
import math
import torch
import argparse
import numpy as np
import soundfile as sf
import torch.nn.functional as F
import cv2
from glob import glob
from natsort import natsorted
from models.model import SyncTransformer
from hparams import hparams
from torch.utils import data as data_utils
from torchaudio.transforms import MelScale

v_context = 5
mel_step_size = 16
BATCH_SIZE = 1
TOP_DB = -hparams.min_level_db
MIN_LEVEL = np.exp(TOP_DB / -20 * np.log(10))
melscale = MelScale(n_mels=hparams.num_mels, sample_rate=hparams.sample_rate,
                    f_min=hparams.fmin, f_max=hparams.fmax,
                    n_stft=hparams.n_stft, norm='slaney', mel_scale='slaney')

parser = argparse.ArgumentParser(description='Run VocaLiST on one video folder and output k15.')
parser.add_argument('--video_dir', type=str, required=True, help='Folder containing lips frames + audio.wav')
parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path of the VocaLiST model')
parser.add_argument('--output', type=str, required=True, help='Path to save k15.txt')
args = parser.parse_args()


class Dataset(data_utils.Dataset):
    def __init__(self, video_folder):
        self.video_folder = video_folder
        self.frames = natsorted(glob(os.path.join(video_folder, "*.jpg")))
        self.wav_path = os.path.join(video_folder, "audio.wav")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        img_names = self.frames
        wav = sf.read(self.wav_path)[0]
        min_length = min(len(img_names), math.floor(len(wav) / 640))
        lastframe = min_length - v_context

        window = []
        for fname in img_names[:lastframe + v_context]:
            img = cv2.imread(fname)
            img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            window.append(img)

        vid = np.concatenate(window, axis=2) / 255.
        vid = vid.transpose(2, 0, 1)
        vid = torch.FloatTensor(vid[:, 48:])

        aud_tensor = torch.FloatTensor(wav)
        spec = torch.stft(aud_tensor, n_fft=hparams.n_fft, hop_length=hparams.hop_size,
                          win_length=hparams.win_size, window=torch.hann_window(hparams.win_size), return_complex=True)
        melspec = melscale(torch.abs(spec).float())
        melspec_tr1 = (20 * torch.log10(torch.clamp(melspec, min=MIN_LEVEL))) - hparams.ref_level_db
        normalized_mel = torch.clip((2 * hparams.max_abs_value) * ((melspec_tr1 + TOP_DB) / TOP_DB) - hparams.max_abs_value,
                                    -hparams.max_abs_value, hparams.max_abs_value)
        mels = normalized_mel.unsqueeze(0)
        return vid, mels, lastframe


def calc_pdist(model, feat1, feat2, vshift=15):
    win_size = vshift * 2 + 1
    feat2p = F.pad(feat2.permute(1, 2, 3, 0), (vshift, vshift)).permute(3, 0, 1, 2)
    dists = []
    for i in range(len(feat1)):
        sync_scores = model(feat1[i].unsqueeze(0).repeat(win_size, 1, 1, 1).to(device),
                            feat2p[i:i + win_size].to(device))
        dists.append(sync_scores.cpu())
    return dists


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SyncTransformer().to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dataset = Dataset(args.video_dir)
    loader = data_utils.DataLoader(dataset, batch_size=1, num_workers=0)

    for vid, aud, lastframe in loader:
        vid = vid.view(1, lastframe + v_context, 3, 48, 96)
        batch_size = 20
        lim_in, lcc_in = [], []

        for i in range(0, lastframe, batch_size):
            im_batch = [vid[:, vframe:vframe + v_context, :, :, :].view(1, -1, 48, 96)
                        for vframe in range(i, min(lastframe, i + batch_size))]
            cc_batch = [aud[:, :, :, int(80.*(vframe / hparams.fps)):int(80.*(vframe / hparams.fps)) + mel_step_size]
                        for vframe in range(i, min(lastframe, i + batch_size))]
            lim_in.append(torch.cat(im_batch, 0))
            lcc_in.append(torch.cat(cc_batch, 0))

        lim_in = torch.cat(lim_in, 0).to(device)
        lcc_in = torch.cat(lcc_in, 0).to(device)
        dists = calc_pdist(model, lim_in, lcc_in, vshift=hparams.v_shift)

        dist_tensor_k5 = torch.stack(dists)
        dist_tensor_k15 = (dist_tensor_k5[5:-5] + dist_tensor_k5[4:-6] + dist_tensor_k5[6:-4] +
                           dist_tensor_k5[3:-7] + dist_tensor_k5[7:-3] + dist_tensor_k5[2:-8] +
                           dist_tensor_k5[8:-2] + dist_tensor_k5[1:-9] + dist_tensor_k5[9:-1] +
                           dist_tensor_k5[:-10] + dist_tensor_k5[10:]) / 11

        offsets_k15 = hparams.v_shift - torch.argmax(dist_tensor_k15, dim=1)
        correct = ((offsets_k15 == -1) | (offsets_k15 == 0) | (offsets_k15 == 1)).sum().item()
        k15 = correct / len(offsets_k15)

        with open(args.output, "w") as f:
            f.write(str(k15))

        print(f"[INFO] k15 = {k15:.4f} guardado en {args.output}")

