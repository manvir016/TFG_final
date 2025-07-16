import pickle
import sys
import os

sys.path.append(os.getcwd())

import json
from glob import glob
from data_utils.utils import *
import torch.utils.data as data
from data_utils.consts import speaker_id
from data_utils.lower_body import count_part
import random
from data_utils.rotation_conversion import axis_angle_to_matrix, matrix_to_rotation_6d

import random
import librosa
import numpy as np
import soundfile as sf
import uuid


with open('data_utils/hand_component.json') as file_obj:
    comp = json.load(file_obj)
    left_hand_c = np.asarray(comp['left'])
    right_hand_c = np.asarray(comp['right'])


def to3d(data):
    left_hand_pose = np.einsum('bi,ij->bj', data[:, 75:87], left_hand_c[:12, :])
    right_hand_pose = np.einsum('bi,ij->bj', data[:, 87:99], right_hand_c[:12, :])
    data = np.concatenate((data[:, :75], left_hand_pose, right_hand_pose), axis=-1)
    return data


class SmplxDataset():
    '''
    creat a dataset for every segment and concat.
    '''

    def __init__(self,
                 data_root,
                 speaker,
                 motion_fn,
                 audio_fn,
                 audio_sr,
                 fps,
                 feat_method='mel_spec',
                 audio_feat_dim=64,
                 audio_feat_win_size=None,

                 train=True,
                 load_all=False,
                 split_trans_zero=False,
                 limbscaling=False,
                 num_frames=25,
                 num_pre_frames=25,
                 num_generate_length=25,
                 context_info=False,
                 convert_to_6d=False,
                 expression=False,
                 config=None,
                 am=None,
                 am_sr=None,
                 whole_video=False
                 ):

        self.data_root = data_root
        self.speaker = speaker

        self.feat_method = feat_method
        self.audio_fn = audio_fn
        self.audio_sr = audio_sr
        self.fps = fps
        self.audio_feat_dim = audio_feat_dim
        self.audio_feat_win_size = audio_feat_win_size
        self.context_info = context_info  # for aud feat
        self.convert_to_6d = convert_to_6d
        self.expression = expression

        self.train = train
        self.load_all = load_all
        self.split_trans_zero = split_trans_zero
        self.limbscaling = limbscaling
        self.num_frames = num_frames
        self.num_pre_frames = num_pre_frames
        self.num_generate_length = num_generate_length
        # print('num_generate_length ', self.num_generate_length)

        self.config = config
        self.am_sr = am_sr
        self.whole_video = whole_video
        load_mode = self.config.dataset_load_mode

        if load_mode == 'pickle':
            raise NotImplementedError

        elif load_mode == 'csv':
            import pickle
            with open(data_root, 'rb') as f:
                u = pickle._Unpickler(f)
                data = u.load()
                self.data = data[0]
            if self.load_all:
                self._load_npz_all()

        elif load_mode == 'json':
            self.annotations = glob(data_root + '/*pkl')
            if len(self.annotations) == 0:
                raise FileNotFoundError(data_root + ' are empty')
            self.annotations = sorted(self.annotations)
            self.img_name_list = self.annotations

            if self.load_all:
                self._load_them_all(am, am_sr, motion_fn)

    def _load_npz_all(self):
        self.loaded_data = {}
        self.complete_data = []
        data = self.data
        shape = data['body_pose_axis'].shape[0]
        self.betas = data['betas']
        self.img_name_list = []
        for index in range(shape):
            img_name = f'{index:6d}'
            self.img_name_list.append(img_name)

            jaw_pose = data['jaw_pose'][index]
            leye_pose = data['leye_pose'][index]
            reye_pose = data['reye_pose'][index]
            global_orient = data['global_orient'][index]
            body_pose = data['body_pose_axis'][index]
            left_hand_pose = data['left_hand_pose'][index]
            right_hand_pose = data['right_hand_pose'][index]

            full_body = np.concatenate(
                (jaw_pose, leye_pose, reye_pose, global_orient, body_pose, left_hand_pose, right_hand_pose))
            assert full_body.shape[0] == 99
            if self.convert_to_6d:
                full_body = to3d(full_body)
                full_body = torch.from_numpy(full_body)
                full_body = matrix_to_rotation_6d(axis_angle_to_matrix(full_body))
                full_body = np.asarray(full_body)
                if self.expression:
                    expression = data['expression'][index]
                    full_body = np.concatenate((full_body, expression))
                # full_body = np.concatenate((full_body, non_zero))
            else:
                full_body = to3d(full_body)
                if self.expression:
                    expression = data['expression'][index]
                    full_body = np.concatenate((full_body, expression))

            self.loaded_data[img_name] = full_body.reshape(-1)
            self.complete_data.append(full_body.reshape(-1))

        self.complete_data = np.array(self.complete_data)

        if self.audio_feat_win_size is not None:
            self.audio_feat = get_mfcc_old(self.audio_fn).transpose(1, 0)
            # print(self.audio_feat.shape)
        else:
            if self.feat_method == 'mel_spec':
                self.audio_feat = get_melspec(self.audio_fn, fps=self.fps, sr=self.audio_sr, n_mels=self.audio_feat_dim)
            elif self.feat_method == 'mfcc':
                self.audio_feat = get_mfcc(self.audio_fn,
                                           smlpx=True,
                                           sr=self.audio_sr,
                                           n_mfcc=self.audio_feat_dim,
                                           win_size=self.audio_feat_win_size
                                           )

    def _load_them_all(self, am, am_sr, motion_fn):
        self.loaded_data = {}
        self.complete_data = []
        #f = open(motion_fn, 'rb+')
        #data = pickle.load(f)
        with open(motion_fn, 'rb') as f:
                data = pickle.load(f)

        use_silence = random.random() < 0.25  # 5% of samples = silent
        use_noise_only = not use_silence and random.random() < 0.25  # 5% = noise


        if use_silence or use_noise_only:
      
            # Duration: random between 2 and 8 seconds
            duration_seconds = random.uniform(2.0, 8.0)
            sr = 16000
            num_samples = int(sr * duration_seconds)

            if use_silence:
                print(f"[AUG] Injecting SILENT audio, duration: {duration_seconds:.2f}s")
                speech_array = np.zeros(num_samples)
            else:
                noise_dir = os.path.join(os.path.dirname(__file__), '..', 'demo_audio', 'noise')
                noise_files = [f for f in os.listdir(noise_dir) if f.endswith('.wav')]
                
                if noise_files:
                    noise_file = random.choice(noise_files)
                    noise_array, _ = librosa.load(os.path.join(noise_dir, noise_file), sr=sr)

                    if len(noise_array) < num_samples:
                        noise_array = np.tile(noise_array, int(np.ceil(num_samples / len(noise_array))))
                    speech_array = noise_array[:num_samples]

                    print(f"[AUG] Injecting NOISE-ONLY audio from '{noise_file}', duration: {duration_seconds:.2f}s")
                else:
                    print("[WARNING] No noise files found, defaulting to silence")
                    speech_array = np.zeros(num_samples)
            

            unique_id = uuid.uuid4().hex[:8]
            label = "silent" if use_silence else "noise"
            filename = f"{label}_augmented_{unique_id}.wav"
            debug_audio_path = os.path.join(os.path.dirname(__file__), '..', 'demo_audio', 'debug_aug_audio', filename)
            os.makedirs(os.path.dirname(debug_audio_path), exist_ok=True)
            sf.write(debug_audio_path, speech_array, samplerate=sr)
            print(f"[DEBUG] Saved augmented audio: {debug_audio_path}")



            self.betas = np.array(data['betas'])

            # Approximate number of frames
            bs = num_samples // (sr // self.fps)

            zero_shape = (bs, 3)
            full_body = np.concatenate([
                np.zeros(zero_shape),  # jaw_pose
                np.zeros(zero_shape),  # leye_pose
                np.zeros(zero_shape),  # reye_pose
                np.zeros(zero_shape),  # global_orient
                np.zeros((bs, 21 * 3)),  # body_pose_axis
                np.zeros((bs, 12 * 3)),  # left_hand_pose
                np.zeros((bs, 12 * 3)),  # right_hand_pose
            ], axis=1)

            full_body = to3d(full_body)
            
            if self.convert_to_6d:
                full_body = torch.from_numpy(full_body)
                full_body = matrix_to_rotation_6d(axis_angle_to_matrix(full_body.reshape(-1, 55, 3))).reshape(-1, 330)
                full_body = np.asarray(full_body)
            if self.expression:
                full_body = np.concatenate((full_body, np.zeros((bs, 100))), axis=1)
            self.complete_data = full_body

        else:

            self.betas = np.array(data['betas'])

            jaw_pose = np.array(data['jaw_pose'])
            leye_pose = np.array(data['leye_pose'])
            reye_pose = np.array(data['reye_pose'])
            global_orient = np.array(data['global_orient']).squeeze()
            body_pose = np.array(data['body_pose_axis'])
            left_hand_pose = np.array(data['left_hand_pose'])
            right_hand_pose = np.array(data['right_hand_pose'])

            full_body = np.concatenate(
                (jaw_pose, leye_pose, reye_pose, global_orient, body_pose, left_hand_pose, right_hand_pose), axis=1)
            assert full_body.shape[1] == 99


            if self.convert_to_6d:
                full_body = to3d(full_body)
                full_body = torch.from_numpy(full_body)
                full_body = matrix_to_rotation_6d(axis_angle_to_matrix(full_body.reshape(-1, 55, 3))).reshape(-1, 330)
                full_body = np.asarray(full_body)
                if self.expression:
                    expression = np.array(data['expression'])
                    full_body = np.concatenate((full_body, expression), axis=1)

            else:
                full_body = to3d(full_body)
                expression = np.array(data['expression'])
                full_body = np.concatenate((full_body, expression), axis=1)

            self.complete_data = full_body


            # Carga del audio original
            speech_array, sampling_rate = librosa.load(self.audio_fn, sr=16000, dtype=np.float32)


            # NUEVO: Añadir ruido dinámicamente
            noise_prob = 0.75  # 75% de probabilidad de aplicar ruido
            noise_dir = os.path.join(os.path.dirname(__file__), '..', 'demo_audio', 'noise')
            noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith('.wav')]

            if random.random() < noise_prob and len(noise_files) > 0:
                noise_file = random.choice(noise_files)
                noise_array, _ = librosa.load(noise_file, sr=16000)

                # Ajustar longitud del ruido para coincidir con el audio original
                if len(noise_array) < len(speech_array):
                    repeats = int(np.ceil(len(speech_array) / len(noise_array)))
                    noise_array = np.tile(noise_array, repeats)[:len(speech_array)]
                else:
                    start_idx = random.randint(0, len(noise_array) - len(speech_array))
                    noise_array = noise_array[start_idx:start_idx + len(speech_array)]

                # Añadir ruido con intensidad controlada
                noise_level = random.uniform(0.05, 0.2)
                speech_array = speech_array + noise_array * noise_level
                print(f"[INFO] Ruido añadido: {os.path.basename(noise_file)} con nivel {noise_level:.2f}")
                
                # Ruta relativa desde data_utils hasta demo_audio/debug
                unique_id = uuid.uuid4().hex[:8]
                filename = f"debug_augmented_{unique_id}.wav"
                debug_audio_path = os.path.join(os.path.dirname(__file__), '..', 'demo_audio', 'debug', filename)

                # Asegúrate de que el directorio existe
                os.makedirs(os.path.dirname(debug_audio_path), exist_ok=True)

                # Guarda el audio
                sf.write(debug_audio_path, speech_array, samplerate=16000)
                # Normalizar audio para evitar saturación
                max_amp = np.max(np.abs(speech_array))
                if max_amp > 1.0:
                    speech_array = speech_array / max_amp

        # Procesar audio modificado según configuración original
        if self.config.Model.encoder_choice == 'faceformer':
            audio_ft = speech_array.reshape(-1, 1)
        elif self.config.Model.encoder_choice == 'meshtalk':
            audio_ft = 0.01 * speech_array / np.mean(np.abs(speech_array))
        elif self.config.Model.encoder_choice == 'onset':
            audio_ft = librosa.onset.onset_detect(y=speech_array, sr=16000, units='time').reshape(-1, 1)
        else:
            audio = torch.tensor(speech_array, dtype=torch.float32).unsqueeze(0)
            n_fft = 2048
            hop_length = 734 if self.fps == 30 else 1467
            mfcc_transform = ta.transforms.MFCC(
                sample_rate=16000,
                n_mfcc=64,
                melkwargs={"n_fft": n_fft, "hop_length": hop_length, "n_mels": 256, "mel_scale": "htk"}
            )
            audio_ft = mfcc_transform(audio).squeeze(dim=0).transpose(0, 1).numpy()

        self.audio_feat = audio_ft

    def get_dataset(self, normalization=False, normalize_stats=None, split='train'):

        class __Worker__(data.Dataset):
            def __init__(child, index_list, normalization, normalize_stats, split='train') -> None:
                super().__init__()
                child.index_list = index_list
                child.normalization = normalization
                child.normalize_stats = normalize_stats
                child.split = split

            def __getitem__(child, index):
                num_generate_length = self.num_generate_length
                num_pre_frames = self.num_pre_frames
                seq_len = num_generate_length + num_pre_frames
                # print(num_generate_length)

                index = child.index_list[index]
                index_new = index + random.randrange(0, 5, 3)
                if index_new + seq_len > self.complete_data.shape[0]:
                    index_new = index
                index = index_new

                if child.split in ['val', 'pre', 'test'] or self.whole_video:
                    index = 0
                    seq_len = self.complete_data.shape[0]
                seq_data = []
                assert index + seq_len <= self.complete_data.shape[0]
                # print(seq_len)
                seq_data = self.complete_data[index:(index + seq_len), :]
                seq_data = np.array(seq_data)

                '''
                audio feature，
                '''
                if not self.context_info:
                    if not self.whole_video:
                        audio_feat = self.audio_feat[index:index + seq_len, ...]
                        if audio_feat.shape[0] < seq_len:
                            audio_feat = np.pad(audio_feat, [[0, seq_len - audio_feat.shape[0]], [0, 0]],
                                                mode='reflect')

                        assert audio_feat.shape[0] == seq_len and audio_feat.shape[1] == self.audio_feat_dim
                    else:
                        audio_feat = self.audio_feat

                else:  # including feature and history
                    if self.audio_feat_win_size is None:
                        audio_feat = self.audio_feat[index:index + seq_len + num_pre_frames, ...]
                        if audio_feat.shape[0] < seq_len + num_pre_frames:
                            audio_feat = np.pad(audio_feat,
                                                [[0, seq_len + self.num_frames - audio_feat.shape[0]], [0, 0]],
                                                mode='constant')

                        assert audio_feat.shape[0] == self.num_frames + seq_len and audio_feat.shape[
                            1] == self.audio_feat_dim

                if child.normalization:
                    data_mean = child.normalize_stats['mean'].reshape(1, -1)
                    data_std = child.normalize_stats['std'].reshape(1, -1)
                    seq_data[:, :330] = (seq_data[:, :330] - data_mean) / data_std
                if child.split in['train', 'test']:
                    if self.convert_to_6d:
                        if self.expression:
                            data_sample = {
                                'poses': seq_data[:, :330].astype(np.float).transpose(1, 0),
                                'expression': seq_data[:, 330:].astype(np.float).transpose(1, 0),
                                # 'nzero': seq_data[:, 375:].astype(np.float).transpose(1, 0),
                                'aud_feat': audio_feat.astype(np.float).transpose(1, 0),
                                'speaker': speaker_id[self.speaker],
                                'betas': self.betas,
                                'aud_file': self.audio_fn,
                            }
                        else:
                            data_sample = {
                                'poses': seq_data[:, :330].astype(np.float).transpose(1, 0),
                                'nzero': seq_data[:, 330:].astype(np.float).transpose(1, 0),
                                'aud_feat': audio_feat.astype(np.float).transpose(1, 0),
                                'speaker': speaker_id[self.speaker],
                                'betas': self.betas
                            }
                    else:
                        if self.expression:
                            data_sample = {
                                'poses': seq_data[:, :165].astype(np.float).transpose(1, 0),
                                'expression': seq_data[:, 165:].astype(np.float).transpose(1, 0),
                                'aud_feat': audio_feat.astype(np.float).transpose(1, 0),
                                # 'wv2_feat': wv2_feat.astype(np.float).transpose(1, 0),
                                'speaker': speaker_id[self.speaker],
                                'aud_file': self.audio_fn,
                                'betas': self.betas
                            }
                        else:
                            data_sample = {
                                'poses': seq_data.astype(np.float).transpose(1, 0),
                                'aud_feat': audio_feat.astype(np.float).transpose(1, 0),
                                'speaker': speaker_id[self.speaker],
                                'betas': self.betas
                            }
                    return data_sample
                else:
                    data_sample = {
                        'poses': seq_data[:, :330].astype(np.float).transpose(1, 0),
                        'expression': seq_data[:, 330:].astype(np.float).transpose(1, 0),
                        # 'nzero': seq_data[:, 325:].astype(np.float).transpose(1, 0),
                        'aud_feat': audio_feat.astype(np.float).transpose(1, 0),
                        'aud_file': self.audio_fn,
                        'speaker': speaker_id[self.speaker],
                        'betas': self.betas
                    }
                    return data_sample
            def __len__(child):
                return len(child.index_list)

        if split == 'train':
            index_list = list(
                range(0, min(self.complete_data.shape[0], self.audio_feat.shape[0]) - self.num_generate_length - self.num_pre_frames,
                      6))
        elif split in ['val', 'test']:
            index_list = list([0])
        if self.whole_video:
            index_list = list([0])
        self.all_dataset = __Worker__(index_list, normalization, normalize_stats, split)

    def __len__(self):
        return len(self.img_name_list)


