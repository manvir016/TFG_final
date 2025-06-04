import os
import sys
# os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(os.getcwd())

from transformers import Wav2Vec2Processor
from glob import glob

import numpy as np
import json

from nets import *
from trainer.options import parse_args
from data_utils import torch_data
from trainer.config import load_JsonConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from data_utils.rotation_conversion import rotation_6d_to_matrix, matrix_to_axis_angle
from data_utils.lower_body import part2full, pred2poses, poses2pred, poses2poses
from visualise.rendering import RenderTool



import smplx
import argparse



def load_model(config, model_path, model_class):
    model = model_class(config)
    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt['generator'])
    model.eval()
    return model


def get_vertices(smplx_model, poses, betas):
    vertices = []
    for i in range(poses.shape[0]):
        pose = poses[i]
        output = smplx_model(
            betas=betas,
            expression=pose[165:265].unsqueeze(0),
            jaw_pose=pose[0:3].unsqueeze(0),
            leye_pose=pose[3:6].unsqueeze(0),
            reye_pose=pose[6:9].unsqueeze(0),
            global_orient=pose[9:12].unsqueeze(0),
            body_pose=pose[12:75].unsqueeze(0),
            left_hand_pose=pose[75:120].unsqueeze(0),
            right_hand_pose=pose[120:165].unsqueeze(0),
            return_verts=True
        )
        vertices.append(output.vertices.detach().cpu().numpy().squeeze())
    return np.asarray(vertices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy_file', required=True, help='Ruta al archivo .npy de predicciones')
    parser.add_argument('--save_dir', required=True, help='Directorio donde guardar los frames')
    args = parser.parse_args()

    # Cargar predicciones (T, C)
    pred = np.load(args.npy_file)  # pred.shape = (T, C)
    pred = torch.tensor(pred, dtype=torch.float32)

    # Procesar rotaciones si está en 6D
    if pred.shape[1] == 330 + 100:
        rot = pred[:, :330].reshape(-1, 6)
        rot_mat = rotation_6d_to_matrix(rot).reshape(pred.shape[0], -1, 3, 3)
        axis_angle = matrix_to_axis_angle(rot_mat).reshape(pred.shape[0], -1)
        pred = torch.cat([axis_angle, pred[:, 330:]], dim=-1)

    # Transformar a full-body (por si acaso)
    pred = part2full(pred, stand=False)

    # Configurar SMPL-X
    smplx_model = smplx.create(
        model_path='visualise',
        model_type='smplx',
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        num_betas=300,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        use_pca=False,
        flat_hand_mean=False,
        create_expression=True,
        num_expression_coeffs=100,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=False,
        dtype=torch.float64
    ).to('cpu')

    # Dummy betas
    betas = torch.zeros([1, 300], dtype=torch.float64)

    # Calcular vértices
    vertices = get_vertices(smplx_model, pred, betas)

    # Renderizar
    os.makedirs(args.save_dir, exist_ok=True)
    rendertool = RenderTool(args.save_dir)
    rendertool._render_frames(vertices, args.save_dir)


if __name__ == '__main__':
    main()

