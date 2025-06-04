python -W ignore scripts/train.py \
--save_dir experiments \
--exp_name smplx_S2G \
--speakers oliver seth conan chemistry \
--config_file ./config/face.json
--resume \
--pretrained_pth ./experiments/2025-05-27-smplx_S2G-face/ckpt-99.pth 
