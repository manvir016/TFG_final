python -W ignore scripts/diversity.py \
--save_dir experiments \
--exp_name smplx_S2G \
--speakers oliver seth conan chemistry \
--config_file ./config/body_pixel.json \
--face_model_path ./experiments/2025-05-11-smplx_S2G-face/ckpt-99.pth \
--body_model_path ./experiments/2025-05-11-smplx_S2G-body-pixel2/ckpt-99.pth \
--infer
