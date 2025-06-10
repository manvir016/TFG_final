python -W ignore scripts/test_body.py \
--save_dir experiments \
--exp_name smplx_S2G \
--speakers oliver seth conan chemistry \
--config_file ./config/body_pixel.json \
--body_model_name s2g_body_pixel \
--body_model_path ./experiments/2025-05-08-smplx_S2G-body-pixel2/ckpt-99.pth \
--infer
