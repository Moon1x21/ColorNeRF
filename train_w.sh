python train_w.py \
  --dataset_name blender \
  --root_dir ./data/nerf_synthetic/lego \
  --N_importance 64 --img_wh 400 400 --noise_std 0 \
  --num_epochs 10 --batch_size 1024 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name blender_nerf_w \
  --data_perturb color \
  --num_epochs 10 \
  --encode_a --encode_t --beta_min 0.1 --num_gpus 1
