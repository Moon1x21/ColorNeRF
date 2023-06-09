python train.py \
   --dataset_name blender \
   --root_dir ./data/nerf_synthetic/materials \
   --N_importance 64 --img_wh 400 400 --noise_std 0 \
   --num_epochs 20 --batch_size 1024 \
   --optimizer adam --lr 5e-4 --lr_scheduler cosine \
   --exp_name blender_mat_mine \
   --data_perturb color \
   --num_epochs 10 \
   --encode_a --encode_t --beta_min 0.1
