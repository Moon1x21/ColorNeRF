python eval.py \
   --dataset_name blender \
   --root_dir ./data/nerf_synthetic/materials \
   --N_importance 64 --img_wh 400 400 \
   --ckpt_path /home/d9/Documents/nerf_pl/ckpts/blender_mat_mine/epoch=9.ckpt \
   --encode_a --encode_t --exp_name blender_materials_mine