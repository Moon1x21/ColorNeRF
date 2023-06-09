python eval_w.py \
   --dataset_name blender \
   --root_dir ./data/nerf_synthetic/hotdog \
   --N_importance 64 --img_wh 400 400 \
   --ckpt_path /home/d9/Documents/nerf_pl/ckpts/blender_nerf_w_hotdog/epoch=9.ckpt \
   --encode_a --encode_t --scene_name blender_hotdog_nerf_w