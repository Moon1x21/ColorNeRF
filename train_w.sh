python train_w.py \
  --root_dir ./data/sacre_coeur/ --dataset_name phototourism \
  --img_downscale 2 --use_cache --N_importance 64 --N_samples 64 \
   --N_vocab 1500 \
  --num_epochs 10 --batch_size 1024 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --root_dir ./data/sacre_coeur/ --dataset_name phototourism \
  --exp_name sacre_coeur_scale2_nerf_w  --num_gpus 2