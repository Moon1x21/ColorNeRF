python train.py \
  --root_dir ./data/sacre_coeur/ --dataset_name phototourism \
  --img_downscale 2 --use_cache --N_importance 64 --N_samples 64 \
   --N_vocab 1500 --encode_a --encode_t \
  --num_epochs 10 --batch_size 1024 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --root_dir ./data/sacre_coeur/ --dataset_name phototourism \
  --exp_name sacre_mine_1  --num_gpus 2