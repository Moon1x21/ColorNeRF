python eval.py \
  --root_dir ./data/brandenburg_gate/ \
  --dataset_name phototourism --scene_name brandenburg_test \
  --split test --N_samples 256 --N_importance 256 \
  --N_vocab 1500 --encode_a --encode_t \
  --ckpt_path ckpts/brandenburg_scale2/epoch\=19.ckpt \
  --chunk 16384 --img_wh 320 240 