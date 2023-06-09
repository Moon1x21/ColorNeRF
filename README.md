# CorolNeRF

## Network architecture

<p align="center">
   <img src="https://github.com/Moon1x21/ColorNeRF/assets/62733294/29933850-e860-4920-8c0b-d22f845c7fc8">
</p>

# :computer: Installation

## Hardware

* OS: Ubuntu 18.04
* NVIDIA GPU with **CUDA>=10.2** 

## Software

* Clone this repo by `git clone https://github.com/Moon1x21/ColorNeRF.git`
* Python>=3.6 (installation via [anaconda](https://www.anaconda.com/distribution/) is recommended, use `conda create -n cnerf python=3.6` to create a conda environment and activate it by `conda activate cnerf`)
* Python libraries
    * Install core requirements by `pip install -r requirements.txt`
    
# :key: Training

## Blender

<details>
  <summary>Steps</summary>
   
### Data download

Download `nerf_synthetic.zip` from [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

### Data perturbations

All random seeds are fixed to reproduce the same perturbations every time.
For detailed implementation, see [blender.py](datasets/blender.py).

*  Color perturbations: Uses the same parameters in the paper.

![color](https://user-images.githubusercontent.com/11364490/105580035-4ad3b780-5dcd-11eb-97cc-4cea3c9743ac.gif)


### Training model

Base:
```
python train.py \
   --dataset_name blender \
   --root_dir $BLENDER_DIR \
   --N_importance 64 --img_wh 400 400 --noise_std 0 \
   --num_epochs 10 --batch_size 1024 \
   --optimizer adam --lr 5e-4 --lr_scheduler cosine \
   --encode_a --encode_t --exp_name exp 
```

Add `--encode_a` for appearance embedding, `--encode_t` for uncertainty

Add `--data_perturb color` to perturb the dataset.

Example:
```
python train.py \
   --dataset_name blender \
   --root_dir $BLENDER_DIR \
   --N_importance 64 --img_wh 400 400 --noise_std 0 \
   --num_epochs 10 --batch_size 1024 \
   --optimizer adam --lr 5e-4 --lr_scheduler cosine \
   --exp_name exp \
   --data_perturb occ \
   --encode_t --encode_a --beta_min 0.1
```

</details>

## Phototourism dataset

<details>
  <summary>Steps</summary>

### Data download

Download the scenes you want from [here](https://www.cs.ubc.ca/~kmyi/imw2020/data.html) (train/test splits are only provided for "Brandenburg Gate", "Sacre Coeur" and "Trevi Fountain", if you want to train on other scenes, you need to clean the data (Section C) and split the data by yourself)

Download the train/test split from the "Additional links" [here](https://nerf-w.github.io/) and put under each scene's folder (the **same level** as the "dense" folder)

(Optional but **highly** recommended) Run `python prepare_phototourism.py --root_dir $ROOT_DIR --img_downscale {an integer, e.g. 2 means half the image sizes}` to prepare the training data and save to disk first, if you want to run multiple experiments or run on multiple gpus. This will **largely** reduce the data preparation step before training.

### Data visualization (Optional)

Take a look at [phototourism_visualization.ipynb](https://nbviewer.jupyter.org/github/kwea123/nerf_pl/blob/nerfw/phototourism_visualization.ipynb), a quick visualization of the data: scene geometry, camera poses, rays and bounds, to assure you that my data convertion works correctly.

### Training model

Run (example)

```
python train.py \
  --root_dir /home/ubuntu/data/IMC-PT/brandenburg_gate/ --dataset_name phototourism \
  --img_downscale 8 --use_cache --N_importance 64 --N_samples 64 \
  --encode_a --encode_t --beta_min 0.03 --N_vocab 1500 \
  --num_epochs 10 --batch_size 1024 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name brandenburg_scale8_cnerf
```


`--N_vocab` should be set to an integer larger than the number of images (dependent on different scenes). For example, "brandenburg_gate" has in total 1363 images (under `dense/images/`), so any number larger than 1363 works (no need to set to exactly the same number). **Attention!** If you forget to set this number, or it is set smaller than the number of images, the program will yield `RuntimeError: CUDA error: device-side assert triggered` (which comes from `torch.nn.Embedding`).

</details>

## NeRF-W

If you want to train original NeRF-W for comparing ColorNeRF, using train_w.py file.

## Pretrained models and logs
Download the pretrained models and training logs in [release]().

# :mag_right: Testing

Use [eval.py](eval.py) to create the whole sequence of moving views.
It will create folder `results/{dataset_name}/{exp_name}` and run inference on all test data, finally create a gif out of them.

## Blender result

All the experiments are trained for 10 epochs.

1.  Result shows ColorNeRF is able to handle color variation.

<p align="center">
  <img src="https://user-images.githubusercontent.com/11364490/105775080-8d51eb80-5fa9-11eb-9e89-7147c6377453.gif">
  <img src="https://user-images.githubusercontent.com/11364490/105630746-43c0ae00-5e8e-11eb-856a-e6ce7ac8c16f.gif">
  <br>
  Left: NeRF, PSNR=18.83 (paper=15.73). Right: <a href=https://github.com/kwea123/nerf_pl/releases/tag/nerfw_all>pretrained</a> <b>NeRF-W</b>, PSNR=<b>24.86</b> (paper=22.19).
</p>

2. Reference: Original NeRF (without `--encode_a` and `--encode_t`) trained on perturbed data. As the results show original NeRF is not able to handle the color perturbation as the network has the precondition that the color is static.

<p align="center">
   <img src="https://user-images.githubusercontent.com/11364490/105649082-0e4dac00-5ef2-11eb-9d56-946e2ac068c4.gif">
   <br>
   PSNR= 
</p>

3. Reference: NeRF-w (with `--encode_a` and `--encode_t`) trained on perturbed data. Even though the NeRF-W is able to handle the color perturbation, there are still some details are missing.

<p align="center">
   <img src="https://user-images.githubusercontent.com/11364490/105649082-0e4dac00-5ef2-11eb-9d56-946e2ac068c4.gif">
   <br>
   PSNR= 
</p>

## Reference
This code is based on Unofficial implementation of [NeRF-W](https://github.com/kwea123/nerf_pl) 
'''
@inproceedings{martinbrualla2020nerfw,
                          author = {Martin-Brualla, Ricardo
                                    and Radwan, Noha
                                    and Sajjadi, Mehdi S. M.
                                    and Barron, Jonathan T.
                                    and Dosovitskiy, Alexey
                                    and Duckworth, Daniel},
                          title = {{NeRF in the Wild: Neural Radiance Fields for
                                  Unconstrained Photo Collections}},
                          booktitle = {CVPR},
                          year={2021}
                      }
'''
'''
@article{mildenhall2021nerf,
  title={Nerf: Representing scenes as neural radiance fields for view synthesis},
  author={Mildenhall, Ben and Srinivasan, Pratul P and Tancik, Matthew and Barron, Jonathan T and Ramamoorthi, Ravi and Ng, Ren},
  journal={Communications of the ACM},
  volume={65},
  number={1},
  pages={99--106},
  year={2021},
  publisher={ACM New York, NY, USA}
}
'''