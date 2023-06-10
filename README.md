# CorolNeRF

## Network architecture

<p align="center">
   <img src="https://github.com/Moon1x21/ColorNeRF/assets/62733294/29933850-e860-4920-8c0b-d22f845c7fc8">
</p>

## Overall research

Recently, the Neural Radiance Fields (NeRF) shows remarkable progresses in learing to synthesize novel view of 3D objects and scenes. Although, NeRF is able to represent the 3D object and scenes, the results of the network is higly depended on the setting conditions. As the NeRF has a precondition that the color of the object is static, the training datasets have to be acquisited in controlled settings. However, in our real life, the color of the objects or scenes has variation in captured images due to the dynamic environment. When using those images, NeRF is unable to synthesize realistic scene. 

To handle this problem, this research presents **ColorNeRF** that can represent the realistic results even in using the images acquisited in uncontrolled environment. In this research, the network is trained the color perturabtion per-image using embedding vector. Through this, the network can handle the color variation and implicitly learn the actual color of the scene. Secondly, this research added the uncertainty value to alleviate the uncertainty of each color.

<details>
  <summary>연구 배경</summary>

## Introduction
현재 View synthesis는 많은 발전을 이루었다. 특히 Neural rendering 방식을 이용하여 더욱 발전된 성과를 보여주었다. 대표적인 Neural Radiance Fields (NeRF)는 실제로
정교한 View synthesis 결과를 보여주고 있다. 하지만, 이러한 좋은 결과를 보여주는 연구에서도 아직 해결해야 할 부분은 존재한다. 현재
NeRF는 제한된 환경에서 얻은 이미지를 이용하여 학습을 진행한다. 만약, 제한된 환경에서 얻은 이미지가 아닌, 실제 환경에서 얻은 이미지들을 사용하게 된다면 낮은 성능을 보여주게
된다. 이는 NeRF 의 전제 조건에 의해 발생하게 되는데, 기존의 NeRF 는 학습에서 사용되는 이미지의 색상 정보와 기하학적인 정보는 모든 이미지에 대해 불변하는 정보라고 가정하고
학습을 진행한다. 하지만, 이는 실생활에서 사용하기에는 적절하지 못한 전제 조건이다. 왜냐하면 동일한 위치에서 동일한 물체를 찍게 되어도, 시간의 변화나 날씨 등에 의해 다른 광량을 가지게
되고, 이로 인해 동일한 위치에서 찍은 이미지라고 하더라도 색상의 변화가 존재하기 때문이다. 또한, 주변에 존재하는 물체가 특정 색상을 지닌 경우에, 해당 물체가 특정 색상을 지닌 조명과
같은 역할을 하여 이미지에 찍힌 물체는 밝기 변화 뿐만 아니라 특정 색상의 변화 또한 존재하게 된다. 따라서 NeRF 는 이러한 이미지를 이용하여 학습할 경우 제대로 학습이 진행되지 않고,
기존의 성능보다 훨씬 떨어지는 성능을 보여주게 된다. 만약 이러한 경우, 제한되지 않는 상황에서 얻은 이미지들을 전처리하여 기존 모델에 학습을 진행하는 방안이 존재하지만, 이때
우리는 어떠한 이미지를 기준으로 하여 이미지를 전처리 해야할지 정해야 하고, 만약 다양한 조건에서의 물체를 Synthesis 하길 원하는 경우 각각의 조건마다 이에 맞게끔 모두 전처리를
하여 모델에 넣어 학습을 진행해야 하는 문제점이 발생한다. 이러한 경우에는 각 상황에 따라 모델을 다시 학습해야 하는 문제점이 존재하게 되는데, 이를 해결하여 다양한 상황에서 얻은
이미지를 한번에 처리하여, 우리가 원하는 조건을 선택했을 때 해당 조건에 맞는 물체를 Synthesis 할 수 있는 모델을 구현할 수 있다면, 더욱 효과적일 것이라고 생각이 되었다.

따라서 이러한 문제점을 해결하기 위해 여러 네트워크들 제안되었는데, NeRF-W는 Image dependent 한 Embedding vector 를 이용하여 동일 지점이라도 다른 색상 정보를 학습할 수
있도록 구현하였다. 하지만, 해당 네트워크에서 사용되는 Embedding vector 정보 또한, 학습을 통해 진행되기 때문에, 완벽한 신뢰도를 가지기 힘들다는 단점이 있다. 따라서 이러한 이슈를 해결하기 위해 Sparse 한 input 을 이용하면서도 더욱 정교하고, Unseen view 를 표현해낼 수 있는 Method 에 대한 연구를 진행하게 되었다.
</details>

<details>
  <summary>관련 연구</summary>

## 1. View synthesis
이미지 기반 렌더링이라고도 하는 View synthesis 은 제한된 수의 입력 이미지에서 장면 또는 개체의 새로운 scene 을 만드는 데 중점을 두는 연구 영역이다. 이는 일반적으로 컴퓨터
비전 기술과 알고리즘을 사용하여 진행한다. View synthesis 에 대한 일반적인 접근 방식 중 하나는 Stereo 또는 Multi-view 이미지를 사용하여 서로 다른 관점에서 동일한 장면을 캡처하는 것이다. Viewpoint 간의 차이를 분석함으로써 해당 view 의 Depth 정보를 추정하고 새로운 viewpoint 에 대한 이미지를 생성할 수 있다. 또 다른 접근 방식은 심층 신경망과 같은 기계 학습 기술을 사용하여 입력 이미지 세트에서 새로운 보기를 생성하는 방법을 배우는 것이다.
View synthesis 연구의 주요 과제 중 하나는 고품질의 사실적인 이미지를 생성하는 것이다. 이를 위해서는 장면 형상 및 모양의 정확한 추정뿐만 아니라 합성된 이미지를 렌더링하고 합성하기
위한 정교한 기술이 필요합니다. 또 다른 과제는 입력 이미지의 occlusion 부분과 blurry 한 부분을 처리하는 것으로, 장면 형상 및 모양을 정확하게 추정하기 어려울 수 있다.

## 2. Neural Radiance Fields
NeRF는 Novel View Synthesis 분야에서, Point Cloud 나 Mesh, Voxel 등으로 표현되는 3D Object 자체를 렌더링하는 것이 아닌, 3D Object 를 바라본 모습(이미지)들을 예측할 수 있는
모델을 만드는 것이 목표이다. 3D object 자체를 represent 하지 않아도, 해당 Object 를 represent 할 때의 계산 방법을 아는 것 또한 3D object 를 represent 할 수 있다고 할 수 있다. 
NeRF는 여러 개의 Input 이미지와 그에 해당되는 100 개의 Camera transpose (translation + rotation) 값들을 사용한다. 학습을 진행할 때에, 각 Camera ray 와 해당 Camera ray 를 지나는 3D 상에서의 point 를 샘플링한다. 샘플링한 3D point 와 Camera ray 를 Network input 으로 사용하여 해당 3D point 의 color 값과 Density 를 도출한다. 이때, Density 는 해당 물체가 존재할 확률을 표현한다. 이후 각 샘플링된 3D point 들의 Color 와 Density 를 사용하여 rendering 공식을 이용하여 최종 Camera 에서 보여지는 색을 계산한 후, image 상의 색과 비교를 진행하여 Loss 를 구해 학습한다. NeRF 사용된 Rendering 공식과 Loss 함수는 다음과 같다

<p align="center">
   <img src="https://github.com/Moon1x21/ColorNeRF/assets/62733294/a79bea89-374f-46cf-a41c-a497456bc9c6">
</p>

## 3. NeRF in the Wild (NeRF-W)
기존의 NeRF 는 실생활에서 얻어지는 이미지를 이용하여 학습을 진행하게 되면, 제대로 된 결과를 도출하지 못한다는 한계점이 존재한다. NeRF-W는 이를 해결하기 위해 NeRF 가
학습 시에 색상과 형상 정보의 불변하다고 가정하는 점을 문제 삼고, 이를 개선하여 다양한 환경에서 얻은 이미지에서도 균일한 View synthesis 결과를 얻을 수 있게 하였다. 우선 NeRF-W는 색상 정보가 이미지에 따라 다르게 구해질 수 있다는 점을 해결하기 위해, 이미지 별로 Appearence embedding vector 를 학습시에 추가하여 Image-dependent 한 색상 정보를 얻을 수
있도록 구현하였다. Density 에 해당하는 네트워크와 Color 정보를 얻는 네트워크를 구분하고, Color 정보를 얻는 네트워크에 Embedding vector 를 추가하므로써, 이미지 별 색상정보를 얻을
뿐만 아니라, 이미지에 독립적인 형상을 얻을 수 있도록 하였다. 색상 정보뿐만 아니라, NeRF-W는 인터넷이나 실생활에서 얻은 이미지는 다양한 장애물에 의해 가려지게 되는 상황이 많이 발생하는데, 이러한 이미지를 사용하더라도 균일한 결과를 얻고자 하였다. 따라서, Transient object 를 위해 MLP 를 추가하고, Uncertainty 값을 추가로 도출하여 해당 문제를 해결하고자 하였다. Transient Object MLP 에서 얻은 색상 및 Density 정보와, Static MLP 에서 얻은 색상 및 Density 정보를 이용하여 최종 색상을 랜더링하여 값을 얻게 된다.
추가로, 손실함수를 계산할 때, 베이지안 학습 프레임워크를 이용하여, 얻은 색상을 평균, 최종 Uncertainty 값을 분산으로 두어 가우시안 분포를 생성하고, 이때 기존 색상 정보가 나오게 될
확률을 이용하여 Negative log likelihood 를 사용해 손실함수를 만들었다. 이때, 최종 Uncertainty 값은 각 3 차원 포인트에 해당하는 Uncertainty 값들을 랜더링한 값을 이용하여 사용하였다.

<p align="center">
   <img src="https://github.com/Moon1x21/ColorNeRF/assets/62733294/abb56d37-b6f6-44e1-8577-d78048337680">
</p>

</details>

<details>
  <summary>제안 방안</summary>

## Method
다음은 임베딩 벡터의 불안정성을 완화하며 다양한 이미지를 사용하여 균일한 View synthesis 를 하기 위해 고안해 낸 새로운 알고리즘에 대해 소개한다. 
우선, 이미지 상에서 색상 변동에 대한 정의를 먼저 내려야 한다. 기존의 외부환경에서 얻은 이미지를 기준으로 원활한 View synthesis 가 가능하기 위해서는 색상의 차이가 밝기 차이만
존재하는지, 아니면 색상 자체의 차이 또한 존재하는지 알아보아야 한다. 외부환경에서 얻은 이미지의 색사이 변화하는 경우 대부분 밝기 변화량이 주된 변화인데, RGB 자체에 대한 변화량은
주변에 존재하는 물체가 조명과 같은 역할을 하여 색상을 변경하는 경우가 존재할 수 있다고 판단하였다. 따라서 밝기 및 색상 자체에서도 균일한 결과를 도출하는 네트워크를 구현하고자 함.
기존의 방식은 좌표값을 통하여 바로 해당 위치의 색상 정보를 얻게 된다. 하지만, 기존의 NeRF 는 색상 정보의 변동에 대한 계산을 하지 않기 때문에, 물체의 기존 색상 정보를
학습하는데에 더욱 오랜 시간이 걸리게 된다. 또한, NeRF-W 의 경우 임베딩 벡터를 이용하여 색상 정보를 바로 도출하는데, 이는 색상 정보들의 변동에 대한 정보와 색상 자체의 정보가 모두
하나의 모델에서 계산하여 도출되기 때문에, 오히려 좋지 못한 성능이 도출 될 것이라고 생각하였다. 추가적으로 하나의 네트워크에서 임베딩 벡터 또한 학습을 하기 때문에, 하나의
네트워크에서 세가지의 불안정한 값이 존재하기 때문에, 네트워크를 학습할 때에 있어서 불안정한 값을 조금이나마 줄이고자 하였다. 따라서, 처음 임베딩 벡터, 좌표값 그리고 카메라
방향을 통하여 해당 위치에 대한 색상 변동값과 밀도값을 도출하도록 모델을 변경하였다. 이를 통하여, 색상 정보에 대해서 두 개의 불안전한 값을 모두 처리하는 것이 아닌, 색상에 대한
변동값만을 도출하여 모델의 불안정성을 줄였다. 이후, 각 색상당 변동값과 위치 정보를 입력받아 최종 색상과 해당 색상의 불확실성에 대한 값을 도출해내는 모델을 추가적으로 만들었다. 이를 통하여 최종적으로 얻어낸 각 위치에 해당하는 색상값을 랜더링 수식을 통해 최종 이차원 이미지 상에서 보이는 색상을 얻고, 이를 기존에 가지고 있는 이미지와 비교를 하여 손실함수를 구현하였다. 또한, 추가적으로 확실성에 대한 값을 추가하였는데, 이는 각 색상에 대한 불안정성을 조금이나마 낮추기 위하여 얻은 값으로써, 이를 통해 확실하지 않은 위치에 대한 신뢰도를 줄여 오류를 최소화 하고자 하였다. 이를 통하여 확실하지 않은 물체의 색상으로 인해 생기는 손실을 더욱 줄일 수 있다. 이미지의 색상 변동량과 최종 이미지 색상을 두 모델을 이용하여 학습하는 방안을 고안해낸
이유는 기존의 연구결과 중, 동적인 물체를 렌더링을 하기 위해 고안해낸 모델에서 사용한 방식을 차용하였다. 해당 논문이 주장하는 바로, 위치정보의 변동값과 최종 색상을 동일한
네트워크에서 계산하게 된다면, 색상에 해당하는 공간과 좌표 이동에 해당하는 공간, 이 두 공간이 하나의 네트워크에서 계산되기 때문에, 성능이 저하되었다고 한다. 이와 동일하게, 각
이미지의 임베딩 공간과 좌표를 이용하여 색상 변동량에 해당하는 공간으로 변환을 해주고, 이를 이용하여 최종 색상을 얻어내는 것이 더욱 좋은 성능을 보일 것이라고 생각하였기 때문에 해당
모델을 구현하게 되었다. 구현해낸 모델의 전반적인 구조는 상단의 그림과 같으며, 네트워크 구조에서 쓰여있는 Noise 는 이미지 변동량에 해당하는 값으로써, 각 RGB 값에 해당하는 색상 변동량을 나타낸다.
</details>

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


## Phototourism dataset

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

## NeRF-W

If you want to train original NeRF-W for comparing ColorNeRF, using train_w.py file.

## Pretrained models and logs
Download the pretrained models and training logs in [release]().

# :mag_right: Testing

Use [eval.py](eval.py) to create the whole sequence of moving views.
It will create folder `results/{dataset_name}/{exp_name}` and run inference on all test data, finally create a gif out of them.

## Blender result

All the experiments are trained for 10 epochs.

### Lego

1.  Result shows **ColorNeRF** is able to handle color variation. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/11364490/105775080-8d51eb80-5fa9-11eb-9e89-7147c6377453.gif">
  <br>
  PSNR = 28.59
</p>

2. **NeRF** trained on perturbed data. As the results show original NeRF is not able to handle the color perturbation as the network has the precondition that the color is static.

<p align="center">
   <img src="https://user-images.githubusercontent.com/11364490/105649082-0e4dac00-5ef2-11eb-9d56-946e2ac068c4.gif">
   <br>
   PSNR = 24.67
</p>

3. **NeRF-W** trained on perturbed data. Even though the NeRF-W is able to handle the color perturbation, there are still some details are missing.

<p align="center">
   <img src="https://github.com/Moon1x21/ColorNeRF/assets/62733294/0e1e25e2-3688-4382-a989-a66f34974f4d">
   <br>
   PSNR = 27.73
</p>

### Hotdog

1.  Result shows **ColorNeRF** is able to handle color variation. 

<p align="center">
  <img src="https://github.com/Moon1x21/ColorNeRF/assets/62733294/c63931bf-7b38-45f0-b204-60c84dcc4fab">
  <br>
  PSNR = 31.86
</p>

2. **NeRF** trained on perturbed data. As the results show original NeRF is not able to handle the color perturbation as the network has the precondition that the color is static.

<p align="center">
   <img src="https://user-images.githubusercontent.com/11364490/105649082-0e4dac00-5ef2-11eb-9d56-946e2ac068c4.gif">
   <br>
   PSNR = 31.52
</p>

3. **NeRF-W** trained on perturbed data. Even though the NeRF-W is able to handle the color perturbation, there are still some details are missing.

<p align="center">
   <img src="https://github.com/Moon1x21/ColorNeRF/assets/62733294/71778823-e3de-4c82-b5b2-a97151267684">
   <br>
   PSNR = 26.77
</p>

## Reference
This code is based on Unofficial implementation of [NeRF-W](https://github.com/kwea123/nerf_pl) 

```
@inproceedings{martinbrualla2020nerfw,
                          author = {Martin-Brualla, Ricardo
                                    and Radwan, Noha
                                    and Sajjadi, Mehdi S. M.
                                    and Barron, Jonathan T.
                                    and Dosovitskiy, Alexey
                                    and Duckworth, Daniel},
                          title = {{NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections}},
                          booktitle = {CVPR},
                          year={2021}
                      }
```


```
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
```