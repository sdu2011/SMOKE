# SMOKE: Single-Stage Monocular 3D Object Detection via Keypoint Estimation

<img align="center" src="figures/animation.gif" width="750">

[Video](https://www.youtube.com/watch?v=pvM_bASOQmo)

This repository is the official implementation of our paper [SMOKE: Single-Stage Monocular 3D Object Detection via Keypoint Estimation](https://arxiv.org/pdf/2002.10111.pdf).
For more details, please see our paper.

## Introduction
SMOKE is a **real-time** monocular 3D object detector for autonomous driving. 
The runtime on a single NVIDIA TITAN XP GPU is **~30ms**. 
Part of the code comes from [CenterNet](https://github.com/xingyizhou/CenterNet), 
[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark),
and [Detectron2](https://github.com/facebookresearch/detectron2).

The performance on KITTI 3D detection (3D/BEV) is as follows:

|             |     Easy      |    Moderate    |     Hard     |
|-------------|:-------------:|:--------------:|:------------:|
| Car         | 14.17 / 21.08 | 9.88 / 15.13   | 8.63 / 12.91 | 
| Pedestrian  | 5.16  / 6.22  | 3.24 / 4.05    | 2.53 / 3.38  | 
| Cyclist     | 1.11  / 1.62  | 0.60 / 0.98    | 0.47 / 0.74  |

The pretrained weights can be downloaded [here](https://drive.google.com/open?id=11VK8_HfR7t0wm-6dCNP5KS3Vh-Qm686-).

## Requirements
All codes are tested under the following environment:
*   Ubuntu 16.04
*   Python 3.7
*   Pytorch 1.3.1
*   CUDA 10.0

## Dataset
We train and test our model on official [KITTI 3D Object Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). 
Please first download the dataset and organize it as following structure:
```
kitti
│──training
│    ├──calib 
│    ├──label_2 
│    ├──image_2
│    └──ImageSets
└──testing
     ├──calib 
     ├──image_2
     └──ImageSets
```  

## Setup
1. We use `conda` to manage the environment:
```
conda create -n SMOKE python=3.7
```

2. Clone this repo:
```
git clone https://github.com/lzccccc/SMOKE
```

3. Build codes:
```
python setup.py build develop
```

4. Link to dataset directory:
```
mkdir datasets
ln -s /path_to_kitti_dataset datasets/kitti
```

## Getting started
First check the config file under `configs/`. 

We train the model on 4 GPUs with 32 batch size:
```
python tools/plain_train_net.py --num-gpus 4 --config-file "configs/smoke_gn_vector.yaml"
```

For single GPU training, simply run:
```
python tools/plain_train_net.py --config-file "configs/smoke_gn_vector.yaml"
```

We currently only support single GPU testing:
```
python tools/plain_train_net.py --eval-only --config-file "configs/smoke_gn_vector.yaml"
```

## Acknowledgement
[CenterNet](https://github.com/xingyizhou/CenterNet)

[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)

[Detectron2](https://github.com/facebookresearch/detectron2)


## Citations
Please cite our paper if you find SMOKE is helpful for your research.
```
@article{liu2020SMOKE,
  title={{SMOKE}: Single-Stage Monocular 3D Object Detection via Keypoint Estimation},
  author={Zechen Liu and Zizhang Wu and Roland T\'oth},
  journal={arXiv preprint arXiv:2002.10111},
  year={2020}
}
```


----
安装:
pip install Pillow
pip install yacs
pip install scikit-image
pip install torchvision==0.4.2
pip install torch==1.3.1
pip install tqdm

torch和torchvision的版本对应关系.
```
torch	torchvision	python
master / nightly	master / nightly	>=3.6
1.6.0	0.7.0	>=3.6
1.5.1	0.6.1	>=3.5
1.5.0	0.6.0	>=3.5
1.4.0	0.5.0	==2.7, >=3.5, <=3.8
1.3.1	0.4.2	==2.7, >=3.5, <=3.7
1.3.0	0.4.1	==2.7, >=3.5, <=3.7
1.2.0	0.4.0	==2.7, >=3.5, <=3.7
1.1.0	0.3.0	==2.7, >=3.5, <=3.7
<=1.0.1	0.2.2	==2.7, >=3.5, <=3.7
```

在项目根目录执行:
生成训练/测试集描述文件.
python scripts/gen_train.py
python scripts/gen_test.py


训练:
python tools/plain_train_net.py --config-file "configs/smoke_gn_vector.yaml"

推理:
python tools/plain_train_net.py --eval-only --config-file "configs/smoke_gn_vector.yaml"


bug修复:
1.  RuntimeError: received 0 items of ancdata
https://www.cnblogs.com/zhengbiqing/p/10478311.html

2. torch_shm_manager: error while loading shared libraries: libcudart.so.10.1: cannot open shared object file: No such file or directory
    1. ~~确认torch支持的cuda版本~~
```
Python 3.7.9 (default, Aug 31 2020, 12:42:55) 
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.version.cuda
'10.1.243'
>>> 

```
    2. 安装匹配版本的cuda


---
config/path_catalog.py定义了数据集的路径信息



---
kitti数据集含义:
- calib文件
<https://stackoverflow.com/questions/29407474/how-to-understand-the-kitti-camera-calibration-files>
<https://zhuanlan.zhihu.com/p/99114433>
<https://medium.com/test-ttile/kitti-3d-object-detection-dataset-d78a762b5a4>

可以参考scripts/some_test.py中的代码打印理解.

**0 represents the left grayscale, 1 the right grayscale, 2 the left color and 3 the right color camera.**
PX是一个3x4矩阵.代码camera x的内参矩阵.最后一列的参数貌似是为了畸变校正用的(不确定,因为cv一般校准出来的畸变矩阵是5个参数),这个训练代码里也只是用了这个3x4矩阵的前三列.

- annotation
<https://github.com/bostondiditeam/kitti/blob/master/resources/devkit_object/readme.txt>

---
kitti.py:加载kitti数据集.


## gaussian_radius
讲的很详细很清楚:<https://zhuanlan.zhihu.com/p/96856635> 

总结一下就是:**iou通过保证预测的框的lt,rb两个点在GT BOX的lt,rb点的以r为半径的圆内即可.**