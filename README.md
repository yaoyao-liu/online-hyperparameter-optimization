# Online Hyperparameter Optimization for Class-Incremental Learning

[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/yaoyao-liu/online-hyperparameter-optimization/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg?style=flat-square&logo=python&color=3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.2.0-%237732a8?style=flat-square&logo=PyTorch&color=EE4C2C)](https://pytorch.org/)

[[Paper](https://pure.mpg.de/rest/items/item_3478882_1/component/file_3478883/content)] [[Project Page](https://class-il.mpi-inf.mpg.de/online-hyperparameter-optimization/)]

This repository contains the PyTorch implementation for the [AAAI 2023](https://aaai.org/Conferences/AAAI-23/) Paper ["Online Hyperparameter Optimization for Class-Incremental Learning"](https://pure.mpg.de/rest/items/item_3478882_1/component/file_3478883/content) by [Yaoyao Liu](https://people.mpi-inf.mpg.de/~yaliu/), [Yingying Li](https://yingying.li), [Bernt Schiele](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/people/bernt-schiele/), and [Qianru Sun](https://qianrusun1015.github.io). If you have any questions on this repository or the related paper, feel free to [create an issue](https://github.com/yaoyao-liu/online-hyperparameter-optimization/issues/new) or [send me an email](mailto:yliu538@jhu.edu).

### Getting Started

In order to run this repository, we advise you to install python 3.6 and PyTorch 1.2.0 with Anaconda.

You may download Anaconda and read the installation instruction on their official website:
<https://www.anaconda.com/download/>

Create a new environment and install PyTorch and torchvision on it:

```bash
conda create --name AANets-PyTorch python=3.6
conda activate AANets-PyTorch
conda install pytorch=1.2.0 
conda install torchvision -c pytorch
```
Then, you need to install the following packages using `pip`:
```
pip install tqdm scipy sklearn tensorboardX Pillow==6.2.2
```
Next, clone this repository and enter the folder `online-hyperparameter-optimization`:
```bash
git clone https://github.com/yaoyao-liu/online-hyperparameter-optimization.git
cd online-hyperparameter-optimization

```

### Download the Datasets
#### CIFAR-100
It will be downloaded automatically by `torchvision` when running the experiments.

#### ImageNet-Subset
We create the ImageNet-Subset following [LUCIR](https://github.com/hshustc/CVPR19_Incremental_Learning).
You may download the dataset using the following links:
- [Download from Google Drive](https://drive.google.com/file/d/1n5Xg7Iye_wkzVKc0MTBao5adhYSUlMCL/view?usp=sharing)
- [Download from 百度网盘](https://pan.baidu.com/s/1MnhITYKUI1i7aRBzsPrCSw) (提取码: 6uj5)

File information:
```
File name: ImageNet-Subset.tar
Size: 15.37 GB
MD5: ab2190e9dac15042a141561b9ba5d6e9
```
You need to untar the downloaded file, and put the folder `seed_1993_subset_100_imagenet` in the folder `data`.

Please note that the ImageNet-Subset is created from ImageNet. ImageNet is only allowed to be downloaded by researchers for non-commercial research and educational purposes. See the terms of ImageNet [here](https://image-net.org/download.php).

### Running Experiments
#### Running Experiments w/ [LUCIR](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.pdf) on CIFAR-100

```bash
python run_tfh_exp.py # Training from half
python run_tfs_exp.py # Training from scratch
```

We will update the code for other baselines later.

### Citations

Please cite our papers if they are helpful to your work:

```bibtex
@inproceedings{Liu2023Online,
  author       = {Liu, Yaoyao and
                  Li, Yingying and
                  Schiele, Bernt and
                  Sun, Qianru},
  title        = {Online Hyperparameter Optimization for Class-Incremental Learning},
  booktitle    = {Thirty-Seventh {AAAI} Conference on Artificial Intelligence},
  year         = {2023}
}
```
