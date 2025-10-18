# SEGS-SLAM

<div align="center">

![SEGS-SLAM Logo](https://img.shields.io/badge/SEGS--SLAM-ICCV%202025-blue)
![License](https://img.shields.io/badge/License-GPL%203.0-green)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-blue)
![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-orange)

**Structure-enhanced 3D Gaussian Splatting SLAM with Appearance Embedding**

[🌐 Homepage](https://segs-slam.github.io/) | [📄 Paper](https://arxiv.org/abs/2501.05242) | [📝 知乎介绍](https://zhuanlan.zhihu.com/p/1922411865323045454/preview?comment=0&catalog=0)

</div>

## 📖 Table of Contents

- [🎯 Overview](#-overview)
- [📣 News](#-news)
- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🎮 Usage Examples](#-usage-examples)
- [🔬 Evaluation](#-evaluation)
- [📚 Citation](#citation)

## 🎯 Overview

SEGS-SLAM is a cutting-edge **Structure-enhanced 3D Gaussian Splatting SLAM** system that combines the power of traditional SLAM with modern neural rendering techniques. It leverages 3D Gaussian Splatting for high-quality scene reconstruction while maintaining robust tracking capabilities.

### 🌟 Key Features

- **🔄 Real-time SLAM**: Integrated ORB-SLAM3 for robust camera tracking
- **🎨 Neural Rendering**: 3D Gaussian Splatting for photorealistic scene reconstruction
- **🏗️ Structure Enhancement**: Improved geometric consistency and scene understanding
- **📱 Multi-sensor Support**: Monocular, stereo, and RGB-D camera support
- **⚡ GPU Acceleration**: CUDA-optimized rendering and optimization
- **🔧 Flexible Configuration**: Extensive parameter tuning for different scenarios

## 📣 News

### 📢 Current Status:
- **TODO**: 
  1. Clean code for easier reading, which will be a slow process
  2. A viewer
  3. Support and test for real-world RealSense camera, useful for robotics
- ✅ **[2025.08.10]** Evaluation code release
- ✅ **[2025.07.10]** Full paper upload
- ✅ **[2025.06.28]** Official code publish. Enjoying it 😊😊😊
- ✅ **[2025.06.28]** Brief introduction written at [SEGS-SLAM zhihu](https://zhuanlan.zhihu.com/p/1922411865323045454/preview?comment=0&catalog=0)
- ✅ **[2025.06.26]** Paper accepted by ICCV 2025 🎉🎉🎉
- ✅ **[2024.12.26]** Open-sourced executable files available at https://github.com/segs-slam/SEGS-SLAM for running without compilation

## 🚀 Quick Start

We provide two ways to run our algorithm:

- **Method 1: Use Docker for one-click environment setup (Recommended)**
- **Method 2: Manually configure the environment (Follow our guide step by step)**

### 2.1 Docker (Strongly Recommended)

#### Install Dependencies
You can install Docker [here](https://docs.docker.com/engine/install/)

Also add the [nvidia-docker](https://nvidia.github.io/nvidia-docker/) repository:
```bash
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```

Install the Nvidia container/docker toolkits:
```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit nvidia-docker2
sudo systemctl restart docker
```

#### Container Setup
Build the docker container:
```bash
git clone https://github.com/leaner-forever/SEGS-SLAM
cd SEGS-SLAM/docker
docker build -t SEGS-slam-image .
docker images
sudo docker run --gpus all -it --name segs-slam segs-slam-image /bin/bash
```

Then you can go to [📦 Installation](https://github.com/leaner-forever/SEGS-SLAM/blob/master/README.md#-installation)directly. 

### 2.2 Prerequisites

Install system dependencies:
```bash
sudo apt install libeigen3-dev libboost-all-dev libjsoncpp-dev libopengl-dev mesa-utils libglfw3-dev libglm-dev python3-pip python3-dev libjpeg-dev libpng-dev libtiff-dev curl zip libavcodec-dev libavformat-dev libswscale-dev libswresample-dev libssl-dev libflann-dev libusb-1.0-0-dev liblz4-dev libgtest-dev libopenni-dev libgdal-dev libosmesa6-dev libx11-dev qt5-qmake qtbase5-dev freeglut3-dev libpcap-dev 
```

#### System Requirements

| Dependencies                                                                               | Tested with                            |
| ------------------------------------------------------------------------------------------ | -------------------------------------- |
| **OS**                                                                                     | Ubuntu 20.04 LTS, Ubuntu 22.04 LTS     |
| **gcc**                                                                                    | 10.5.0, 11.4.0                         |
| **cmake**                                                                                  | 3.27.5, 3.22.1                         |
| **[CUDA](https://developer.nvidia.com/cuda-toolkit-archive)**                              | 11.8                                   |
| **[cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)**                                | 8.9.3, 8.7.0                           |
| **[OpenCV](https://opencv.org/releases)** (with opencv_contrib and CUDA)                   | 4.7.0, 4.8.0                           |
| **[LibTorch](https://pytorch.org/get-started/locally)**                                    | cxx11-abi-shared-with-deps-2.0.1+cu118 |
| **[TorchScatter](https://pypi.org/project/torch-scatter/#description)**                    | 2.1.2                                  |
| **[Intel® RealSense™ SDK 2.0](https://github.com/IntelRealSense/librealsense)** (optional) | Latest                                 |

#### Using LibTorch

If you do not have LibTorch installed in system search paths, add one line before `find_package(Torch REQUIRED)` in `CMakeLists.txt`:

**[Option 1] Conda**: If using Conda with compatible PyTorch:
```cmake
# [For Jatson Orin] To install Pytorch in Jatson developer kit:
# export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
# pip install --no-cache $TORCH_INSTALL

set(Torch_DIR /the_path_to_conda/python3.x/site-packages/torch/share/cmake/Torch)
```

**[Option 2] Download**: Download libtorch from [cu118](https://download.pytorch.org/libtorch/cu118) and extract:
```cmake
set(Torch_DIR /the_path_to_where_you_extracted_LibTorch/libtorch/share/cmake/Torch)
```

#### Using OpenCV with opencv_contrib and CUDA

For version 4.7.0, download from [OpenCV releases](https://github.com/opencv/opencv/releases) and [opencv_contrib](https://github.com/opencv/opencv_contrib/tags), then build:

```bash
cd ~/opencv
cd opencv-4.7.0/
mkdir build
cd build

# Build options used in our tests:
cmake -DCMAKE_BUILD_TYPE=RELEASE -DWITH_CUDA=ON -DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON -DWITH_NVCUVID=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.8 -DOPENCV_EXTRA_MODULES_PATH="../../opencv_contrib-4.7.0/modules" -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_JASPER=ON -DBUILD_CCALIB=ON -DBUILD_JPEG=ON -DWITH_FFMPEG=ON ..

make -j8
# NOTE: Compilation may stick at 99% during final linking - wait until completion

sudo make install
```

If installing to custom path, add `-DCMAKE_INSTALL_PREFIX=/your_preferred_path` and help SEGS-SLAM find OpenCV by adding to `CMakeLists.txt`:
```cmake
set(OpenCV_DIR /your_preferred_path/lib/cmake/opencv4)
```

#### Using TorchScatter

TorchScatter is for efficient scatter operations on sparse tensors. Visit [TorchScatter](https://pypi.org/project/torch-scatter/#description) for detailed instructions.

**Install**:
```bash
# Install torch_scatter-2.1.2
mkdir build 
cd build 
# Add -DCMAKE_PREFIX_PATH=<Torch dir> 
cmake -DCMAKE_PREFIX_PATH=/libtorch/share/cmake/Torch \
    -DWITH_CUDA=ON  \
    ..  
make  
make install 
```

## 📦 Installation

Download the repository (skip if using docker):
```bash
git clone https://github.com/leaner-forever/SEGS-SLAM
cd SEGS-SLAM
cd ORB-SLAM3/Vocabulary/
tar -xf ORBvoc.txt.tar.gz
cd ../..
chmod +x ./build.sh
./build.sh
```

**Note**: Ensure `set(Torch_DIR /home/lzy/dependency/libtorch/share/cmake/Torch)` is correctly set at line 23 of `CMakeLists.txt`.

## 🎮 Usage Examples

The benchmark datasets mentioned in our paper: [Replica (NICE-SLAM Version)](https://github.com/cvg/nice-slam), [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download), [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets).

### Download Datasets
```bash
cd scripts
chmod +x ./*.sh
./download_replica.sh
./download_tum.sh
./download_euroc.sh
```

**Note**: For ScanNet dataset, follow the [ScanNet](http://www.scan-net.org/) website procedure and extract color/depth frames from `.sens` files using this [code](https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/reader.py).

We use these sequences:
```
scene0000_00, scene0059_00, scene0106_00, scene0169_00
scene0181_00, scene0207_00, scene0472_00
```

### Running Examples

#### 1. Basic Testing
For testing, use these commands after specifying `PATH_TO_tum` and `PATH_TO_SAVE_RESULTS`:
```bash
cd ..
chmod +x ./bin/tum_rgbd ./bin/tum_mono ./bin/replica_rgbd ./bin/replica_mono ./bin/euroc_stereo

./bin/tum_rgbd \
    ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ./cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_desk.yaml \
    ./cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml \
    PATH_TO_tum/rgbd_dataset_freiburg1_desk \
    ./cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg1_desk.txt \
    PATH_TO_SAVE_RESULTS \
    no_viewer \
    undistorted_image
```

**Note**: Currently, the viewer is under development, so we disable it by adding `no_viewer` during evaluation.

#### 2. Batch Experiments
We provide scripts to run experiments on all benchmark datasets. Each sequence runs five times to reduce nondeterministic effects:
```bash
cd scripts
chmod +x ./*.sh
./replica_mono.sh
./replica_rgbd.sh
./tum_mono.sh
./tum_rgbd.sh
./euroc_stereo.sh
./scannet_rgbd.sh
# etc.
```

## 🔬 Evaluation

### Prerequisites

Ensure your results are stored in the correct format:
```
results
├── replica_mono_0
│   ├── office0
│   ├── ....
│   └── room2
├── replica_rgbd_0
│   ├── office0
│   ├── ....
│   └── room2
└── [replica/tum/euroc]_[mono/stereo/rgbd]_num  ....
    ├── scene_1
    ├── ....
    └── scene_n
```

#### Environment Setup
Our default install method uses [Anaconda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install):
```bash
cd /SEGS-SLAM/eval
conda env create --file environment.yml
conda activate segs-slam
```

#### Data Preparation

**Convert Replica GT camera pose files**:
```bash
python shapeReplicaGT.py --replica_dataset_path PATH_TO_REPLICA_DATASET
```

**Copy TUM camera.yaml** to dataset paths:
```bash
cp TUM/fr1/camera.yaml PATH_TO_TUM_DATASET/rgbd_dataset_freiburg1_desk
cp TUM/fr2/camera.yaml PATH_TO_TUM_DATASET/rgbd_dataset_freiburg2_xyz
```

**Note**: Some TUM sequences contain distorted images requiring undistortion before evaluation. The `camera.yaml` file serves as an indicator in `run.py`.

### Running Evaluation

**Get all metrics**:
```bash
python onekey.py --dataset_center_path PATH_TO_ALL_DATASET --result_main_folder RESULTS_PATH
```

This generates `RESULTS_PATH/log.txt` and `RESULTS_PATH/log.csv`.

**Evaluate all sequences** (if environments are set up):
```bash
cd SEGS-SLAM/scripts
chmod +x ./*.sh
./all.sh
```

## 📚 Citation

If you use SEGS-SLAM in your research, please cite our paper:

```bibtex
@inproceedings{tianci2025segsslam,
    title = {SEGS-SLAM: Structure-enhanced 3D Gaussian Splatting SLAM with Appearance Embedding},
    author = {Tianci Wen, Zhiang Liu, Yongchun Fang},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision},
    year = {2025}
}
```

---

<div align="center">

**Made with ❤️ by the SEGS-SLAM Team**

*If you find this project helpful, please give us a ⭐ star!*

</div>



