#!/bin/bash

for i in 0 1 2 3 4
do

./bin/scannet_rgbd \
    ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ./cfg/ORB_SLAM3/RGB-D/Scannet/0000.yaml\
    ./cfg/gaussian_mapper/RGB-D/ScanNet/scannet_rgbd.yaml \
    /wtc/ssd/datasets/scannet/scene0000_00 \
    ./results/scannet_rgbd_$i/0000  \
    no_viewer

./bin/scannet_rgbd \
    ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ./cfg/ORB_SLAM3/RGB-D/Scannet/0059.yaml\
    ./cfg/gaussian_mapper/RGB-D/ScanNet/scannet_rgbd.yaml \
    /wtc/ssd/datasets/scannet/scene0059_00 \
    ./results/scannet_rgbd_$i/0059  \
    no_viewer

./bin/scannet_rgbd \
    ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ./cfg/ORB_SLAM3/RGB-D/Scannet/0106.yaml\
    ./cfg/gaussian_mapper/RGB-D/ScanNet/scannet_rgbd.yaml \
    /wtc/ssd/datasets/scannet/scene0106_00 \
    ./results/scannet_rgbd_$i/0106  \
    no_viewer

./bin/scannet_rgbd \
    ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ./cfg/ORB_SLAM3/RGB-D/Scannet/0169.yaml\
    ./cfg/gaussian_mapper/RGB-D/ScanNet/scannet_rgbd.yaml \
    /wtc/ssd/datasets/scannet/scene0169_00 \
    ./results/scannet_rgbd_$i/0169  \
    no_viewer

./bin/scannet_rgbd \
    ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ./cfg/ORB_SLAM3/RGB-D/Scannet/0181.yaml\
    ./cfg/gaussian_mapper/RGB-D/ScanNet/scannet_rgbd.yaml \
    /wtc/ssd/datasets/scannet/scene0181_00 \
    ./results/scannet_rgbd_$i/0181  \
    no_viewer

./bin/scannet_rgbd \
    ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ./cfg/ORB_SLAM3/RGB-D/Scannet/0207.yaml\
    ./cfg/gaussian_mapper/RGB-D/ScanNet/scannet_rgbd.yaml \
    /wtc/ssd/datasets/scannet/scene0207_00 \
    ./results/scannet_rgbd_$i/0207  \
    no_viewer

./bin/scannet_rgbd \
    ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ./cfg/ORB_SLAM3/RGB-D/Scannet/0472.yaml\
    ./cfg/gaussian_mapper/RGB-D/ScanNet/scannet_rgbd.yaml \
    /wtc/ssd/datasets/scannet/scene0472_00 \
    ./results/scannet_rgbd_$i/0472  \
    no_viewer
done
