#!/bin/bash

for i in 0 1 2 3 4
do
../bin/tum_mono \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg2_xyz.yaml \
    ../cfg/gaussian_mapper/Monocular/TUM/tum_mono.yaml \
    /home/lzy/datasets/TUM/rgbd_dataset_freiburg2_large_no_loop \
    ../results/tum_mono_$i/rgbd_dataset_freiburg2_large_no_loop

../bin/kitti_stereo \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Monocular/KITTI/KITTI00-02.yaml  \
    ../cfg/gaussian_mapper/Stereo/KITTI/kitti_stereo.yaml \
    /home/lzy/datasets/kitti/data_odometry_color/dataset/sequences/02 \
    ../results/kitti_mono_$i/02 \
    no_viewer

# ./Examples/Stereo/stereo_kitti 
# ./Vocabulary/ORBvoc.txt 
# ./Examples/Stereo/KITTI04-12.yaml 
# ../Dataset/KITTI/kitti-odometry/data_odometry_gray/dataset/sequences/06

done