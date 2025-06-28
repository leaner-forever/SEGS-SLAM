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


# for i in 0 1 2 3 4
# do
# ../bin/tum_rgbd \
#     ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
#     ../cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_desk.yaml \
#     ../cfg/gaussian_mapper/RGB-D/ScanNet/scannet_rgbd.yaml \
#     /home/lzy/workingspace/MonoGS/datasets/tum/rgbd_dataset_freiburg1_desk \
#     ../cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg1_desk.txt \
#     ../results/tum_rgbd_$i/rgbd_dataset_freiburg1_desk \
#     no_viewer

# ../bin/tum_rgbd \
#     ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
#     ../cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg2_xyz.yaml \
#     ../cfg/gaussian_mapper/RGB-D/ScanNet/scannet_rgbd.yaml \
#     /home/lzy/workingspace/MonoGS/datasets/tum/rgbd_dataset_freiburg2_xyz \
#     ../cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg2_xyz.txt \
#     ../results/tum_rgbd_$i/rgbd_dataset_freiburg2_xyz \
#     no_viewer

# ../bin/tum_rgbd \
#     ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
#     ../cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
#     ../cfg/gaussian_mapper/RGB-D/ScanNet/scannet_rgbd.yaml \
#     /home/lzy/workingspace/MonoGS/datasets/tum//rgbd_dataset_freiburg3_long_office_household \
#     ../cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt \
#     ../results/tum_rgbd_$i/rgbd_dataset_freiburg3_long_office_household \
#     no_viewer

# # ../bin/tum_mono \
# #     ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
# #     ../cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg2_xyz.yaml \
# #     ../cfg/gaussian_mapper/RGB-D/ScanNet/scannet_rgbd.yaml \
# #     /home/lzy/datasets/TUM/rgbd_dataset_freiburg2_large_no_loop \
# #     ../cfg/ORB_SLAM3/RGB-D/TUM/associations/fr2_large_no_loop.txt \
# #     ../results/tum_mono_$i/rgbd_dataset_freiburg2_large_no_loop

# # ../bin/tum_mono \
# #     ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
# #     ../cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg2_xyz.yaml \
# #     ../cfg/gaussian_mapper/RGB-D/ScanNet/scannet_rgbd.yaml \
# #     /home/lzy/datasets/TUM/rgbd_dataset_freiburg2_large_with_loop \
# #     ../cfg/ORB_SLAM3/RGB-D/TUM/associations/fr2_large_with_loop.txt \
# #     ../results/tum_mono_$i/rgbd_dataset_freiburg2_large_with_loop
# #     no_viewer

    
# done

# ./bin/tum_rgbd     ./ORB-SLAM3/Vocabulary/ORBvoc.txt     ./cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml     ./cfg/gaussian_mapper/RGB-D/ScanNet/scannet_rgbd.yaml     /home/lzy/workingspace/MonoGS/datasets/tum//rgbd_dataset_freiburg3_long_office_household     ./cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt     ./results/tum_rgbd_$i/rgbd_dataset_freiburg3_long_office_household
