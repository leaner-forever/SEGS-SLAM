#!/bin/bash

# for i in 0 1 2 3 4
for i in 0
do
../bin/tum_mono \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg1_desk.yaml \
    ../cfg/gaussian_mapper/Monocular/TUM/tum_mono.yaml \
    /home/lzy/workingspace/MonoGS/datasets/tum/rgbd_dataset_freiburg1_desk \
    ../results/tum_mono_$i/rgbd_dataset_freiburg1_desk \
    no_viewer\
    undistorted_image

../bin/tum_mono \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg2_xyz.yaml \
    ../cfg/gaussian_mapper/Monocular/TUM/tum_mono.yaml \
    /home/lzy/workingspace/MonoGS/datasets/tum/rgbd_dataset_freiburg2_xyz \
    ../results/tum_mono_$i/rgbd_dataset_freiburg2_xyz \
    no_viewer\
    undistorted_image

../bin/tum_mono \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg3_long_office_household.yaml \
    ../cfg/gaussian_mapper/Monocular/TUM/tum_mono.yaml \
    /home/lzy/workingspace/MonoGS/datasets/tum/rgbd_dataset_freiburg3_long_office_household \
    ../results/tum_mono_$i/rgbd_dataset_freiburg3_long_office_household \
    no_viewer
done

./bin/tum_mono \
    ./ORB-SLAM3/Vocabulary/ORBvoc.txt     \
    ./cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg3_long_office_household.yaml     \
    ./cfg/gaussian_mapper/Monocular/TUM/tum_mono.yaml     \
    /home/lzy/workingspace/MonoGS/datasets/tum/rgbd_dataset_freiburg3_long_office_household     \
    ./results/tum_mono_/rgbd_dataset_freiburg3_long_office_household     \
    no_viewer

./bin/tum_mono \
    ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ./cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg1_desk.yaml \
    ./cfg/gaussian_mapper/Monocular/TUM/tum_mono.yaml \
    /home/lzy/workingspace/MonoGS/datasets/tum/rgbd_dataset_freiburg1_desk \
    ./results/tum_mono_$i/rgbd_dataset_freiburg1_desk \
    no_viewer\
    undistorted_image

./bin/tum_mono \
    ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ./cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg2_xyz.yaml \
    ./cfg/gaussian_mapper/Monocular/TUM/tum_mono.yaml \
    /home/lzy/workingspace/MonoGS/datasets/tum/rgbd_dataset_freiburg2_xyz \
    ./results/tum_mono_$i/rgbd_dataset_freiburg2_xyz \
    no_viewer\
    undistorted_image

# for i in 0 1 2 3 4
# do
# ../bin/tum_mono \
#     ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
#     ../cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg2_xyz.yaml \
#     ../cfg/gaussian_mapper/Monocular/TUM/tum_mono.yaml \
#     /home/lzy/datasets/TUM/rgbd_dataset_freiburg2_large_no_loop \
#     ../results/tum_mono_$i/rgbd_dataset_freiburg2_large_no_loop

# ../bin/tum_mono \
#     ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
#     ../cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg2_xyz.yaml \
#     ../cfg/gaussian_mapper/Monocular/TUM/tum_mono.yaml \
#     /home/lzy/datasets/TUM/rgbd_dataset_freiburg2_large_with_loop \
#     ../results/tum_mono_$i/rgbd_dataset_freiburg2_large_with_loop \
#     no_viewer

# done


cd .. 

cd eval

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gaussian_splatting

python onekey.py --dataset_center_path "/home/lzy/workingspace/MonoGS/datasets/" --result_main_folder "/home/lzy/workingspace/Scaffold-GS-cpp/results/"