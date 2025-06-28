#!/bin/bash

# for i in 0 1 2 3 4
for i in 0
do

../bin/tum_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_desk.yaml \
    ../cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml \
    /home/lzy/workingspace/MonoGS/datasets/tum/rgbd_dataset_freiburg1_desk \
    ../cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg1_desk.txt \
    ../results/tum_rgbd_$i/rgbd_dataset_freiburg1_desk \
    no_viewer \
    undistorted_image

../bin/tum_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg2_xyz.yaml \
    ../cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml \
    /home/lzy/workingspace/MonoGS/datasets/tum/rgbd_dataset_freiburg2_xyz \
    ../cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg2_xyz.txt \
    ../results/tum_rgbd_$i/rgbd_dataset_freiburg2_xyz \
    no_viewer \
    undistorted_image

../bin/tum_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
    ../cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml \
    /home/lzy/workingspace/MonoGS/datasets/tum/rgbd_dataset_freiburg3_long_office_household  \
    ../cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt \
    ../results/tum_rgbd_$i/rgbd_dataset_freiburg3_long_office_household \
    no_viewer

# ../bin/tum_rgbd \
#     ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
#     ../cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_desk.yaml \
#     ../cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg1_desk.yaml \
#     /home/lzy/workingspace/MonoGS/datasets/tum/rgbd_dataset_freiburg1_desk \
#     ../cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg1_desk.txt \
#     ../results/tum_rgbd_$i/rgbd_dataset_freiburg1_desk \
#     no_viewer \
#     undistorted_image

# ../bin/tum_rgbd \
#     ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
#     ../cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg2_xyz.yaml \
#     ../cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg2_xyz.yaml \
#     /home/lzy/workingspace/MonoGS/datasets/tum/rgbd_dataset_freiburg2_xyz \
#     ../cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg2_xyz.txt \
#     ../results/tum_rgbd_$i/rgbd_dataset_freiburg2_xyz \
#     no_viewer \
#     undistorted_image

# ../bin/tum_rgbd \
#     ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
#     ../cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
#     ../cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
#     /home/lzy/workingspace/MonoGS/datasets/tum/rgbd_dataset_freiburg3_long_office_household  \
#     ../cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt \
#     ../results/tum_rgbd_$i/rgbd_dataset_freiburg3_long_office_household \
#     no_viewer

# ../bin/tum_mono \
#     ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
#     ../cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg2_xyz.yaml \
#     ../cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml \
#     /home/lzy/datasets/TUM/rgbd_dataset_freiburg2_large_no_loop \
#     ../cfg/ORB_SLAM3/RGB-D/TUM/associations/fr2_large_no_loop.txt \
#     ../results/tum_mono_$i/rgbd_dataset_freiburg2_large_no_loop

# ../bin/tum_mono \
#     ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
#     ../cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg2_xyz.yaml \
#     ../cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml \
#     /home/lzy/datasets/TUM/rgbd_dataset_freiburg2_large_with_loop \
#     ../cfg/ORB_SLAM3/RGB-D/TUM/associations/fr2_large_with_loop.txt \
#     ../results/tum_mono_$i/rgbd_dataset_freiburg2_large_with_loop
#     no_viewer

    
done

./bin/tum_rgbd \
    ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ./cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
    ./cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml \
    /home/lzy/workingspace/MonoGS/datasets/tum/rgbd_dataset_freiburg3_long_office_household  \
    ./cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt \
    ./results/tum_rgbd_/rgbd_dataset_freiburg3_long_office_household \
    no_viewer

./bin/tum_rgbd \
    ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ./cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg2_xyz.yaml \
    ./cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg2_xyz.yaml \
    /home/lzy/workingspace/MonoGS/datasets/tum/rgbd_dataset_freiburg2_xyz \
    ./cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg2_xyz.txt \
    ./results/tum_rgbd/rgbd_dataset_freiburg2_xyz \
    no_viewer \
    undistorted_image

./bin/tum_rgbd \
    ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ./cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_desk.yaml \
    ./cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg1_desk.yaml \
    /home/lzy/workingspace/MonoGS/datasets/tum/rgbd_dataset_freiburg1_desk \
    ./cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg1_desk.txt \
    ./results/tum_rgbd_$i/rgbd_dataset_freiburg1_desk \
    no_viewer \
    undistorted_image
# ./bin/tum_rgbd     ./ORB-SLAM3/Vocabulary/ORBvoc.txt     ./cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml     ./cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml     /home/lzy/workingspace/MonoGS/datasets/tum/rgbd_dataset_freiburg3_long_office_household     ./cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt     ./results/tum_rgbd_/rgbd_dataset_freiburg3_long_office_household no_viewer

# ./bin/tum_rgbd \
#     ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
#     ./cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg2_xyz.yaml \
#     ./cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg2_xyz.yaml \
#     /home/lzy/workingspace/MonoGS/datasets/tum/rgbd_dataset_freiburg2_xyz \
#     ./cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg2_xyz.txt \
#     ./results/tum_rgbd/rgbd_dataset_freiburg2_xyz \
#     no_viewer

# ./bin/tum_rgbd \
#     ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
#     ./cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_desk.yaml \
#     ./cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg1_desk.yaml \
#     /home/lzy/workingspace/MonoGS/datasets/tum/rgbd_dataset_freiburg1_room \
#     ./cfg/ORB_SLAM3/RGB-D/TUM/associations/fr1_room.txt \
#     ./results/tum_rgbd_$i/rgbd_dataset_freiburg1_room \
#     no_viewer

# ./bin/tum_rgbd \
#     ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
#     ./cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg2_xyz.yaml \
#     ./cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg2_xyz.yaml \
#     /home/lzy/workingspace/MonoGS/datasets/tum/rgbd_dataset_freiburg2_desk \
#     ./cfg/ORB_SLAM3/RGB-D/TUM/associations/fr2_desk.txt \
#     ./results/tum_rgbd/rgbd_dataset_freiburg2_desk \
#     no_viewer

# ./bin/tum_rgbd \
#     ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
#     ./cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
#     ./cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml \
#     /home/lzy/workingspace/MonoGS/datasets/tum/rgbd_dataset_freiburg3_large_cabinet  \
#     ./cfg/ORB_SLAM3/RGB-D/TUM/associations/fr3_cabinet.txt \
#     ./results/tum_rgbd_/rgbd_dataset_freiburg3_large_cabinet \
#     no_viewer

# ./bin/tum_rgbd \
#     ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
#     ./cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
#     ./cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml \
#     /home/lzy/workingspace/MonoGS/datasets/tum/rgbd_dataset_freiburg3_structure_texture_far  \
#     ./cfg/ORB_SLAM3/RGB-D/TUM/associations/fr3_str_tex_far.txt \
#     ./results/tum_rgbd_/rgbd_dataset_freiburg3_structure_texture_far \
#     no_viewer

# ./bin/tum_rgbd \
#     ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
#     ./cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
#     ./cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml \
#     /home/lzy/workingspace/MonoGS/datasets/tum/rgbd_dataset_freiburg3_structure_texture_near  \
#     ./cfg/ORB_SLAM3/RGB-D/TUM/associations/fr3_str_tex_near.txt \
#     ./results/tum_rgbd_/rgbd_dataset_freiburg3_structure_texture_near \
#     no_viewer