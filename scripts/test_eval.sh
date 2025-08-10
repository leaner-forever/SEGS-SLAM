#!/bin/bash

#5 is for test
for i in 0 1 2 3 4
do

../bin/replica_mono \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Monocular/Replica/office0.yaml \
    ../cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml \
    /home/lzy/workingspace/MonoGS/datasets/replica/office0 \
    ../results/replica_mono_$i/office0 \
    no_viewer

../bin/replica_mono \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Monocular/Replica/office1.yaml \
    ../cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml \
    /home/lzy/workingspace/MonoGS/datasets/replica/office1 \
    ../results/replica_mono_$i/office1 \
    no_viewer

../bin/replica_mono \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Monocular/Replica/office2.yaml \
    ../cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml \
    /home/lzy/workingspace/MonoGS/datasets/replica/office2 \
    ../results/replica_mono_$i/office2 \
    no_viewer

../bin/replica_mono \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Monocular/Replica/office3.yaml \
    ../cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml \
    /home/lzy/workingspace/MonoGS/datasets/replica/office3 \
    ../results/replica_mono_$i/office3 \
    no_viewer

../bin/replica_mono \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Monocular/Replica/office4.yaml \
    ../cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml \
    /home/lzy/workingspace/MonoGS/datasets/replica/office4 \
    ../results/replica_mono_$i/office4 \
    no_viewer

../bin/replica_mono \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Monocular/Replica/room0.yaml \
    ../cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml \
    /home/lzy/workingspace/MonoGS/datasets/replica/room0 \
    ../results/replica_mono_$i/room0 \
    no_viewer

../bin/replica_mono \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Monocular/Replica/room1.yaml \
    ../cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml \
    /home/lzy/workingspace/MonoGS/datasets/replica/room1 \
    ../results/replica_mono_$i/room1 \
    no_viewer

../bin/replica_mono \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Monocular/Replica/room2.yaml \
    ../cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml \
    /home/lzy/workingspace/MonoGS/datasets/replica/room2 \
    ../results/replica_mono_$i/room2 \
    no_viewer

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

../bin/tum_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_desk.yaml \
    ../cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg1_desk.yaml \
    /home/lzy/workingspace/MonoGS/datasets/tum/rgbd_dataset_freiburg1_desk \
    ../cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg1_desk.txt \
    ../results/tum_rgbd_$i/rgbd_dataset_freiburg1_desk \
    no_viewer \
    undistorted_image

../bin/tum_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg2_xyz.yaml \
    ../cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg2_xyz.yaml \
    /home/lzy/workingspace/MonoGS/datasets/tum/rgbd_dataset_freiburg2_xyz \
    ../cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg2_xyz.txt \
    ../results/tum_rgbd_$i/rgbd_dataset_freiburg2_xyz \
    no_viewer \
    undistorted_image

../bin/tum_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
    ../cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
    /home/lzy/workingspace/MonoGS/datasets/tum/rgbd_dataset_freiburg3_long_office_household  \
    ../cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt \
    ../results/tum_rgbd_$i/rgbd_dataset_freiburg3_long_office_household \
    no_viewer

../bin/euroc_stereo \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Stereo/EuRoC/EuRoC.yaml \
    ../cfg/gaussian_mapper/Stereo/EuRoC/EuRoC.yaml \
    /home/lzy/workingspace/MonoGS/datasets/EuRoC/MH_01_easy \
    ../cfg/ORB_SLAM3/Stereo/EuRoC/EuRoC_TimeStamps/MH01.txt \
    ../results/euroc_stereo_$i/MH_01_easy \
    no_viewer

../bin/euroc_stereo \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Stereo/EuRoC/EuRoC.yaml \
    ../cfg/gaussian_mapper/Stereo/EuRoC/EuRoC.yaml \
    /home/lzy/workingspace/MonoGS/datasets/EuRoC/MH_02_easy \
    ../cfg/ORB_SLAM3/Stereo/EuRoC/EuRoC_TimeStamps/MH02.txt \
    ../results/euroc_stereo_$i/MH_02_easy \
    no_viewer

../bin/euroc_stereo \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Stereo/EuRoC/EuRoC.yaml \
    ../cfg/gaussian_mapper/Stereo/EuRoC/EuRoC.yaml \
    /home/lzy/workingspace/MonoGS/datasets/EuRoC/V1_01_easy \
    ../cfg/ORB_SLAM3/Stereo/EuRoC/EuRoC_TimeStamps/V101.txt \
    ../results/euroc_stereo_$i/V1_01_easy \
    no_viewer

../bin/euroc_stereo \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/Stereo/EuRoC/EuRoC.yaml \
    ../cfg/gaussian_mapper/Stereo/EuRoC/EuRoC.yaml \
    /home/lzy/workingspace/MonoGS/datasets/EuRoC/V2_01_easy \
    ../cfg/ORB_SLAM3/Stereo/EuRoC/EuRoC_TimeStamps/V201.txt \
    ../results/euroc_stereo_$i/V2_01_easy \
    no_viewer

../bin/replica_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/Replica/office0.yaml \
    ../cfg/gaussian_mapper/RGB-D/Replica/office0.yaml \
    /home/lzy/workingspace/MonoGS/datasets/replica/office0 \
    ../results/replica_rgbd_$i/office0 \
    no_viewer

../bin/replica_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/Replica/office1.yaml \
    ../cfg/gaussian_mapper/RGB-D/Replica/office1.yaml \
    /home/lzy/workingspace/MonoGS/datasets/replica/office1 \
    ../results/replica_rgbd_$i/office1 \
    no_viewer

../bin/replica_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/Replica/office2.yaml \
    ../cfg/gaussian_mapper/RGB-D/Replica/office2.yaml \
     /home/lzy/workingspace/MonoGS/datasets/replica/office2 \
    ../results/replica_rgbd_$i/office2 \
    no_viewer

../bin/replica_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/Replica/office3.yaml \
    ../cfg/gaussian_mapper/RGB-D/Replica/office3.yaml \
     /home/lzy/workingspace/MonoGS/datasets/replica/office3 \
    ../results/replica_rgbd_$i/office3 \
    no_viewer

../bin/replica_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/Replica/office4.yaml \
    ../cfg/gaussian_mapper/RGB-D/Replica/office4.yaml \
     /home/lzy/workingspace/MonoGS/datasets/replica/office4 \
    ../results/replica_rgbd_$i/office4 \
    no_viewer

../bin/replica_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/Replica/room0.yaml \
    ../cfg/gaussian_mapper/RGB-D/Replica/room0.yaml \
     /home/lzy/workingspace/MonoGS/datasets/replica/room0 \
    ../results/replica_rgbd_$i/room0 \
    no_viewer

../bin/replica_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/Replica/room1.yaml \
    ../cfg/gaussian_mapper/RGB-D/Replica/room1.yaml \
     /home/lzy/workingspace/MonoGS/datasets/replica/room1 \
    ../results/replica_rgbd_$i/room1 \
    no_viewer

../bin/replica_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/Replica/room2.yaml \
    ../cfg/gaussian_mapper/RGB-D/Replica/room2.yaml \
     /home/lzy/workingspace/MonoGS/datasets/replica/room2 \
    ../results/replica_rgbd_$i/room2 \
    no_viewer
done

cd .. 

cd eval

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gaussian_splatting

python onekey.py --dataset_center_path "/home/lzy/workingspace/MonoGS/datasets/" --result_main_folder "/home/lzy/workingspace/SEGS-SLAM/results/"