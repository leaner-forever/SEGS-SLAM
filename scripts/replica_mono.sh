#!/bin/bash
for i in 0 1 2 3 4
# for i in 0 1 2 3 4
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
done


cd .. 

cd eval

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gaussian_splatting

python onekey.py --dataset_center_path "/home/lzy/workingspace/MonoGS/datasets/" --result_main_folder "/home/lzy/workingspace/Scaffold-GS-cpp/results/"


./bin/replica_mono \
    ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ./cfg/ORB_SLAM3/Monocular/Replica/office0.yaml \
    ./cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml \
    /home/lzy/workingspace/MonoGS/datasets/replica/office0 \
    ./results/replica_mono_$i/office0 \
    no_viewer

./bin/replica_mono \
    ./ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ./cfg/ORB_SLAM3/Monocular/Replica/office1.yaml \
    ./cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml \
    /home/lzy/workingspace/MonoGS/datasets/replica/office1 \
    ./results/replica_mono_$i/office1 \
    no_viewer