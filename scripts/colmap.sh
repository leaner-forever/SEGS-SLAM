#!/bin/bash

# ../bin/train_colmap \
#     ../cfg/colmap/gaussian_splatting.yaml \
#     /home/rapidlab/programs/NeuralSLAM_ws/gaussian-splatting-materials/tandt_db/db/drjohnson \
#     ../results/colmap/drjohnson \
#     no_viewer

# ../bin/train_colmap \
#     ../cfg/colmap/gaussian_splatting.yaml \
#     /home/lzy/datasets/TUM/rgbd_dataset_freiburg3_long_office_household/3DGS/ \
#     ../results/colmap/TUM_fr3 \
#     no_viewer

# mlp_opacity_lr_init: 0.01
../bin/train_colmap \
    ../cfg/colmap/sf_gs_01.yaml \
    /home/lzy/datasets/TUM/rgbd_dataset_freiburg3_long_office_household/3DGS/ \
    ../results/colmap/TUM_fr3 \
    no_viewer

../bin/train_colmap \
    ../cfg/colmap/sf_gs_01.yaml \
    /home/lzy/workingspace/Scaffold-GS/data/tum_colmap/fr1_desk/ \
    ../results/colmap/TUM_fr1_desk \
    no_viewer

../bin/train_colmap \
    ../cfg/colmap/sf_gs_01.yaml \
    /home/lzy/workingspace/Scaffold-GS/data/tum_colmap/fr1_desk/ \
    ../results/colmap/TUM_fr1_desk \
    no_viewer


./bin/train_colmap \
    ./cfg/colmap/sf_gs_01.yaml \
    /home/lzy/datasets/TUM/rgbd_dataset_freiburg3_long_office_household/3DGS/ \
    ./results/colmap/TUM_fr3 \
    no_viewer

./bin/train_colmap \
    ./cfg/colmap/sf_gs_01.yaml \
    /home/lzy/workingspace/Scaffold-GS/data/tum_colmap/fr1_desk/ \
    ./results/colmap/TUM_fr1_desk \
    no_viewer

./bin/train_colmap \
    ./cfg/colmap/sf_gs_01.yaml \
    /home/lzy/workingspace/Scaffold-GS/data/tum_colmap/fr2_xyz/ \
    ./results/colmap/TUM_fr2_xyz \
    no_viewer

# sleep 20s

# ../bin/train_colmap \
#     ../cfg/colmap/sf_gs_01.yaml \
#     /home/lzy/datasets/TUM/rgbd_dataset_freiburg3_long_office_household/3DGS/ \
#     ../results/colmap/TUM_fr3 \
#     no_viewer

# sleep 20s

# ../bin/train_colmap \
#     ../cfg/colmap/sf_gs_01.yaml \
#     /home/lzy/datasets/TUM/rgbd_dataset_freiburg3_long_office_household/3DGS/ \
#     ../results/colmap/TUM_fr3 \
#     no_viewer

# sleep 20s

# ../bin/train_colmap \
#     ../cfg/colmap/sf_gs_01.yaml \
#     /home/lzy/datasets/TUM/rgbd_dataset_freiburg3_long_office_household/3DGS/ \
#     ../results/colmap/TUM_fr3 \
#     no_viewer

# sleep 20s

# ../bin/train_colmap \
#     ../cfg/colmap/sf_gs_01.yaml \
#     /home/lzy/datasets/TUM/rgbd_dataset_freiburg3_long_office_household/3DGS/ \
#     ../results/colmap/TUM_fr3 \
#     no_viewer

# ./bin/train_colmap     ./cfg/colmap/gaussian_splatting.yaml     /home/lzy/datasets/TUM/rgbd_dataset_freiburg3_long_office_household/3DGS/     ./results/colmap/TUM_fr3     no_viewer