/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * 
 * This file is Derivative Works of Gaussian Splatting,
 * created by Longwei Li, Huajian Huang, Hui Cheng and Sai-Kit Yeung in 2023,
 * as part of Photo-SLAM.
 */

#pragma once

#include <string>
#include <filesystem>
#include <iostream>

class GaussianModelParams
{
public:
    GaussianModelParams(
        std::filesystem::path source_path = "",
        std::filesystem::path model_path = "",
        std::filesystem::path exec_path = "",
        int sh_degree = 3,
        std::string images = "images",
        float resolution = -1.0f,
        bool white_background = false,
        std::string data_device = "cuda",
        bool eval = false,

        int feat_dim = 32,
        int n_offsets = 10,
        float voxel_size =  0.001,
        int update_depth = 3,
        int update_init_factor = 16,
        int update_hierachy_factor = 4,

        bool use_feat_bank = true,
        int appearance_dim = 32,
        bool lowpoly = false,
        int ds = 1,
        float ratio = 1,
        bool undistorted = false ,
        
        bool add_opacity_dist = false,
        bool add_cov_dist = false,
        bool add_color_dist = false,
        int embedding_dim = 200,

        
        bool use_coarse_anchor = false,
        float coarse_voxel_size = 0.2,
        int feat_dim_coarse = 32,
        int n_offsets_coarse = 10,
        int appearance_dim_coarse = 32
        );
    void logger();

public:
    int sh_degree_;
    std::filesystem::path source_path_;
    std::filesystem::path model_path_;
    std::string images_;
    float resolution_;
    bool white_background_;
    std::string data_device_;
    bool eval_;

    int feat_dim;
    int n_offsets;
    float voxel_size; 
    int update_depth;
    int update_init_factor;
    int update_hierachy_factor;

    bool use_feat_bank;
    int appearance_dim;
    bool lowpoly;
    int ds;
    float ratio; 
    bool undistorted;

    bool add_opacity_dist;
    bool add_cov_dist;
    bool add_color_dist = false;

    int embedding_dim;

    bool use_coarse_anchor;
    float coarse_voxel_size;
    int feat_dim_coarse;
    int n_offsets_coarse;
    int appearance_dim_coarse;
};

class GaussianPipelineParams
{
public:
    GaussianPipelineParams(
        bool convert_SHs = false,
        bool compute_cov3D = false);

public:
    bool convert_SHs_;
    bool compute_cov3D_;
};

class GaussianOptimizationParams
{
public:
    GaussianOptimizationParams(
        int iterations = 30'000,
        float position_lr_init = 0.00016f,
        float position_lr_final = 0.0000016f,
        float position_lr_delay_mult = 0.01f,
        int position_lr_max_steps = 30'000,

        float offset_lr_init = 0.01,
        float offset_lr_final = 0.0001,
        float offset_lr_delay_mult = 0.01,
        int offset_lr_max_steps = 30'000,

        float feature_lr = 0.0025f,
        float opacity_lr = 0.05f,
        float scaling_lr = 0.005f,
        float rotation_lr = 0.001f,

        float mlp_opacity_lr_init = 0.002,
        float mlp_opacity_lr_final = 0.00002,
        float mlp_opacity_lr_delay_mult = 0.01,
        int mlp_opacity_lr_max_steps = 30'000,

        float mlp_cov_lr_init = 0.004, 
        float mlp_cov_lr_final = 0.004,
        float mlp_cov_lr_delay_mult = 0.01,
        int mlp_cov_lr_max_steps = 30'000,

        float mlp_color_lr_init = 0.008, 
        float mlp_color_lr_final = 0.00005,
        float mlp_color_lr_delay_mult = 0.01,
        int mlp_color_lr_max_steps = 30'000,

        float mlp_featurebank_lr_init = 0.01, 
        float mlp_featurebank_lr_final = 0.00001,
        float mlp_featurebank_lr_delay_mult = 0.01,
        int mlp_featurebank_lr_max_steps = 30'000,

        float appearance_lr_init = 0.05, 
        float appearance_lr_final = 0.0005,
        float appearance_lr_delay_mult = 0.01,
        int appearance_lr_max_steps = 30'000,

        float percent_dense = 0.01f,
        float lambda_dssim = 0.2f,

        int start_stat = 500,
        int update_from = 1500,
        int update_interval = 100,
        int update_until = 15000,

        float min_opacity = 0.005, 
        float success_threshold = 0.8,
        float densify_grad_threshold = 0.0002,

        float anchor_lr_init_coarse = 0.00016f,
        float anchor_lr_final_coarse = 0.0000016f,
        float anchor_lr_delay_mult_coarse = 0.01f,
        int anchor_lr_max_steps_coarse = 30'000,

        float offset_lr_init_coarse = 0.01,
        float offset_lr_final_coarse = 0.0001,
        float offset_lr_delay_mult_coarse = 0.01,
        int offset_lr_max_steps_coarse = 30'000,

        float feature_lr_coarse = 0.0025f,
        float opacity_lr_coarse = 0.05f,
        float scaling_lr_coarse = 0.005f,
        float rotation_lr_coarse = 0.001f,

        float mlp_opacity_lr_init_coarse = 0.002,
        float mlp_opacity_lr_final_coarse = 0.00002,
        float mlp_opacity_lr_delay_mult_coarse = 0.01,
        int mlp_opacity_lr_max_steps_coarse = 30'000,

        float mlp_cov_lr_init_coarse = 0.004, 
        float mlp_cov_lr_final_coarse = 0.004,
        float mlp_cov_lr_delay_mult_coarse = 0.01,
        int mlp_cov_lr_max_steps_coarse = 30'000,

        float mlp_color_lr_init_coarse = 0.008, 
        float mlp_color_lr_final_coarse = 0.00005,
        float mlp_color_lr_delay_mult_coarse = 0.01,
        int mlp_color_lr_max_steps_coarse = 30'000,

        float mlp_featurebank_lr_init_coarse = 0.01, 
        float mlp_featurebank_lr_final_coarse = 0.00001,
        float mlp_featurebank_lr_delay_mult_coarse = 0.01,
        int mlp_featurebank_lr_max_steps_coarse = 30'000,

        float appearance_lr_init_coarse = 0.05, 
        float appearance_lr_final_coarse = 0.0005,
        float appearance_lr_delay_mult_coarse = 0.01,
        int appearance_lr_max_steps_coarse = 30'000);

    void logger();

public:
    int iterations_;
    float position_lr_init_;
    float position_lr_final_;
    float position_lr_delay_mult_;
    int position_lr_max_steps_;

    float offset_lr_init;
    float offset_lr_final;
    float offset_lr_delay_mult;
    int offset_lr_max_steps;

    float feature_lr_;
    float opacity_lr_;
    float scaling_lr_;
    float rotation_lr_;

    float mlp_opacity_lr_init;
    float mlp_opacity_lr_final;
    float mlp_opacity_lr_delay_mult;
    int mlp_opacity_lr_max_steps;

    float mlp_cov_lr_init;
    float mlp_cov_lr_final;
    float mlp_cov_lr_delay_mult;
    int mlp_cov_lr_max_steps;

    float mlp_color_lr_init;
    float mlp_color_lr_final;
    float mlp_color_lr_delay_mult;
    int mlp_color_lr_max_steps;

    float mlp_featurebank_lr_init;
    float mlp_featurebank_lr_final;
    float mlp_featurebank_lr_delay_mult;
    int mlp_featurebank_lr_max_steps;

    float appearance_lr_init;
    float appearance_lr_final;
    float appearance_lr_delay_mult;
    int appearance_lr_max_steps;

    float percent_dense_;
    float lambda_dssim_;

    int start_stat = 500;
    int update_from = 1500;
    int update_interval = 100;
    int update_until = 15000;
    
    float min_opacity = 0.005;
    float success_threshold = 0.8;
    float densify_grad_threshold = 0.0002;

    float anchor_lr_init_coarse;
    float anchor_lr_final_coarse;
    float anchor_lr_delay_mult_coarse;
    int anchor_lr_max_steps_coarse;

    float offset_lr_init_coarse;
    float offset_lr_final_coarse;
    float offset_lr_delay_mult_coarse;
    int offset_lr_max_steps_coarse;

    float feature_lr_coarse;
    float opacity_lr_coarse;
    float scaling_lr_coarse;
    float rotation_lr_coarse;

    float mlp_opacity_lr_init_coarse;
    float mlp_opacity_lr_final_coarse;
    float mlp_opacity_lr_delay_mult_coarse;
    int mlp_opacity_lr_max_steps_coarse;

    float mlp_cov_lr_init_coarse;
    float mlp_cov_lr_final_coarse;
    float mlp_cov_lr_delay_mult_coarse;
    int mlp_cov_lr_max_steps_coarse;

    float mlp_color_lr_init_coarse;
    float mlp_color_lr_final_coarse;
    float mlp_color_lr_delay_mult_coarse;
    int mlp_color_lr_max_steps_coarse;

    float mlp_featurebank_lr_init_coarse;
    float mlp_featurebank_lr_final_coarse;
    float mlp_featurebank_lr_delay_mult_coarse;
    int mlp_featurebank_lr_max_steps_coarse;

    float appearance_lr_init_coarse;
    float appearance_lr_final_coarse;
    float appearance_lr_delay_mult_coarse;
    int appearance_lr_max_steps_coarse;
};
