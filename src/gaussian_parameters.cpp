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

#include "include/gaussian_parameters.h"


GaussianModelParams::GaussianModelParams(
    std::filesystem::path source_path,
    std::filesystem::path model_path,
    std::filesystem::path exec_path,
    int sh_degree,
    std::string images,
    float resolution,
    bool white_background,
    std::string data_device,
    bool eval,
        
    int feat_dim,
    int n_offsets,
    float voxel_size, 
    int update_depth,
    int update_init_factor,
    int update_hierachy_factor,

    bool use_feat_bank,
    int appearance_dim,
    bool lowpoly,
    int ds,
    float ratio, 
    bool undistorted,

    bool add_opacity_dist,
    bool add_cov_dist,
    bool add_color_dist,
    int embedding_dim,

    bool use_coarse_anchor,
    float coarse_voxel_size,
    int feat_dim_coarse,
    int n_offsets_coarse,
    int appearance_dim_coarse
    )
    : sh_degree_(sh_degree),
      images_(images),
      resolution_(resolution),
      white_background_(white_background),
      data_device_(data_device),
      eval_(eval),

      feat_dim(feat_dim),
      n_offsets(n_offsets),
      voxel_size(voxel_size), 
      update_depth(update_depth),
      update_init_factor(update_init_factor),
      update_hierachy_factor(update_hierachy_factor),

      use_feat_bank(use_feat_bank), 
      appearance_dim(appearance_dim),
      lowpoly(lowpoly),
      ds(ds),
      ratio(ratio), 
      undistorted(undistorted),
      add_opacity_dist(add_opacity_dist),
      add_cov_dist(add_cov_dist),
      add_color_dist(add_color_dist),
      embedding_dim(embedding_dim),

      use_coarse_anchor(use_coarse_anchor),
      coarse_voxel_size(coarse_voxel_size),
      feat_dim_coarse(feat_dim_coarse),
      n_offsets_coarse(n_offsets_coarse),
      appearance_dim_coarse(appearance_dim_coarse)
{
    if (source_path.is_absolute())
        source_path_ = source_path;
    else
        source_path_ = exec_path / source_path;

    if (model_path.is_absolute())
        model_path_ = model_path;
    else
        model_path_ = exec_path / model_path;
}

void GaussianModelParams::logger()
{
    std::cout << "[Gaussian Param]args: "
              << "add_color_dist = " << add_color_dist
              << ", add_cov_dist = " << add_cov_dist
              << ", add_opacity_dist = " << add_opacity_dist
              << ", appearance_dim = " << appearance_dim
              << ", data_device_ = " << data_device_
              << ", ds = " << ds
              << ", feat_dim = " << feat_dim
              << ", lowpoly = " << lowpoly
              << ", n_offsets = " << n_offsets
              << ", ratio = " << ratio
              << ", resolution_ = " << resolution_
              << ", sh_degree_ = " << sh_degree_
              << ", undistorted = " << undistorted
              << ", update_depth = " << update_depth
              << ", update_hierachy_factor = " << update_hierachy_factor
              << ", update_init_factor = " << update_init_factor
              << ", use_feat_bank = " << use_feat_bank
              << ", voxel_size = " << voxel_size
              << ", white_background_ = " << white_background_
              << std::endl;
}

GaussianPipelineParams::GaussianPipelineParams(bool convert_SHs, bool compute_cov3D)
    : convert_SHs_(convert_SHs), compute_cov3D_(compute_cov3D)
{}

GaussianOptimizationParams::GaussianOptimizationParams(
    int iterations,
    float position_lr_init,
    float position_lr_final,
    float position_lr_delay_mult,
    int position_lr_max_steps,

    float offset_lr_init,
    float offset_lr_final,
    float offset_lr_delay_mult,
    int offset_lr_max_steps,

    float feature_lr,
    float opacity_lr,
    float scaling_lr,
    float rotation_lr,

    float mlp_opacity_lr_init, 
    float mlp_opacity_lr_final,
    float mlp_opacity_lr_delay_mult,
    int mlp_opacity_lr_max_steps,
    float mlp_cov_lr_init, 
    float mlp_cov_lr_final,
    float mlp_cov_lr_delay_mult,
    int mlp_cov_lr_max_steps,
    float mlp_color_lr_init, 
    float mlp_color_lr_final,
    float mlp_color_lr_delay_mult,
    int mlp_color_lr_max_steps,
    float mlp_featurebank_lr_init, 
    float mlp_featurebank_lr_final,
    float mlp_featurebank_lr_delay_mult,
    int mlp_featurebank_lr_max_steps,
    float appearance_lr_init, 
    float appearance_lr_final,
    float appearance_lr_delay_mult,
    int appearance_lr_max_steps,

    float percent_dense,
    float lambda_dssim,

    int start_stat, 
    int update_from,
    int update_interval,
    int update_until,

    float min_opacity, 
    float success_threshold,
    float densify_grad_threshold,

    float anchor_lr_init_coarse,
    float anchor_lr_final_coarse,
    float anchor_lr_delay_mult_coarse,
    int anchor_lr_max_steps_coarse,

    float offset_lr_init_coarse,
    float offset_lr_final_coarse,
    float offset_lr_delay_mult_coarse,
    int offset_lr_max_steps_coarse,

    float feature_lr_coarse,
    float opacity_lr_coarse,
    float scaling_lr_coarse,
    float rotation_lr_coarse,

    float mlp_opacity_lr_init_coarse, 
    float mlp_opacity_lr_final_coarse,
    float mlp_opacity_lr_delay_mult_coarse,
    int mlp_opacity_lr_max_steps_coarse,

    float mlp_cov_lr_init_coarse, 
    float mlp_cov_lr_final_coarse,
    float mlp_cov_lr_delay_mult_coarse1,
    int mlp_cov_lr_max_steps_coarse,

    float mlp_color_lr_init_coarse, 
    float mlp_color_lr_final_coarse,
    float mlp_color_lr_delay_mult_coarse,
    int mlp_color_lr_max_steps_coarse,

    float mlp_featurebank_lr_init_coarse, 
    float mlp_featurebank_lr_final_coarse,
    float mlp_featurebank_lr_delay_mult_coarse,
    int mlp_featurebank_lr_max_steps_coarse,

    float appearance_lr_init_coarse,
    float appearance_lr_final_coarse,
    float appearance_lr_delay_mult_coarse,
    int appearance_lr_max_steps_coarse)
    : iterations_(iterations),
      position_lr_init_(position_lr_init),
      position_lr_final_(position_lr_final),
      position_lr_delay_mult_(position_lr_delay_mult),
      position_lr_max_steps_(position_lr_max_steps),
      feature_lr_(feature_lr),
      opacity_lr_(opacity_lr_),
      scaling_lr_(scaling_lr),
      rotation_lr_(rotation_lr),
      percent_dense_(percent_dense),
      lambda_dssim_(lambda_dssim),

      offset_lr_init(offset_lr_init),
      offset_lr_final(offset_lr_final),
      offset_lr_delay_mult(offset_lr_delay_mult),
      offset_lr_max_steps(offset_lr_max_steps),
      mlp_opacity_lr_init(mlp_opacity_lr_init), 
      mlp_opacity_lr_final(mlp_opacity_lr_final),
      mlp_opacity_lr_delay_mult(mlp_opacity_lr_delay_mult),
      mlp_opacity_lr_max_steps(mlp_opacity_lr_max_steps),
      mlp_cov_lr_init(mlp_cov_lr_init), 
      mlp_cov_lr_final(mlp_cov_lr_final),
      mlp_cov_lr_delay_mult(mlp_cov_lr_delay_mult),
      mlp_cov_lr_max_steps(mlp_cov_lr_max_steps),
      mlp_color_lr_init(mlp_color_lr_init),
      mlp_color_lr_final(mlp_color_lr_final),
      mlp_color_lr_delay_mult(mlp_color_lr_delay_mult),
      mlp_color_lr_max_steps(mlp_color_lr_max_steps),
      mlp_featurebank_lr_init(mlp_featurebank_lr_init), 
      mlp_featurebank_lr_final(mlp_featurebank_lr_final),
      mlp_featurebank_lr_delay_mult(mlp_featurebank_lr_delay_mult),
      mlp_featurebank_lr_max_steps(mlp_featurebank_lr_max_steps),
      appearance_lr_init(appearance_lr_init), 
      appearance_lr_final(appearance_lr_final),
      appearance_lr_delay_mult(appearance_lr_delay_mult),
      appearance_lr_max_steps(appearance_lr_max_steps),
      start_stat(start_stat), 
      update_from(update_from),
      update_interval(update_interval),
      update_until(update_until),
      min_opacity(min_opacity), 
      success_threshold(success_threshold),
      densify_grad_threshold(densify_grad_threshold),

      anchor_lr_init_coarse(anchor_lr_init_coarse),
      anchor_lr_final_coarse(anchor_lr_final_coarse),
      anchor_lr_delay_mult_coarse(anchor_lr_delay_mult_coarse),
      anchor_lr_max_steps_coarse(anchor_lr_max_steps_coarse),

      offset_lr_init_coarse(offset_lr_init_coarse),
      offset_lr_final_coarse(offset_lr_final_coarse),
      offset_lr_delay_mult_coarse(offset_lr_delay_mult_coarse),
      offset_lr_max_steps_coarse(offset_lr_max_steps_coarse),

      feature_lr_coarse(feature_lr_coarse),
      opacity_lr_coarse(opacity_lr_coarse),
      scaling_lr_coarse(scaling_lr_coarse),
      rotation_lr_coarse(rotation_lr_coarse),

      mlp_opacity_lr_init_coarse(mlp_opacity_lr_init_coarse), 
      mlp_opacity_lr_final_coarse(mlp_opacity_lr_final_coarse),
      mlp_opacity_lr_delay_mult_coarse(mlp_opacity_lr_delay_mult_coarse),
      mlp_opacity_lr_max_steps_coarse(mlp_opacity_lr_max_steps_coarse),

      mlp_cov_lr_init_coarse(mlp_cov_lr_init_coarse), 
      mlp_cov_lr_final_coarse(mlp_cov_lr_final_coarse),
      mlp_cov_lr_delay_mult_coarse(mlp_cov_lr_delay_mult_coarse),
      mlp_cov_lr_max_steps_coarse(mlp_cov_lr_max_steps_coarse),

      mlp_color_lr_init_coarse(mlp_color_lr_init_coarse), 
      mlp_color_lr_final_coarse(mlp_color_lr_final_coarse),
      mlp_color_lr_delay_mult_coarse(mlp_color_lr_delay_mult_coarse),
      mlp_color_lr_max_steps_coarse(mlp_color_lr_max_steps_coarse),

      mlp_featurebank_lr_init_coarse(mlp_featurebank_lr_init_coarse), 
      mlp_featurebank_lr_final_coarse(mlp_featurebank_lr_final_coarse),
      mlp_featurebank_lr_delay_mult_coarse(mlp_featurebank_lr_delay_mult_coarse),
      mlp_featurebank_lr_max_steps_coarse(mlp_featurebank_lr_max_steps_coarse),

      appearance_lr_init_coarse(appearance_lr_init_coarse), 
      appearance_lr_final_coarse(appearance_lr_final_coarse),
      appearance_lr_delay_mult_coarse(appearance_lr_delay_mult_coarse),
      appearance_lr_max_steps_coarse(appearance_lr_max_steps_coarse)

{
}

void GaussianOptimizationParams::logger()
{
    std::cout << "[Gaussian Param]args: " << "iterations = " << iterations_
              << ", position_lr_init_ = " << position_lr_init_ << ", position_lr_final_ = " << position_lr_final_
              << ", position_lr_delay_mult_ = " << position_lr_delay_mult_ << ", position_lr_max_steps_ = " << position_lr_max_steps_
              << ", offset_lr_init = " << offset_lr_init << ", offset_lr_final = " << offset_lr_final
              << ", offset_lr_delay_mult = " << offset_lr_delay_mult << ", offset_lr_max_steps = " << offset_lr_max_steps
              << ", feature_lr_ = " << feature_lr_ << ", opacity_lr_ = " << opacity_lr_
              << ", scaling_lr_ = " << scaling_lr_ << ", rotation_lr_ = " << rotation_lr_
              << ", mlp_opacity_lr_init = " << mlp_opacity_lr_init << ", mlp_opacity_lr_final = " << mlp_opacity_lr_final
              << ", mlp_opacity_lr_delay_mult = " << mlp_opacity_lr_delay_mult << ", mlp_opacity_lr_max_steps = " << mlp_opacity_lr_max_steps
              << ", mlp_cov_lr_init = " << mlp_cov_lr_init << ", mlp_cov_lr_final = " << mlp_cov_lr_final
              << ", mlp_cov_lr_delay_mult = " << mlp_cov_lr_delay_mult << ", mlp_cov_lr_max_steps = " << mlp_cov_lr_max_steps
              << ", mlp_color_lr_init = " << mlp_color_lr_init << ", mlp_color_lr_final = " << mlp_color_lr_final
              << ", mlp_color_lr_delay_mult = " << mlp_color_lr_delay_mult << ", mlp_color_lr_max_steps = " << mlp_color_lr_max_steps

              << ", mlp_featurebank_lr_init = " << mlp_featurebank_lr_init << ", mlp_featurebank_lr_final = " << mlp_featurebank_lr_final
              << ", mlp_featurebank_lr_delay_mult = " << mlp_featurebank_lr_delay_mult << ", mlp_featurebank_lr_max_steps = " << mlp_featurebank_lr_max_steps
              << ", appearance_lr_init = " << appearance_lr_init << ", appearance_lr_final = " << appearance_lr_final
              << ", appearance_lr_delay_mult = " << appearance_lr_delay_mult << ", appearance_lr_max_steps = " << appearance_lr_max_steps
              << ", percent_dense_ = " << percent_dense_ << ", lambda_dssim_ = " << lambda_dssim_
              << ", start_stat = " << start_stat << ", update_from = " << update_from
              << ", update_interval = " << update_interval << ", update_until = " << update_until
              << ", min_opacity = " << min_opacity << ", success_threshold = " << success_threshold
              << ", densify_grad_threshold = " << densify_grad_threshold

              << std::endl;



}
