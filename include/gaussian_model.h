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

#include <memory>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cmath>

#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"

#include "third_party/simple-knn/spatial.h"
#include "third_party/tinyply/tinyply.h"
#include "types.h"
#include "point3d.h"
#include "operate_points.h"
#include "general_utils.h"
#include "sh_utils.h"
#include "tensor_utils.h"
#include "gaussian_parameters.h"

#include "mlp.h"
#include "embedding.h"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree_pointcloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/filters/passthrough.h>

#include <torchscatter/scatter.h>
#include <torchscatter/cuda/scatter_cuda.h>
#include <torch/script.h> 

using PointT = pcl::PointXYZRGB;
using PointCloudT = pcl::PointCloud<PointT>;

#define GAUSSIAN_MODEL_TENSORS_TO_VEC                        \
    this->Tensor_vec_anchor = {this->_anchor};              \
    this->Tensor_vec_offset = {this->_offset};              \
    this->Tensor_vec_anchor_feat = {this->_anchor_feat};    \
    this->Tensor_vec_opacity = {this->_opacity};            \
    this->Tensor_vec_scaling = {this->_scaling};            \
    this->Tensor_vec_rotation = {this->_rotation};          

#define GAUSSIAN_MODEL_INIT_TENSORS(device_type)                                             \
    this->_anchor = torch::empty(0, torch::TensorOptions().device(device_type));                    \
    this->_offset = torch::empty(0, torch::TensorOptions().device(device_type));                    \
    this->_anchor_feat = torch::empty(0, torch::TensorOptions().device(device_type));               \
    this->opacity_accum = torch::empty(0, torch::TensorOptions().device(device_type));             \
    this->_scaling = torch::empty(0, torch::TensorOptions().device(device_type));                   \
    this->_rotation = torch::empty(0, torch::TensorOptions().device(device_type));                  \
    this->_opacity = torch::empty(0, torch::TensorOptions().device(device_type));                   \
    this->max_radii2D = torch::empty(0, torch::TensorOptions().device(device_type));               \
    this->offset_gradient_accum = torch::empty(0, torch::TensorOptions().device(device_type));     \
    this->offset_denom = torch::empty(0, torch::TensorOptions().device(device_type));              \
    this->anchor_demon = torch::empty(0, torch::TensorOptions().device(device_type));              \
    this->mlp_opacity->to(this->device_type_);                                                     \
    this->mlp_cov->to(this->device_type_);                                                         \
    this->mlp_color->to(this->device_type_);                                                       \
    if(this->use_feat_bank) {        this->mlp_feature_bank->to(this->device_type_);     }  \
    GAUSSIAN_MODEL_TENSORS_TO_VEC
                                            
#define GAUSSIAN_MODEL_TENSORS_TO_VEC_COARSE                        \
    this->Tensor_vec_anchor_c = {this->_anchor_c};              \
    this->Tensor_vec_offset_c = {this->_offset_c};              \
    this->Tensor_vec_anchor_feat_c = {this->_anchor_feat_c};    \
    this->Tensor_vec_opacity_c = {this->_opacity_c};            \
    this->Tensor_vec_scaling_c = {this->_scaling_c};            \
    this->Tensor_vec_rotation_c = {this->_rotation_c};          

#define GAUSSIAN_MODEL_INIT_TENSORS_COARSE(device_type)                                             \
    this->_anchor_c = torch::empty(0, torch::TensorOptions().device(device_type));                    \
    this->_offset_c = torch::empty(0, torch::TensorOptions().device(device_type));                    \
    this->_anchor_feat_c = torch::empty(0, torch::TensorOptions().device(device_type));               \
    this->opacity_accum_c = torch::empty(0, torch::TensorOptions().device(device_type));             \
    this->_scaling_c = torch::empty(0, torch::TensorOptions().device(device_type));                   \
    this->_rotation_c = torch::empty(0, torch::TensorOptions().device(device_type));                  \
    this->_opacity_c = torch::empty(0, torch::TensorOptions().device(device_type));                   \
    this->max_radii2D_c = torch::empty(0, torch::TensorOptions().device(device_type));               \
    this->offset_gradient_accum_c = torch::empty(0, torch::TensorOptions().device(device_type));     \
    this->offset_denom_c = torch::empty(0, torch::TensorOptions().device(device_type));              \
    this->anchor_demon_c = torch::empty(0, torch::TensorOptions().device(device_type));              \
    GAUSSIAN_MODEL_TENSORS_TO_VEC_COARSE

class GaussianModel
{
public:
    GaussianModel(const int sh_degree);
    GaussianModel(const GaussianModelParams& model_params);

    std::shared_ptr<Embedding> get_appearance();
    torch::Tensor get_scaling();
    torch::nn::Sequential get_featurebank_mlp();
    torch::nn::Sequential get_opacity_mlp();
    torch::nn::Sequential get_cov_mlp();
    torch::nn::Sequential get_color_mlp();
    torch::Tensor get_rotation();
    torch::Tensor get_anchor();
    torch::Tensor get_opacity();
    torch::Tensor get_covariance(int scaling_modifier = 1);

    void oneUpShDegree();
    void setShDegree(const int sh);

    void createCoarseAnchorFromPcd(torch::Tensor points);
    void createFromPcd(
        std::map<point3D_id_t, Point3D> pcd,
        const float spatial_lr_scale);

    void increasePcdCoarse(torch::Tensor new_point_cloud);
    void increasePcd(std::vector<float> points, std::vector<float> colors, const int iteration);
    void increasePcd(torch::Tensor& new_point_cloud, torch::Tensor& new_colors, const int iteration);

    void applyScaledTransformation(
        const float s = 1.0,
        const Sophus::SE3f T = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()));
    void scaledTransformationPostfix(
        torch::Tensor& new_xyz,
        torch::Tensor& new_scaling);

    void scaledTransformVisiblePointsOfKeyframe(
        torch::Tensor& point_not_transformed_flags,
        torch::Tensor& diff_pose,
        torch::Tensor& kf_world_view_transform,
        torch::Tensor& kf_full_proj_transform,
        const int kf_creation_iter,
        const int stable_num_iter_existence,
        int& num_transformed,
        const float scale = 1.0f);

    void trainingSetup(const GaussianOptimizationParams& training_args);
    float updateLearningRate(int step);
    void setPositionLearningRate(float position_lr);
    void setOpacityLearningRate(float opacity_lr);
    void setScalingLearningRate(float scaling_lr);
    void setRotationLearningRate(float rot_lr);
    void setAnchorFeatureLearningRate(float feature_lr);

    void resetOpacity();
    torch::Tensor replaceTensorToOptimizer(torch::Tensor& t, int tensor_idx);

    void loadPly(std::filesystem::path ply_path);
    void savePly(std::filesystem::path result_path);
    void saveSparsePointsPly(std::filesystem::path result_path);
    void save_mlp_checkpoints(std::filesystem::path result_path);

    float percentDense();
    void setPercentDense(const float percent_dense);

    void setApperance();

    void eval();

    void train();

    void training_statis(
        torch::Tensor &viewspace_point_tensor,
        torch::Tensor &opacity,
        torch::Tensor &update_filter,
        torch::Tensor &offset_selection_mask,
        torch::Tensor &anchor_visible_mask);

    void prune_anchor(torch::Tensor &mask);

    void anchor_growing(torch::Tensor &grads, float threshold, torch::Tensor &offset_mask);

    void adjust_anchor(
        int check_interval = 100, 
        float success_threshold = 0.8, 
        float grad_threshold = 0.0002, 
        float min_opacity = 0.005);

    torch::Tensor inverse_sigmoid(torch::Tensor x);

    void densificationPostfix(
        torch::Tensor &new_anchor,
        torch::Tensor &new_offsets,
        torch::Tensor &new_feat,
        torch::Tensor &new_opacities,
        torch::Tensor &new_scaling,
        torch::Tensor &new_rotation);

    void densificationPostfixCoarse(
        torch::Tensor &new_anchor,
        torch::Tensor &new_offsets,
        torch::Tensor &new_feat,
        torch::Tensor &new_opacities,
        torch::Tensor &new_scaling,
        torch::Tensor &new_rotation);

protected:
    float exponLrFunc(int step);
    float getExponLrFunc(int step,
                         float lr_init, float lr_final,
                         float lr_delay_mult, int max_steps);

public:
    torch::DeviceType device_type_;

    int active_sh_degree_;
    int max_sh_degree_;

    torch::Tensor sparse_points_xyz_;
    torch::Tensor sparse_points_color_;

    int feat_dim;
    int n_offsets;
    float voxel_size;
    int update_depth;
    int update_init_factor;
    int update_hierachy_factor;
    bool use_feat_bank;

    int embedding_dim;
    int appearance_dim;
    int ratio;
    bool add_opacity_dist;
    bool add_cov_dist;
    bool add_color_dist;

    torch::Tensor _anchor;
    torch::Tensor _offset;
    torch::Tensor _anchor_feat;

    torch::Tensor opacity_accum;

    torch::Tensor _scaling;
    torch::Tensor _rotation;
    torch::Tensor _opacity;
    torch::Tensor max_radii2D;

    torch::Tensor offset_gradient_accum;
    torch::Tensor offset_denom;

    torch::Tensor anchor_demon;

    std::vector<torch::Tensor>
        Tensor_vec_anchor,
        Tensor_vec_offset,
        Tensor_vec_anchor_feat,
        Tensor_vec_scaling,
        Tensor_vec_rotation,
        Tensor_vec_opacity;

    std::shared_ptr<torch::optim::Adam> optimizer_; 
    float percent_dense_;
    float spatial_lr_scale_;

    int opacity_dist_dim = 1;
    int cov_dist_dim = 1;
    int color_dist_dim = 1;

    torch::nn::Sequential mlp_feature_bank;
    torch::nn::Sequential mlp_opacity;
    torch::nn::Sequential mlp_cov;
    torch::nn::Sequential mlp_color;
    std::shared_ptr<Embedding> embedding_appearance;
    torch::nn::Sequential mlp_apperance;
    bool use_mlp_apperance = true;

    bool use_coarse_anchor = false;
    float coarse_voxel_size = 0.2;
    int feat_dim_coarse;
    int n_offsets_coarse;
    int appearance_dim_coarse;
    torch::Tensor _anchor_c;
    torch::Tensor _offset_c;
    torch::Tensor _anchor_feat_c;

    torch::Tensor opacity_accum_c;

    torch::Tensor _scaling_c;
    torch::Tensor _rotation_c;
    torch::Tensor _opacity_c;
    torch::Tensor max_radii2D_c;

    torch::Tensor offset_gradient_accum_c;
    torch::Tensor offset_denom_c;

    torch::Tensor anchor_demon_c;

    std::vector<torch::Tensor>
        Tensor_vec_anchor_c,
        Tensor_vec_offset_c,
        Tensor_vec_anchor_feat_c,
        Tensor_vec_scaling_c,
        Tensor_vec_rotation_c,
        Tensor_vec_opacity_c;

    torch::nn::Sequential mlp_feature_bank_c;
    torch::nn::Sequential mlp_opacity_c;
    torch::nn::Sequential mlp_cov_c;
    torch::nn::Sequential mlp_color_c;
    torch::nn::Sequential mlp_apperance_c;

    bool hasSLAM = false;

protected:
    float lr_init_;
    float lr_final_;
    int lr_delay_steps_;
    float lr_delay_mult_;
    int max_steps_;

    float offset_lr_init;
    float offset_lr_final;
    float offset_lr_delay_mult;
    int offset_lr_max_steps;

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

    std::mutex mutex_settings_;

    float anchor_lr_init_coarse;
    float anchor_lr_final_coarse;
    float anchor_lr_delay_mult_coarse;
    int anchor_lr_max_steps_coarse;

    float offset_lr_init_coarse;
    float offset_lr_final_coarse;
    float offset_lr_delay_mult_coarse;
    int offset_lr_max_steps_coarse;

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
