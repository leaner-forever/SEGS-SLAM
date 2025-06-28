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

#include "include/gaussian_model.h"
#include <filesystem>

GaussianModel::GaussianModel(const int sh_degree)
    : active_sh_degree_(0), spatial_lr_scale_(0.0),
      lr_delay_steps_(0), lr_delay_mult_(1.0), max_steps_(1000000)
{
    this->max_sh_degree_ = sh_degree;

    if (torch::cuda::is_available())
        this->device_type_ = torch::kCUDA;
    else
        this->device_type_ = torch::kCPU;

    GAUSSIAN_MODEL_INIT_TENSORS(this->device_type_)
}

GaussianModel::GaussianModel(const GaussianModelParams &model_params)
    : active_sh_degree_(0), spatial_lr_scale_(0.0),
      lr_delay_steps_(0), lr_delay_mult_(1.0), max_steps_(1000000)
{
    std::cout << "[Gaussian Model]Model creating..." << std::endl;

    this->max_sh_degree_ = model_params.sh_degree_;

    if (model_params.data_device_ == "cuda")
        this->device_type_ = torch::kCUDA;
    else
        this->device_type_ = torch::kCPU;

    this->feat_dim = model_params.feat_dim;
    this->n_offsets = model_params.n_offsets;
    this->voxel_size =model_params.voxel_size;
    this->update_depth = model_params.update_depth;
    this->update_init_factor = model_params.update_init_factor;
    this->update_hierachy_factor = model_params.update_hierachy_factor;
    this->use_feat_bank = model_params.use_feat_bank;

    this->appearance_dim = model_params.appearance_dim;
    this->ratio = model_params.ratio;
    this->add_opacity_dist = model_params.add_opacity_dist;
    this->add_cov_dist = model_params.add_cov_dist;
    this->add_color_dist = model_params.add_color_dist;
    this->embedding_dim = model_params.embedding_dim;

    if (add_opacity_dist) opacity_dist_dim = 1;  else opacity_dist_dim = 0;
    this->mlp_opacity = torch::nn::Sequential(
        torch::nn::Linear(feat_dim + 3 + opacity_dist_dim, feat_dim),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Linear(feat_dim, n_offsets),
        torch::nn::Tanh()
    );
    this->mlp_opacity->to(torch::kCUDA);
    if (add_cov_dist) cov_dist_dim = 1;  else cov_dist_dim = 0;
    this->mlp_cov = torch::nn::Sequential(
        torch::nn::Linear(feat_dim + 3 + cov_dist_dim, feat_dim),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Linear(feat_dim, 7 * n_offsets)
    );
    this->mlp_cov->to(torch::kCUDA);
    if (add_color_dist) color_dist_dim = 1;  else color_dist_dim = 0;
    this->mlp_color = torch::nn::Sequential(
        torch::nn::Linear(feat_dim + 3 + color_dist_dim + appearance_dim, feat_dim),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Linear(feat_dim, 3 * n_offsets),
        torch::nn::Sigmoid()
    );
    this->mlp_color->to(torch::kCUDA);
    this->mlp_apperance = torch::nn::Sequential(
        torch::nn::Linear(7, appearance_dim)
    );

    this->mlp_apperance->to(torch::kCUDA);

    if(this->use_feat_bank) {
        this->mlp_feature_bank = torch::nn::Sequential(
            torch::nn::Linear(3 + 1, feat_dim),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
            torch::nn::Linear(feat_dim, 3),
            torch::nn::Softmax(torch::nn::SoftmaxOptions(1)));
        this->mlp_feature_bank->to(this->device_type_);
    } 
    this->mlp_feature_bank->to(torch::kCUDA);
    
    GAUSSIAN_MODEL_INIT_TENSORS(this->device_type_)

    this->use_coarse_anchor = model_params.use_coarse_anchor;
    std::cout << "[Gaussian Model]use coarse anchor" << use_coarse_anchor << std::endl;
    if( this->use_coarse_anchor)
    {
        this->feat_dim_coarse = model_params.feat_dim_coarse;
        this->n_offsets_coarse = model_params.n_offsets_coarse;
        this->coarse_voxel_size =model_params.coarse_voxel_size;
        this->appearance_dim_coarse = model_params.appearance_dim_coarse;
    
        if (add_opacity_dist) opacity_dist_dim = 1;  else opacity_dist_dim = 0;
        this->mlp_opacity_c = torch::nn::Sequential(
            torch::nn::Linear(feat_dim_coarse + 3 + opacity_dist_dim, feat_dim_coarse),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
            torch::nn::Linear(feat_dim_coarse, n_offsets_coarse),
            torch::nn::Tanh()
        );
        this->mlp_opacity_c->to(torch::kCUDA);
        if (add_cov_dist) cov_dist_dim = 1;  else cov_dist_dim = 0;
        this->mlp_cov_c = torch::nn::Sequential(
            torch::nn::Linear(feat_dim_coarse + 3 + cov_dist_dim, feat_dim_coarse),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
            torch::nn::Linear(feat_dim_coarse, 7 * n_offsets_coarse)
        );
        this->mlp_cov_c->to(torch::kCUDA);
        if (add_color_dist) color_dist_dim = 1;  else color_dist_dim = 0;
        this->mlp_color_c = torch::nn::Sequential(
            torch::nn::Linear(feat_dim_coarse + 3 + color_dist_dim + appearance_dim_coarse, feat_dim_coarse),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
            torch::nn::Linear(feat_dim_coarse, 3 * n_offsets_coarse),
            torch::nn::Sigmoid()
        );
        this->mlp_color_c->to(torch::kCUDA); 

        this->mlp_apperance_c = torch::nn::Sequential(
            torch::nn::Linear(7, appearance_dim_coarse)
        );
        this->mlp_apperance_c->to(torch::kCUDA);

        if(this->use_feat_bank) {
            this->mlp_feature_bank_c = torch::nn::Sequential(
                torch::nn::Linear(3 + 1, feat_dim_coarse),
                torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
                torch::nn::Linear(feat_dim_coarse, 3),
                torch::nn::Softmax(torch::nn::SoftmaxOptions(1)));
            this->mlp_feature_bank_c->to(this->device_type_);
        } 
        this->mlp_feature_bank_c->to(torch::kCUDA);

        GAUSSIAN_MODEL_INIT_TENSORS_COARSE(this->device_type_)

        std::cout << "[Gaussian Model]use coarse anchor" << std::endl;
    }

    torch::nn::Sequential Mlp_opacity = torch::nn::Sequential(
        torch::nn::Linear(feat_dim + 3 + opacity_dist_dim, feat_dim),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Linear(feat_dim, n_offsets),
        torch::nn::Tanh()
    );
    Mlp_opacity->to(torch::kCUDA);

    torch::nn::Sequential Mlp_cov = torch::nn::Sequential(
        torch::nn::Linear(feat_dim + 3 + cov_dist_dim, feat_dim),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Linear(feat_dim, 7 * n_offsets)
    );
    Mlp_cov->to(torch::kCUDA);

    torch::nn::Sequential Mlp_color = torch::nn::Sequential(
        torch::nn::Linear(feat_dim + 3 + color_dist_dim + appearance_dim, feat_dim),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Linear(feat_dim, 3 * n_offsets),
        torch::nn::Sigmoid()
    );
    Mlp_color->to(torch::kCUDA);

    std::cout << "[Gaussian Model]Model created!" << std::endl;
}

std::shared_ptr<Embedding> GaussianModel::get_appearance()
{
    return this->embedding_appearance;
}

torch::Tensor GaussianModel::get_scaling()
{
    return torch::exp(this->_scaling);
}

torch::nn::Sequential GaussianModel::get_featurebank_mlp()
{
    return this->mlp_feature_bank;
}

torch::nn::Sequential GaussianModel::get_opacity_mlp()
{
    return this->mlp_opacity;
}

torch::nn::Sequential GaussianModel::get_cov_mlp()
{
    return this->mlp_cov;
}

torch::nn::Sequential GaussianModel::get_color_mlp()
{
    return this->mlp_color;
}

torch::Tensor GaussianModel::get_rotation()
{
    return torch::nn::functional::normalize(this->_rotation);
}

torch::Tensor GaussianModel::get_anchor()
{
    return this->_anchor;
}

torch::Tensor GaussianModel::get_opacity()
{
    return torch::sigmoid(this->_opacity);
}

torch::Tensor GaussianModel::get_covariance(int scaling_modifier)
{
    auto r = this->_rotation;
    auto R = general_utils::build_rotation(r);

    auto s = scaling_modifier * this->get_scaling();
    auto L = torch::zeros({s.size(0), 3, 3}, torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
    L.select(1, 0).select(1, 0).copy_(s.index({torch::indexing::Slice(), 0}));
    L.select(1, 1).select(1, 1).copy_(s.index({torch::indexing::Slice(), 1}));
    L.select(1, 2).select(1, 2).copy_(s.index({torch::indexing::Slice(), 2}));
    L = R.matmul(L); 

    auto actual_covariance = L.matmul(L.transpose(1, 2));
    auto symm_uncertainty = torch::zeros({actual_covariance.size(0), 6}, torch::TensorOptions().dtype(torch::kFloat).device(device_type_));

    symm_uncertainty.select(1, 0).copy_(actual_covariance.index({torch::indexing::Slice(), 0, 0}));
    symm_uncertainty.select(1, 1).copy_(actual_covariance.index({torch::indexing::Slice(), 0, 1}));
    symm_uncertainty.select(1, 2).copy_(actual_covariance.index({torch::indexing::Slice(), 0, 2}));
    symm_uncertainty.select(1, 3).copy_(actual_covariance.index({torch::indexing::Slice(), 1, 1}));
    symm_uncertainty.select(1, 4).copy_(actual_covariance.index({torch::indexing::Slice(), 1, 2}));
    symm_uncertainty.select(1, 5).copy_(actual_covariance.index({torch::indexing::Slice(), 2, 2}));

    return symm_uncertainty;
}

void GaussianModel::oneUpShDegree()
{
    if (this->active_sh_degree_ < this->max_sh_degree_)
        this->active_sh_degree_ += 1;
}

void GaussianModel::setShDegree(const int sh)
{
    this->active_sh_degree_ = (sh > this->max_sh_degree_ ? this->max_sh_degree_ : sh);
}

void saveTensorToTxt(const torch::Tensor& tensor, const std::string& filename) {
    auto tensor_cpu = tensor.clone().to(torch::kCPU);
    auto sizes = tensor_cpu.sizes();
    std::vector<float> data(tensor_cpu.numel());
    std::memcpy(data.data(), tensor_cpu.data_ptr(), data.size() * sizeof(float));

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file " << filename << " for writing." << std::endl;
        return;
    }

    file << std::fixed << std::setprecision(5); 
    for (size_t i = 0; i < sizes[0]; ++i) {
        for (size_t j = 0; j < sizes[1]; ++j) {
            file << data[i * sizes[1] + j];
            if (j < sizes[1] - 1) {
                file << " ";
            }
        }
        file << "\n";
    }

    file.close();
}

void GaussianModel::createCoarseAnchorFromPcd(torch::Tensor points)
{
    auto scaled_data = torch::round(points / this->coarse_voxel_size);
    auto unique_data = std::get<0>(torch::unique_dim(scaled_data, 0, true, true, true));
    torch::Tensor fused_point_cloud = (unique_data * voxel_size).to(torch::kFloat).to(device_type_);

    torch::Tensor offsets = torch::zeros(
        {fused_point_cloud.size(0), this->n_offsets, 3},
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
    torch::Tensor anchors_feat = torch::zeros(
        {fused_point_cloud.size(0), feat_dim},
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));


    torch::Tensor point_cloud_copy = fused_point_cloud.clone();
    torch::Tensor dist2 = torch::clamp_min(distCUDA2(point_cloud_copy), 0.0000001);
    torch::Tensor scales = torch::log(torch::sqrt(dist2));
    auto scales_ndimension = scales.ndimension();
    scales = scales.unsqueeze(scales_ndimension).repeat({1, 6});
    torch::Tensor rots = torch::zeros({fused_point_cloud.size(0), 4}, torch::TensorOptions().device(device_type_));
    rots.index({torch::indexing::Slice(), 0}) = 1;

    torch::Tensor opacities = general_utils::inverse_sigmoid(
        0.1f * torch::ones(
                   {fused_point_cloud.size(0), 1},
                   torch::TensorOptions().dtype(torch::kFloat).device(device_type_)));

    this->_anchor_c = fused_point_cloud.requires_grad_();
    this->_offset_c = offsets.requires_grad_();
    this->_anchor_feat_c = anchors_feat.requires_grad_();
    this->_scaling_c = scales.requires_grad_();
    this->_rotation_c = rots.set_requires_grad(false);
    this->_opacity_c = opacities.set_requires_grad(false);

    GAUSSIAN_MODEL_TENSORS_TO_VEC_COARSE

    this->max_radii2D_c = torch::zeros({this->get_anchor().size(0)}, torch::TensorOptions().device(device_type_));
}

void GaussianModel::createFromPcd(
    std::map<point3D_id_t, Point3D> pcd,
    const float spatial_lr_scale)
{
    this->spatial_lr_scale_ = spatial_lr_scale;
    int num_points = static_cast<int>(pcd.size());
    torch::Tensor points = torch::zeros(
        {num_points, 3}, 
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
    auto pcd_it = pcd.begin();
    for (int point_idx = 0; point_idx < num_points; ++point_idx) {
        auto& point = (*pcd_it).second;
        points.index({point_idx, 0}) = point.xyz_(0);
        points.index({point_idx, 1}) = point.xyz_(1);
        points.index({point_idx, 2}) = point.xyz_(2);
        ++pcd_it;
    }
    auto scaled_data = torch::round(points / this->voxel_size);
    auto unique_data = std::get<0>(torch::unique_dim(scaled_data, 0, true, true, true));
    torch::Tensor fused_point_cloud = (unique_data * voxel_size).to(torch::kFloat).to(device_type_);

    torch::Tensor offsets = torch::zeros(
        {fused_point_cloud.size(0), this->n_offsets, 3},
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
    torch::Tensor anchors_feat = torch::zeros(
        {fused_point_cloud.size(0), feat_dim},
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));

    torch::Tensor point_cloud_copy = fused_point_cloud.clone();
    torch::Tensor dist2 = torch::clamp_min(distCUDA2(point_cloud_copy), 0.0000001);
    torch::Tensor scales = torch::log(torch::sqrt(dist2));
    auto scales_ndimension = scales.ndimension();
    scales = scales.unsqueeze(scales_ndimension).repeat({1, 6});
    torch::Tensor rots = torch::zeros({fused_point_cloud.size(0), 4}, torch::TensorOptions().device(device_type_));
    rots.index({torch::indexing::Slice(), 0}) = 1;

    torch::Tensor opacities = general_utils::inverse_sigmoid(
        0.1f * torch::ones(
                   {fused_point_cloud.size(0), 1},
                   torch::TensorOptions().dtype(torch::kFloat).device(device_type_)));

    this->_anchor = fused_point_cloud.requires_grad_();
    this->_offset = offsets.requires_grad_();
    this->_anchor_feat = anchors_feat.requires_grad_();
    this->_scaling = scales.requires_grad_();
    this->_rotation = rots.set_requires_grad(false);
    this->_opacity = opacities.set_requires_grad(false);

    GAUSSIAN_MODEL_TENSORS_TO_VEC

    this->max_radii2D = torch::zeros({this->get_anchor().size(0)}, torch::TensorOptions().device(device_type_));

    if( use_coarse_anchor)
        createCoarseAnchorFromPcd(points);
}

void GaussianModel::increasePcdCoarse(torch::Tensor new_point_cloud)
{
    auto scaled_data = torch::round(new_point_cloud / this->voxel_size);
    auto unique_data = std::get<0>(torch::unique_dim(scaled_data, 0, true, true, true));
    torch::Tensor fused_point_cloud = (unique_data * voxel_size).to(torch::kFloat).to(device_type_);

    torch::Tensor offsets = torch::zeros(
        {fused_point_cloud.size(0), this->n_offsets, 3},
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
    torch::Tensor anchors_feat = torch::zeros(
        {fused_point_cloud.size(0), feat_dim},
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));

    torch::Tensor point_cloud_copy = fused_point_cloud.clone();
    torch::Tensor dist2 = torch::clamp_min(distCUDA2(point_cloud_copy), 0.0000001);
    torch::Tensor scales = torch::log(torch::sqrt(dist2));
    auto scales_ndimension = scales.ndimension();
    scales = scales.unsqueeze(scales_ndimension).repeat({1, 6});
    torch::Tensor rots = torch::zeros({fused_point_cloud.size(0), 4}, torch::TensorOptions().device(device_type_));
    rots.index({torch::indexing::Slice(), 0}) = 1;

    torch::Tensor opacities = general_utils::inverse_sigmoid(
        0.1f * torch::ones(
                   {fused_point_cloud.size(0), 1},
                   torch::TensorOptions().dtype(torch::kFloat).device(device_type_)));

    auto zeros_tensor = torch::zeros({opacities.size(0), 1},
                                    torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    auto temp_anchor_demon = torch::cat({this->anchor_demon, zeros_tensor}, 0);
    this->anchor_demon_c = temp_anchor_demon;

    auto temp_opacity_accum= torch::cat({this->opacity_accum, torch::zeros({opacities.size(0), 1},
                                    torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA))},0);
    this->opacity_accum_c = temp_opacity_accum;

    auto temp_offset_denom= torch::cat({this->offset_denom, torch::zeros({opacities.size(0)*this->n_offsets, 1},
                                    torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA))},0);
    this->offset_denom_c = temp_offset_denom;

    auto temp_offset_gradient_accum= torch::cat({this->offset_gradient_accum, torch::zeros({opacities.size(0)*this->n_offsets, 1},
                                    torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA))},0);
    this->offset_gradient_accum_c = temp_offset_gradient_accum;

    auto new_anchor = fused_point_cloud;
    auto new_offsets = offsets;
    auto new_feat = anchors_feat;
    auto new_opacities = opacities;
    auto new_scaling = scales;
    auto new_rotation = rots;

    densificationPostfixCoarse(
        new_anchor,
        new_offsets,
        new_feat,
        new_opacities,
        new_scaling,
        new_rotation
    );
}

void GaussianModel::increasePcd(std::vector<float> points, std::vector<float> colors, const int iteration)
{
    assert(points.size() == colors.size());
    assert(points.size() % 3 == 0);
    auto num_new_points = static_cast<int>(points.size() / 3);
    if (num_new_points == 0)
        return;

    torch::Tensor new_point_cloud = torch::from_blob(
        points.data(), {num_new_points, 3},
        torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);

    auto scaled_data = torch::round(new_point_cloud / this->voxel_size);
    auto unique_data = std::get<0>(torch::unique_dim(scaled_data, 0, true, true, true));
    torch::Tensor fused_point_cloud = (unique_data * voxel_size).to(torch::kFloat).to(device_type_);

    torch::Tensor offsets = torch::zeros(
        {fused_point_cloud.size(0), this->n_offsets, 3},
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
    torch::Tensor anchors_feat = torch::zeros(
        {fused_point_cloud.size(0), feat_dim},
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));

    torch::Tensor point_cloud_copy = fused_point_cloud.clone();
    torch::Tensor dist2 = torch::clamp_min(distCUDA2(point_cloud_copy), 0.0000001);
    torch::Tensor scales = torch::log(torch::sqrt(dist2));
    auto scales_ndimension = scales.ndimension();
    scales = scales.unsqueeze(scales_ndimension).repeat({1, 6});
    torch::Tensor rots = torch::zeros({fused_point_cloud.size(0), 4}, torch::TensorOptions().device(device_type_));
    rots.index({torch::indexing::Slice(), 0}) = 1;

    torch::Tensor opacities = general_utils::inverse_sigmoid(
        0.1f * torch::ones(
                   {fused_point_cloud.size(0), 1},
                   torch::TensorOptions().dtype(torch::kFloat).device(device_type_)));

    auto zeros_tensor = torch::zeros({opacities.size(0), 1},
                                    torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    auto temp_anchor_demon = torch::cat({this->anchor_demon, zeros_tensor}, 0);
    this->anchor_demon = temp_anchor_demon;

    auto temp_opacity_accum= torch::cat({this->opacity_accum, torch::zeros({opacities.size(0), 1},
                                    torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA))},0);
    this->opacity_accum = temp_opacity_accum;

    auto temp_offset_denom= torch::cat({this->offset_denom, torch::zeros({opacities.size(0)*this->n_offsets, 1},
                                    torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA))},0);
    this->offset_denom = temp_offset_denom;

    auto temp_offset_gradient_accum= torch::cat({this->offset_gradient_accum, torch::zeros({opacities.size(0)*this->n_offsets, 1},
                                    torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA))},0);
    this->offset_gradient_accum = temp_offset_gradient_accum;
    
    auto new_anchor = fused_point_cloud;
    auto new_offsets = offsets;
    auto new_feat = anchors_feat;
    auto new_opacities = opacities;
    auto new_scaling = scales;
    auto new_rotation = rots;

    densificationPostfix(
        new_anchor,
        new_offsets,
        new_feat,
        new_opacities,
        new_scaling,
        new_rotation
    );

    if( use_coarse_anchor)
        increasePcdCoarse(new_point_cloud);

    c10::cuda::CUDACachingAllocator::emptyCache();
}

void GaussianModel::increasePcd(torch::Tensor& new_point_cloud, torch::Tensor& new_colors, const int iteration)
{
    auto num_new_points = new_point_cloud.size(0);
    if (num_new_points == 0)
        return;

    if (sparse_points_xyz_.size(0) == 0) {
        sparse_points_xyz_ = new_point_cloud;
        sparse_points_color_ = new_colors;
    }
    else {
        sparse_points_xyz_ = torch::cat({sparse_points_xyz_, new_point_cloud}, 0);
        sparse_points_color_ = torch::cat({sparse_points_color_, new_colors}, 0);
    }

    auto scaled_data = torch::round(new_point_cloud / this->voxel_size);
    auto unique_data = std::get<0>(torch::unique_dim(scaled_data, 0, true, true, true));
    torch::Tensor fused_point_cloud = (unique_data * voxel_size).to(torch::kFloat).to(device_type_);

    torch::Tensor offsets = torch::zeros(
        {fused_point_cloud.size(0), this->n_offsets, 3},
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
    torch::Tensor anchors_feat = torch::zeros(
        {fused_point_cloud.size(0), feat_dim},
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));

    torch::Tensor point_cloud_copy = fused_point_cloud.clone();
    torch::Tensor dist2 = torch::clamp_min(distCUDA2(point_cloud_copy), 0.0000001);
    torch::Tensor scales = torch::log(torch::sqrt(dist2));
    auto scales_ndimension = scales.ndimension();
    scales = scales.unsqueeze(scales_ndimension).repeat({1, 6});
    torch::Tensor rots = torch::zeros({fused_point_cloud.size(0), 4}, torch::TensorOptions().device(device_type_));
    rots.index({torch::indexing::Slice(), 0}) = 1;

    torch::Tensor opacities = general_utils::inverse_sigmoid(
        0.1f * torch::ones(
                   {fused_point_cloud.size(0), 1},
                   torch::TensorOptions().dtype(torch::kFloat).device(device_type_)));

    auto zeros_tensor = torch::zeros({opacities.size(0), 1},
                                    torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    auto temp_anchor_demon = torch::cat({this->anchor_demon, zeros_tensor}, 0);
    this->anchor_demon = temp_anchor_demon;

    auto temp_opacity_accum= torch::cat({this->opacity_accum, torch::zeros({opacities.size(0), 1},
                                    torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA))},0);
    this->opacity_accum = temp_opacity_accum;

    auto temp_offset_denom= torch::cat({this->offset_denom, torch::zeros({opacities.size(0)*this->n_offsets, 1},
                                    torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA))},0);
    this->offset_denom = temp_offset_denom;

    auto temp_offset_gradient_accum= torch::cat({this->offset_gradient_accum, torch::zeros({opacities.size(0)*this->n_offsets, 1},
                                    torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA))},0);
    this->offset_gradient_accum = temp_offset_gradient_accum;

    auto new_anchor = fused_point_cloud;
    auto new_offsets = offsets;
    auto new_feat = anchors_feat;
    auto new_opacities = opacities;
    auto new_scaling = scales;
    auto new_rotation = rots;

    densificationPostfix(
        new_anchor,
        new_offsets,
        new_feat,
        new_opacities,
        new_scaling,
        new_rotation
    );

    if( use_coarse_anchor)
        increasePcdCoarse(new_point_cloud);

    c10::cuda::CUDACachingAllocator::emptyCache();
}

void GaussianModel::applyScaledTransformation(
    const float s,
    const Sophus::SE3f T)
{
}

void GaussianModel::scaledTransformationPostfix(
    torch::Tensor& new_xyz,
    torch::Tensor& new_scaling)
{
}

void GaussianModel::scaledTransformVisiblePointsOfKeyframe(
    torch::Tensor& point_not_transformed_flags,
    torch::Tensor& diff_pose,
    torch::Tensor& kf_world_view_transform,
    torch::Tensor& kf_full_proj_transform,
    const int kf_creation_iter,
    const int stable_num_iter_existence,
    int& num_transformed,
    const float scale)
{
}

void GaussianModel::trainingSetup(const GaussianOptimizationParams& training_args)
{
    setPercentDense(training_args.percent_dense_);

    this->opacity_accum = torch::zeros({this->get_anchor().size(0), 1}, torch::TensorOptions().device(device_type_));

    this->offset_gradient_accum = torch::zeros({this->get_anchor().size(0) * this->n_offsets, 1},
                                               torch::TensorOptions().device(device_type_));
    this->offset_denom = torch::zeros({this->get_anchor().size(0) * this->n_offsets, 1},
                                      torch::TensorOptions().device(device_type_));
    this->anchor_demon = torch::zeros({this->get_anchor().size(0), 1}, torch::TensorOptions().device(device_type_));

    torch::optim::AdamOptions adam_options;
    adam_options.set_lr(0.0);
    adam_options.eps() = 1e-15;

    this->optimizer_.reset(new torch::optim::Adam(Tensor_vec_anchor, adam_options));
    optimizer_->param_groups()[0].options().set_lr(training_args.position_lr_init_ * this->spatial_lr_scale_);

    optimizer_->add_param_group(Tensor_vec_offset);
    optimizer_->param_groups()[1].options().set_lr(training_args.offset_lr_init* this->spatial_lr_scale_);

    optimizer_->add_param_group(Tensor_vec_anchor_feat);
    optimizer_->param_groups()[2].options().set_lr(training_args.feature_lr_ );

    optimizer_->add_param_group(Tensor_vec_opacity);
    optimizer_->param_groups()[3].options().set_lr(training_args.opacity_lr_);

    optimizer_->add_param_group(Tensor_vec_scaling);
    optimizer_->param_groups()[4].options().set_lr(training_args.scaling_lr_);

    optimizer_->add_param_group(Tensor_vec_rotation);
    optimizer_->param_groups()[5].options().set_lr(training_args.rotation_lr_);

    optimizer_->add_param_group(mlp_opacity->parameters());
    optimizer_->param_groups()[6].options().set_lr(training_args.mlp_opacity_lr_init);

    optimizer_->add_param_group(mlp_cov->parameters());
    optimizer_->param_groups()[7].options().set_lr(training_args.mlp_cov_lr_init);

    optimizer_->add_param_group(mlp_color->parameters());
    optimizer_->param_groups()[8].options().set_lr(training_args.mlp_color_lr_init);

    if(use_feat_bank)
    {
        if(appearance_dim > 0 )
        {
            optimizer_->add_param_group(embedding_appearance->parameters());
            optimizer_->param_groups()[9].options().set_lr(training_args.appearance_lr_init);
            optimizer_->add_param_group(mlp_apperance->parameters());
            optimizer_->param_groups()[10].options().set_lr(training_args.appearance_lr_init);
            optimizer_->add_param_group(mlp_feature_bank->parameters());
            optimizer_->param_groups()[11].options().set_lr(training_args.mlp_featurebank_lr_init);
        }
        else
            throw std::runtime_error("appearance_dim shoud > 0!");

    }
    else if (appearance_dim > 0) {
        optimizer_->add_param_group(embedding_appearance->parameters());
        optimizer_->param_groups()[9].options().set_lr(training_args.appearance_lr_init);

        optimizer_->add_param_group(mlp_apperance->parameters());
        optimizer_->param_groups()[10].options().set_lr(training_args.appearance_lr_init);
    }

    if( use_coarse_anchor)
    {
        if(use_feat_bank)
        {
            if(appearance_dim > 0 )
            {
                optimizer_->add_param_group(Tensor_vec_anchor_c);
                optimizer_->param_groups()[12].options().set_lr(training_args.anchor_lr_init_coarse * this->spatial_lr_scale_);

                optimizer_->add_param_group(Tensor_vec_offset_c);
                optimizer_->param_groups()[13].options().set_lr(training_args.offset_lr_init_coarse* this->spatial_lr_scale_);

                optimizer_->add_param_group(Tensor_vec_anchor_feat_c);
                optimizer_->param_groups()[14].options().set_lr(training_args.feature_lr_coarse );

                optimizer_->add_param_group(Tensor_vec_opacity_c);
                optimizer_->param_groups()[15].options().set_lr(training_args.opacity_lr_coarse);

                optimizer_->add_param_group(Tensor_vec_scaling_c);
                optimizer_->param_groups()[16].options().set_lr(training_args.scaling_lr_coarse);

                optimizer_->add_param_group(Tensor_vec_rotation_c);
                optimizer_->param_groups()[17].options().set_lr(training_args.rotation_lr_coarse);

                optimizer_->add_param_group(mlp_opacity_c->parameters());
                optimizer_->param_groups()[18].options().set_lr(training_args.mlp_opacity_lr_init_coarse);

                optimizer_->add_param_group(mlp_cov_c->parameters());
                optimizer_->param_groups()[19].options().set_lr(training_args.mlp_cov_lr_init_coarse);

                optimizer_->add_param_group(mlp_color_c->parameters());
                optimizer_->param_groups()[20].options().set_lr(training_args.mlp_color_lr_init_coarse);

                optimizer_->add_param_group(mlp_apperance_c->parameters());
                optimizer_->param_groups()[21].options().set_lr(training_args.appearance_lr_init_coarse);

                optimizer_->add_param_group(mlp_feature_bank_c->parameters());
                optimizer_->param_groups()[22].options().set_lr(training_args.mlp_featurebank_lr_init_coarse);
            }
            else
                throw std::runtime_error("if use feature bank ,appearance_dim shoud > 0!");
        }
        else if (appearance_dim > 0)
        {
            optimizer_->add_param_group(Tensor_vec_anchor_c);
            optimizer_->param_groups()[11].options().set_lr(training_args.anchor_lr_init_coarse * this->spatial_lr_scale_);

            optimizer_->add_param_group(Tensor_vec_offset_c);
            optimizer_->param_groups()[12].options().set_lr(training_args.offset_lr_init_coarse* this->spatial_lr_scale_);

            optimizer_->add_param_group(Tensor_vec_anchor_feat_c);
            optimizer_->param_groups()[13].options().set_lr(training_args.feature_lr_coarse );

            optimizer_->add_param_group(Tensor_vec_opacity_c);
            optimizer_->param_groups()[14].options().set_lr(training_args.opacity_lr_coarse);

            optimizer_->add_param_group(Tensor_vec_scaling_c);
            optimizer_->param_groups()[15].options().set_lr(training_args.scaling_lr_coarse);

            optimizer_->add_param_group(Tensor_vec_rotation_c);
            optimizer_->param_groups()[16].options().set_lr(training_args.rotation_lr_coarse);

            optimizer_->add_param_group(mlp_opacity_c->parameters());
            optimizer_->param_groups()[17].options().set_lr(training_args.mlp_opacity_lr_init_coarse);

            optimizer_->add_param_group(mlp_cov_c->parameters());
            optimizer_->param_groups()[18].options().set_lr(training_args.mlp_cov_lr_init_coarse);

            optimizer_->add_param_group(mlp_color_c->parameters());
            optimizer_->param_groups()[19].options().set_lr(training_args.mlp_color_lr_init_coarse);

            optimizer_->add_param_group(mlp_apperance_c->parameters());
            optimizer_->param_groups()[20].options().set_lr(training_args.appearance_lr_init_coarse);
        }
        else
        {
            optimizer_->add_param_group(Tensor_vec_anchor_c);
            optimizer_->param_groups()[9].options().set_lr(training_args.anchor_lr_init_coarse * this->spatial_lr_scale_);

            optimizer_->add_param_group(Tensor_vec_offset_c);
            optimizer_->param_groups()[10].options().set_lr(training_args.offset_lr_init_coarse* this->spatial_lr_scale_);

            optimizer_->add_param_group(Tensor_vec_anchor_feat_c);
            optimizer_->param_groups()[11].options().set_lr(training_args.feature_lr_coarse );

            optimizer_->add_param_group(Tensor_vec_opacity_c);
            optimizer_->param_groups()[12].options().set_lr(training_args.opacity_lr_coarse);

            optimizer_->add_param_group(Tensor_vec_scaling_c);
            optimizer_->param_groups()[13].options().set_lr(training_args.scaling_lr_coarse);

            optimizer_->add_param_group(Tensor_vec_rotation_c);
            optimizer_->param_groups()[14].options().set_lr(training_args.rotation_lr_coarse);

            optimizer_->add_param_group(mlp_opacity_c->parameters());
            optimizer_->param_groups()[15].options().set_lr(training_args.mlp_opacity_lr_init_coarse);

            optimizer_->add_param_group(mlp_cov_c->parameters());
            optimizer_->param_groups()[16].options().set_lr(training_args.mlp_cov_lr_init_coarse);

            optimizer_->add_param_group(mlp_color_c->parameters());
            optimizer_->param_groups()[17].options().set_lr(training_args.mlp_color_lr_init_coarse);
        }
    }

    lr_init_ = training_args.position_lr_init_ * this->spatial_lr_scale_;
    lr_final_ = training_args.position_lr_final_ * this->spatial_lr_scale_;
    lr_delay_mult_ = training_args.position_lr_delay_mult_;
    max_steps_ = training_args.position_lr_max_steps_;

    offset_lr_init = training_args.offset_lr_init * spatial_lr_scale_;
    offset_lr_final = training_args.offset_lr_final * spatial_lr_scale_;
    offset_lr_delay_mult = training_args.offset_lr_delay_mult;
    offset_lr_max_steps = training_args.offset_lr_max_steps;

    mlp_opacity_lr_init = training_args.mlp_opacity_lr_init;
    mlp_opacity_lr_final = training_args.mlp_opacity_lr_final;
    mlp_opacity_lr_delay_mult = training_args.mlp_opacity_lr_delay_mult;
    mlp_opacity_lr_max_steps = training_args.mlp_opacity_lr_max_steps;

    mlp_cov_lr_init = training_args.mlp_cov_lr_init;
    mlp_cov_lr_final = training_args.mlp_cov_lr_final;
    mlp_cov_lr_delay_mult = training_args.mlp_cov_lr_delay_mult;
    mlp_cov_lr_max_steps = training_args.mlp_cov_lr_max_steps;

    mlp_color_lr_init = training_args.mlp_color_lr_init;
    mlp_color_lr_final = training_args.mlp_color_lr_final;
    mlp_color_lr_delay_mult = training_args.mlp_color_lr_delay_mult;
    mlp_color_lr_max_steps = training_args.mlp_color_lr_max_steps;

    if (use_feat_bank) {
        mlp_featurebank_lr_init = training_args.mlp_featurebank_lr_init;
        mlp_featurebank_lr_final = training_args.mlp_featurebank_lr_final;
        mlp_featurebank_lr_delay_mult = training_args.mlp_featurebank_lr_delay_mult;
        mlp_featurebank_lr_max_steps = training_args.mlp_featurebank_lr_max_steps;
    }

    if (appearance_dim > 0) {
        appearance_lr_init = training_args.appearance_lr_init;
        appearance_lr_final = training_args.appearance_lr_final;
        appearance_lr_delay_mult = training_args.appearance_lr_delay_mult;
        appearance_lr_max_steps = training_args.appearance_lr_max_steps;
    }

    if( use_coarse_anchor)
    {
        anchor_lr_init_coarse = training_args.anchor_lr_init_coarse * this->spatial_lr_scale_;
        anchor_lr_final_coarse = training_args.anchor_lr_final_coarse * this->spatial_lr_scale_;
        anchor_lr_delay_mult_coarse = training_args.anchor_lr_delay_mult_coarse;
        anchor_lr_max_steps_coarse = training_args.anchor_lr_max_steps_coarse;

        offset_lr_init_coarse = training_args.offset_lr_init_coarse * spatial_lr_scale_;
        offset_lr_final_coarse = training_args.offset_lr_final_coarse * spatial_lr_scale_;
        offset_lr_delay_mult_coarse = training_args.offset_lr_delay_mult_coarse;
        offset_lr_max_steps_coarse = training_args.offset_lr_max_steps_coarse;

        mlp_opacity_lr_init_coarse = training_args.mlp_opacity_lr_init_coarse;
        mlp_opacity_lr_final_coarse = training_args.mlp_opacity_lr_final_coarse;
        mlp_opacity_lr_delay_mult_coarse = training_args.mlp_opacity_lr_delay_mult_coarse;
        mlp_opacity_lr_max_steps_coarse = training_args.mlp_opacity_lr_max_steps_coarse;

        mlp_cov_lr_init_coarse = training_args.mlp_cov_lr_init_coarse;
        mlp_cov_lr_final_coarse = training_args.mlp_cov_lr_final_coarse;
        mlp_cov_lr_delay_mult_coarse = training_args.mlp_cov_lr_delay_mult_coarse;
        mlp_cov_lr_max_steps_coarse = training_args.mlp_cov_lr_max_steps_coarse;

        mlp_color_lr_init_coarse = training_args.mlp_color_lr_init_coarse;
        mlp_color_lr_final_coarse = training_args.mlp_color_lr_final_coarse;
        mlp_color_lr_delay_mult_coarse = training_args.mlp_color_lr_delay_mult_coarse;
        mlp_color_lr_max_steps_coarse = training_args.mlp_color_lr_max_steps_coarse;

        if (use_feat_bank) {
            mlp_featurebank_lr_init_coarse = training_args.mlp_featurebank_lr_init_coarse;
            mlp_featurebank_lr_final_coarse = training_args.mlp_featurebank_lr_final_coarse;
            mlp_featurebank_lr_delay_mult_coarse = training_args.mlp_featurebank_lr_delay_mult_coarse;
            mlp_featurebank_lr_max_steps_coarse = training_args.mlp_featurebank_lr_max_steps_coarse;
        }

        if (appearance_dim > 0) {
            appearance_lr_init_coarse = training_args.appearance_lr_init_coarse;
            appearance_lr_final_coarse = training_args.appearance_lr_final_coarse;
            appearance_lr_delay_mult_coarse = training_args.appearance_lr_delay_mult_coarse;
            appearance_lr_max_steps_coarse = training_args.appearance_lr_max_steps_coarse;
        }
    }
    std::cout << "GaussianModel::trainingSetup finished!" << std::endl;
}

float GaussianModel::updateLearningRate(int step)
{
    double anchor_lr = this->getExponLrFunc(step,
                                           this->lr_init_, this->lr_final_,
                                           this->lr_delay_mult_, this->max_steps_);
    optimizer_->param_groups()[0].options().set_lr(anchor_lr); 

    double offset_lr = this->getExponLrFunc(step,
                                           this->offset_lr_init, this->offset_lr_final,
                                           this->offset_lr_delay_mult, this->offset_lr_max_steps);
    optimizer_->param_groups()[1].options().set_lr(offset_lr); 

    double mlp_opacity_lr = this->getExponLrFunc(step,
                                           this->mlp_opacity_lr_init, this->mlp_opacity_lr_final,
                                           this->mlp_opacity_lr_delay_mult, this->mlp_opacity_lr_max_steps);
    optimizer_->param_groups()[6].options().set_lr(mlp_opacity_lr); 

    double mlp_cov_lr = this->getExponLrFunc(step,
                                           this->mlp_cov_lr_init, this->mlp_cov_lr_final,
                                           this->mlp_cov_lr_delay_mult, this->mlp_cov_lr_max_steps);
    optimizer_->param_groups()[7].options().set_lr(mlp_cov_lr); 

    double mlp_color_lr = this->getExponLrFunc(step,
                                           this->mlp_color_lr_init, this->mlp_color_lr_final,
                                           this->mlp_color_lr_delay_mult, this->mlp_color_lr_max_steps);
    optimizer_->param_groups()[8].options().set_lr(mlp_color_lr); 

    if (use_feat_bank) {
        double mlp_featurebank_lr = this->getExponLrFunc(step,
                                            this->mlp_featurebank_lr_init, this->mlp_featurebank_lr_final,
                                            this->mlp_featurebank_lr_delay_mult, this->mlp_featurebank_lr_max_steps);
        optimizer_->param_groups()[11].options().set_lr(mlp_featurebank_lr); 
    }

    if (appearance_dim > 0) {
        double appearance_lr = this->getExponLrFunc(step,
                                            this->appearance_lr_init, this->appearance_lr_final,
                                            this->appearance_lr_delay_mult, this->appearance_lr_max_steps);
        optimizer_->param_groups()[9].options().set_lr(appearance_lr); 

        optimizer_->param_groups()[10].options().set_lr(appearance_lr); 
    }

    if( use_coarse_anchor)
    {

        double anchor_lr_coarse = this->getExponLrFunc(step,
                                            this->anchor_lr_init_coarse, this->anchor_lr_final_coarse,
                                            this->anchor_lr_delay_mult_coarse, this->anchor_lr_max_steps_coarse);

        double offset_lr_coarse = this->getExponLrFunc(step,
                                            this->offset_lr_init_coarse, this->offset_lr_final_coarse,
                                            this->offset_lr_delay_mult_coarse, this->offset_lr_max_steps_coarse);

        double mlp_opacity_lr_coarse = this->getExponLrFunc(step,
                                            this->mlp_opacity_lr_init_coarse, this->mlp_opacity_lr_final_coarse,
                                            this->mlp_opacity_lr_delay_mult_coarse, this->mlp_opacity_lr_max_steps_coarse);

        double mlp_cov_lr_coarse = this->getExponLrFunc(step,
                                            this->mlp_cov_lr_init_coarse, this->mlp_cov_lr_final_coarse,
                                            this->mlp_cov_lr_delay_mult_coarse, this->mlp_cov_lr_max_steps_coarse);

        double mlp_color_lr_coarse = this->getExponLrFunc(step,
                                            this->mlp_color_lr_init_coarse, this->mlp_color_lr_final_coarse,
                                            this->mlp_color_lr_delay_mult_coarse, this->mlp_color_lr_max_steps_coarse);

        double mlp_featurebank_lr_coarse, appearance_lr_coarse;
        if (use_feat_bank)
        {
            mlp_featurebank_lr_coarse = this->getExponLrFunc(step,
                                            this->mlp_featurebank_lr_init_coarse, this->mlp_featurebank_lr_final_coarse,
                                            this->mlp_featurebank_lr_delay_mult_coarse, this->mlp_featurebank_lr_max_steps_coarse);
        }
        if (appearance_dim > 0) { 
            appearance_lr_coarse = this->getExponLrFunc(step,
                                            this->appearance_lr_init_coarse, this->appearance_lr_final_coarse,
                                            this->appearance_lr_delay_mult_coarse, this->appearance_lr_max_steps_coarse);
        }
        if(use_feat_bank)
        {
            if(appearance_dim > 0 )
            {
                optimizer_->param_groups()[12].options().set_lr(anchor_lr_coarse);
                optimizer_->param_groups()[13].options().set_lr(offset_lr_coarse);
                optimizer_->param_groups()[18].options().set_lr(mlp_opacity_lr_coarse);
                optimizer_->param_groups()[19].options().set_lr(mlp_cov_lr_coarse);
                optimizer_->param_groups()[20].options().set_lr(mlp_color_lr_coarse);
                optimizer_->param_groups()[21].options().set_lr(mlp_featurebank_lr_coarse);
                optimizer_->param_groups()[22].options().set_lr(appearance_lr_coarse);
            }
            else
                throw std::runtime_error("if use feature bank ,appearance_dim shoud > 0!");
        }
        else if (appearance_dim > 0)
        {
            optimizer_->param_groups()[11].options().set_lr(anchor_lr_coarse);
            optimizer_->param_groups()[12].options().set_lr(offset_lr_coarse);
            optimizer_->param_groups()[17].options().set_lr(mlp_opacity_lr_coarse);
            optimizer_->param_groups()[18].options().set_lr(mlp_cov_lr_coarse);
            optimizer_->param_groups()[19].options().set_lr(mlp_color_lr_coarse);
            optimizer_->param_groups()[20].options().set_lr(appearance_lr_coarse);
        }
        else
        {
            optimizer_->param_groups()[9].options().set_lr(anchor_lr_coarse);
            optimizer_->param_groups()[10].options().set_lr(offset_lr_coarse);
            optimizer_->param_groups()[15].options().set_lr(mlp_opacity_lr_coarse);
            optimizer_->param_groups()[16].options().set_lr(mlp_cov_lr_coarse);
            optimizer_->param_groups()[17].options().set_lr(mlp_color_lr_coarse);
        }
    }

    if (step % 3000 == 0)
    {
        std::cout << "[updateLearningRate]: "
                  << "anchor_lr = " << anchor_lr
                  << ", offset_lr = " << offset_lr
                  << ", mlp_opacity_lr = " << mlp_opacity_lr
                  << ", mlp_cov_lr = " << mlp_cov_lr
                  << ", mlp_color_lr = " << mlp_color_lr
                  << std::endl;
    }

    return anchor_lr;
}

void GaussianModel::setPositionLearningRate(float position_lr)
{
    optimizer_->param_groups()[0].options().set_lr(position_lr * this->spatial_lr_scale_);
}

void GaussianModel::setAnchorFeatureLearningRate(float feature_lr)
{
    optimizer_->param_groups()[2].options().set_lr(feature_lr);
}

void GaussianModel::setOpacityLearningRate(float opacity_lr)
{
    optimizer_->param_groups()[3].options().set_lr(opacity_lr);
}
void GaussianModel::setScalingLearningRate(float scaling_lr)
{
    optimizer_->param_groups()[4].options().set_lr(scaling_lr);
}
void GaussianModel::setRotationLearningRate(float rot_lr)
{
    optimizer_->param_groups()[5].options().set_lr(rot_lr);
}

void GaussianModel::resetOpacity()
{
    torch::Tensor opacities_new = general_utils::inverse_sigmoid(
        torch::min(
            this->get_opacity(),
            torch::ones_like(this->get_opacity() * 0.01)));
    torch::Tensor optimizable_tensors = this->replaceTensorToOptimizer(opacities_new, 3);
    this->_opacity = optimizable_tensors;
    this->Tensor_vec_opacity = {this->_opacity};
}

torch::Tensor GaussianModel::replaceTensorToOptimizer(torch::Tensor& tensor, int tensor_idx)
{
    auto& param = this->optimizer_->param_groups()[tensor_idx].params()[0];
    auto& state = optimizer_->state();
    auto key = c10::guts::to_string(param.unsafeGetTensorImpl());
    auto& stored_state = static_cast<torch::optim::AdamParamState&>(*state[key]);
    auto new_state = std::make_unique<torch::optim::AdamParamState>();
    new_state->step(stored_state.step());
    new_state->exp_avg(torch::zeros_like(tensor));
    new_state->exp_avg_sq(torch::zeros_like(tensor));

    state.erase(key);
    param = tensor.requires_grad_();
    key = c10::guts::to_string(param.unsafeGetTensorImpl());
    state[key] = std::move(new_state);

    auto optimizable_tensors = param;
    return optimizable_tensors;
}

void GaussianModel::loadPly(std::filesystem::path ply_path)
{
    std::ifstream instream_binary(ply_path, std::ios::binary);
    if (!instream_binary.is_open() || instream_binary.fail())
        throw std::runtime_error("Fail to open ply file at " + ply_path.string());
    instream_binary.seekg(0, std::ios::beg);

    tinyply::PlyFile ply_file;
    ply_file.parse_header(instream_binary);

    std::cout << "\t[ply_header] Type: " << (ply_file.is_binary_file() ? "binary" : "ascii") << std::endl;
    for (const auto & c : ply_file.get_comments())
        std::cout << "\t[ply_header] Comment: " << c << std::endl;
    for (const auto & c : ply_file.get_info())
        std::cout << "\t[ply_header] Info: " << c << std::endl;

    for (const auto &e : ply_file.get_elements()) {
        std::cout << "\t[ply_header] element: " << e.name << " (" << e.size << ")" << std::endl;
        for (const auto &p : e.properties) {
            std::cout << "\t[ply_header] \tproperty: " << p.name << " (type=" << tinyply::PropertyTable[p.propertyType].str << ")";
            if (p.isList)
                std::cout << " (list_type=" << tinyply::PropertyTable[p.listType].str << ")";
            std::cout << std::endl;
        }
    }

    std::shared_ptr<tinyply::PlyData> anchor, offsets, anchor_feats, opacity, scales, rot;

    try { anchor = ply_file.request_properties_from_element("vertex", { "x", "y", "z" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    int n_f_offset = this->n_offsets * 3;
    if (n_f_offset >= 0) {
        std::vector<std::string> f_offset_element_names(n_f_offset);
        for (int i = 0; i < n_f_offset; ++i)
            f_offset_element_names[i] = "f_offset_" + std::to_string(i);
        try {offsets = ply_file.request_properties_from_element("vertex", f_offset_element_names); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }
    }

    int n_f_anchor_feat = this->feat_dim;
    if (n_f_anchor_feat >= 0) {
        std::vector<std::string> f_anchor_feat_element_names(n_f_anchor_feat);
        for (int i = 0; i < n_f_anchor_feat; ++i)
            f_anchor_feat_element_names[i] = "f_anchor_feat_" + std::to_string(i);
        try {anchor_feats = ply_file.request_properties_from_element("vertex", f_anchor_feat_element_names); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }
    }

    try { opacity = ply_file.request_properties_from_element("vertex", { "opacity" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    try { scales = ply_file.request_properties_from_element(
        "vertex", { "scale_0", "scale_1", "scale_2", "scale_3", "scale_4", "scale_5" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    try { rot = ply_file.request_properties_from_element("vertex", { "rot_0", "rot_1", "rot_2", "rot_3" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    ply_file.read(instream_binary);

    if (anchor)     std::cout << "\tRead " << anchor->count     << " total anchor "     << std::endl;
    if (offsets)    std::cout << "\tRead " << offsets->count    << " total offsets "    << std::endl;
    if (anchor_feats)  std::cout << "\tRead " << anchor_feats->count  << " total anchor_feats "  << std::endl;
    if (opacity) std::cout << "\tRead " << opacity->count << " total opacity " << std::endl;
    if (scales)  std::cout << "\tRead " << scales->count  << " total scales "  << std::endl;
    if (rot)     std::cout << "\tRead " << rot->count     << " total rot "     << std::endl;

    const int num_points = anchor->count;

    const std::size_t n_xyz_bytes = anchor->buffer.size_bytes();
    std::vector<float> anchor_vector(anchor->count * 3);
    std::memcpy(anchor_vector.data(), anchor->buffer.get(), n_xyz_bytes);

    const std::size_t n_f_offsets_bytes = offsets->buffer.size_bytes();
    std::vector<float> f_offsets_vector(offsets->count * n_f_offset);
    std::memcpy(f_offsets_vector.data(), offsets->buffer.get(), n_f_offsets_bytes);
 
    const std::size_t n_f_anchor_feats_bytes = anchor_feats->buffer.size_bytes();
    std::vector<float> f_anchor_feats_vector(anchor_feats->count * n_f_anchor_feat);
    std::memcpy(f_anchor_feats_vector.data(), anchor_feats->buffer.get(), n_f_anchor_feats_bytes);

    const std::size_t n_opacity_bytes = opacity->buffer.size_bytes();
    std::vector<float> opacity_vector(opacity->count * 1);
    std::memcpy(opacity_vector.data(), opacity->buffer.get(), n_opacity_bytes);

    const std::size_t n_scales_bytes = scales->buffer.size_bytes();
    std::vector<float> scales_vector(scales->count * 6);
    std::memcpy(scales_vector.data(), scales->buffer.get(), n_scales_bytes);

    const std::size_t n_rot_bytes = rot->buffer.size_bytes();
    std::vector<float> rot_vector(rot->count * 4);
    std::memcpy(rot_vector.data(), rot->buffer.get(), n_rot_bytes);

    this->_anchor = torch::from_blob(
        anchor_vector.data(), {num_points, 3},
        torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_).requires_grad_();

    auto off = torch::from_blob(
                        f_offsets_vector.data(), {num_points*this->n_offsets*3},
                        torch::TensorOptions().dtype(torch::kFloat32))
                        .to(device_type_);
    auto off_1 = off.reshape({num_points, this->n_offsets * 3});
    auto off_2  = off_1.reshape({num_points, 3, -1});
    this->_offset = off_2.transpose(1, 2).contiguous().requires_grad_();

    this->_anchor_feat = torch::from_blob(
        f_anchor_feats_vector.data(), {num_points, this->feat_dim},
        torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_).requires_grad_();

    this->_opacity = torch::from_blob(
        opacity_vector.data(), {num_points, 1},
        torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_).requires_grad_();

    this->_scaling = torch::from_blob(
        scales_vector.data(), {num_points, 6},
        torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_).requires_grad_();

    this->_rotation = torch::from_blob(
        rot_vector.data(), {num_points, 4},
        torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_).requires_grad_();

    GAUSSIAN_MODEL_TENSORS_TO_VEC
}

void GaussianModel::savePly(std::filesystem::path result_path)
{
    torch::Tensor anchor = this->_anchor.detach().cpu();
    torch::Tensor normals = torch::zeros_like(anchor);
    torch::Tensor anchor_feat = this->_anchor_feat.cpu();
    torch::Tensor offset = this->_offset.detach().transpose(1, 2).flatten(1).contiguous().cpu();
    torch::Tensor opacities = this->_opacity.detach().cpu();
    torch::Tensor scale = this->_scaling.detach().cpu();
    torch::Tensor rotation = this->_rotation.detach().cpu();

    std::filebuf fb_binary;
    fb_binary.open(result_path, std::ios::out | std::ios::binary);
    std::ostream outstream_binary(&fb_binary);
    if (outstream_binary.fail()) throw std::runtime_error("failed to open " + result_path.string());

    tinyply::PlyFile result_file;

    result_file.add_properties_to_element(
        "vertex", {"x", "y", "z"},
        tinyply::Type::FLOAT32, anchor.size(0),
        reinterpret_cast<uint8_t*>(anchor.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    result_file.add_properties_to_element(
        "vertex", {"nx", "ny", "nz"},
        tinyply::Type::FLOAT32, normals.size(0),
        reinterpret_cast<uint8_t*>(normals.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    std::size_t n_anchor_feat = this->_anchor_feat.size(1);
    std::vector<std::string> property_names_anchor_feat(n_anchor_feat);
    for (int i = 0; i < n_anchor_feat; ++i)
        property_names_anchor_feat[i] = "anchor_feat_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_anchor_feat,
        tinyply::Type::FLOAT32, this->_anchor_feat.size(0),
        reinterpret_cast<uint8_t*>(anchor_feat.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    std::size_t n_offset = this->_offset.size(1) * this->_offset.size(2);
    std::vector<std::string> property_names_offset(n_offset);
    for (int i = 0; i < n_offset; ++i)
        property_names_offset[i] = "offset_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_offset,
        tinyply::Type::FLOAT32, this->_offset.size(0),
        reinterpret_cast<uint8_t*>(offset.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    result_file.add_properties_to_element(
        "vertex", {"opacity"},
        tinyply::Type::FLOAT32, opacities.size(0),
        reinterpret_cast<uint8_t*>(opacities.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    std::size_t n_scale = scale.size(1);
    std::vector<std::string> property_names_scale(n_scale);
    for (int i = 0; i < n_scale; ++i)
        property_names_scale[i] = "scale_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_scale,
        tinyply::Type::FLOAT32, scale.size(0),
        reinterpret_cast<uint8_t*>(scale.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    std::size_t n_rotation = rotation.size(1);
    std::vector<std::string> property_names_rotation(n_rotation);
    for (int i = 0; i < n_rotation; ++i)
        property_names_rotation[i] = "rot_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_rotation,
        tinyply::Type::FLOAT32, rotation.size(0),
        reinterpret_cast<uint8_t*>(rotation.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    result_file.write(outstream_binary, true);

    fb_binary.close();
}
void GaussianModel::save_mlp_checkpoints(std::filesystem::path result_path)
{
    this->eval();
    auto opacity_linear1 = dynamic_cast<torch::nn::LinearImpl *>(this->mlp_opacity[0].get());
    auto opacity_linear2 = dynamic_cast<torch::nn::LinearImpl *>(this->mlp_opacity[2].get());
    torch::Tensor opacity_weight1 = opacity_linear1->weight.detach().cpu();
    torch::Tensor opacity_bias1 = opacity_linear1->bias.detach().cpu();
    torch::Tensor opacity_weight2 = opacity_linear2->weight.detach().cpu();
    torch::Tensor opacity_bias2 = opacity_linear2->bias.detach().cpu();
    saveTensorToTxt(opacity_weight1, result_path / "opacity_weight1.txt");
    saveTensorToTxt(opacity_bias1, result_path / "opacity_bias1.txt");
    saveTensorToTxt(opacity_weight2, result_path / "opacity_weight2.txt");
    saveTensorToTxt(opacity_bias2, result_path / "opacity_bias2.txt");
    

    auto cov_linear1 = dynamic_cast<torch::nn::LinearImpl *>(this->mlp_cov[0].get());
    auto cov_linear2 = dynamic_cast<torch::nn::LinearImpl *>(this->mlp_cov[2].get());
    torch::Tensor cov_weight1 = cov_linear1->weight.detach().cpu();
    torch::Tensor cov_bias1 = cov_linear1->bias.detach().cpu();
    torch::Tensor cov_weight2 = cov_linear2->weight.detach().cpu();
    torch::Tensor cov_bias2 = cov_linear2->bias.detach().cpu();
    saveTensorToTxt(cov_weight1, result_path / "cov_weight1.txt");
    saveTensorToTxt(cov_bias1, result_path / "cov_bias1.txt");
    saveTensorToTxt(cov_weight2, result_path / "cov_weight2.txt");
    saveTensorToTxt(cov_bias2, result_path / "cov_bias2.txt");

    auto color_linear1 = dynamic_cast<torch::nn::LinearImpl *>(this->mlp_color[0].get());
    auto color_linear2 = dynamic_cast<torch::nn::LinearImpl *>(this->mlp_color[2].get());
    torch::Tensor color_weight1 = color_linear1->weight.detach().cpu();
    torch::Tensor color_bias1 = color_linear1->bias.detach().cpu();
    torch::Tensor color_weight2 = color_linear2->weight.detach().cpu();
    torch::Tensor color_bias2 = color_linear2->bias.detach().cpu();
    saveTensorToTxt(color_weight1, result_path / "color_weight1.txt");
    saveTensorToTxt(color_bias1, result_path / "color_bias1.txt");
    saveTensorToTxt(color_weight2, result_path / "color_weight2.txt");
    saveTensorToTxt(color_bias2, result_path / "color_bias2.txt");

    if (use_feat_bank ) {
        auto feat_linear1 = dynamic_cast<torch::nn::LinearImpl *>(this->mlp_feature_bank[0].get());
        auto feat_linear2 = dynamic_cast<torch::nn::LinearImpl *>(this->mlp_feature_bank[2].get());
        torch::Tensor feat_weight1 = feat_linear1->weight.detach().cpu();
        torch::Tensor feat_bias1 = feat_linear1->bias.detach().cpu();
        torch::Tensor feat_weight2 = feat_linear2->weight.detach().cpu();
        torch::Tensor feat_bias2 = feat_linear2->bias.detach().cpu();
        saveTensorToTxt(feat_weight1, result_path / "feat_weight1.txt");
        saveTensorToTxt(feat_bias1, result_path / "feat_bias1.txt");
        saveTensorToTxt(feat_weight2, result_path / "feat_weight2.txt");
        saveTensorToTxt(feat_bias2, result_path / "feat_bias2.txt");
    }

    if (appearance_dim > 0 ) {
        torch::Tensor embedding_weight = this->embedding_appearance->get_embedding()->weight.detach().cpu();
        saveTensorToTxt(embedding_weight, result_path / "embedding_weight.txt");
    }

}

void GaussianModel::saveSparsePointsPly(std::filesystem::path result_path)
{
    torch::Tensor xyz = this->sparse_points_xyz_.detach().cpu();
    torch::Tensor normals = torch::zeros_like(xyz);
    torch::Tensor color = (this->sparse_points_color_ * 255.0f).toType(torch::kUInt8).detach().cpu();

    std::filebuf fb_binary;
    fb_binary.open(result_path, std::ios::out | std::ios::binary);
    std::ostream outstream_binary(&fb_binary);
    if (outstream_binary.fail()) throw std::runtime_error("failed to open " + result_path.string());

    tinyply::PlyFile result_file;

    result_file.add_properties_to_element(
        "vertex", {"x", "y", "z"},
        tinyply::Type::FLOAT32, xyz.size(0),
        reinterpret_cast<uint8_t*>(xyz.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    result_file.add_properties_to_element(
        "vertex", {"nx", "ny", "nz"},
        tinyply::Type::FLOAT32, normals.size(0),
        reinterpret_cast<uint8_t*>(normals.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    result_file.add_properties_to_element(
        "vertex", {"red", "green", "blue"},
        tinyply::Type::UINT8, color.size(0),
        reinterpret_cast<uint8_t*>(color.data_ptr<uint8_t>()),
        tinyply::Type::INVALID, 0);

    result_file.write(outstream_binary, true);

    fb_binary.close();
}

float GaussianModel::percentDense()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return percent_dense_;
}

void GaussianModel::setPercentDense(const float percent_dense)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    percent_dense_ = percent_dense;
}


float GaussianModel::exponLrFunc(int step)
{
    if (step < 0 || (lr_init_ == 0.0f && lr_final_ == 0.0f))
        return 0.0f;

    float delay_rate;
    if (lr_delay_steps_ > 0)
        delay_rate = lr_delay_mult_ + (1.0f - lr_delay_mult_) * std::sin(M_PI_2f32 * std::clamp(static_cast<float>(step) / lr_delay_steps_, 0.0f, 1.0f));
    else
        delay_rate = 1.0f;
    float t = std::clamp(static_cast<float>(step) / max_steps_, 0.0f, 1.0f);
    float log_lerp = std::exp(std::log(lr_init_) * (1 - t) + std::log(lr_final_) * t);
    return delay_rate * log_lerp;
}

void GaussianModel::setApperance()
{
    std::cout << "[GaussianModel]setApperance:" << this->embedding_dim << "->" << appearance_dim << " " << this->device_type_ << std::endl;
    if(appearance_dim > 0)
    {
        this->embedding_appearance = std::make_shared<Embedding>(this->embedding_dim, appearance_dim);
        this->embedding_appearance->to(this->device_type_);
    }
}

float GaussianModel::getExponLrFunc(int step,
                        float lr_init, float lr_final,
                        float lr_delay_mult, int max_steps)
{
    if (step < 0 || (lr_init == 0.0f && lr_final == 0.0f))
    return 0.0f;

    float delay_rate;
    if (lr_delay_steps_ > 0)
        delay_rate = lr_delay_mult +
                     (1.0f - lr_delay_mult) * std::sin(M_PI_2f32 * std::clamp(static_cast<float>(step) / lr_delay_steps_, 0.0f, 1.0f));
    else
        delay_rate = 1.0f;
    float t = std::clamp(static_cast<float>(step) / max_steps, 0.0f, 1.0f);
    float log_lerp = std::exp(std::log(lr_init) * (1 - t) + std::log(lr_final) * t);
    return delay_rate * log_lerp;
}

void GaussianModel::eval()
{
    this->mlp_opacity->eval();
    this->mlp_cov->eval();
    this->mlp_color->eval();
    if(this->appearance_dim > 0)
        this->embedding_appearance->eval();
    if(this->use_feat_bank)
        this->mlp_feature_bank->eval();
    if(this->appearance_dim > 0)
        this->mlp_apperance->eval();

    if(use_coarse_anchor)
    {
        this->mlp_opacity_c->eval();
        this->mlp_cov_c->eval();
        this->mlp_color_c->eval();
        if(this->use_feat_bank)
            this->mlp_feature_bank_c->eval();
        if(this->appearance_dim > 0)
            this->mlp_apperance_c->eval();        
    }
}

void GaussianModel::train()
{
    this->mlp_opacity->train();
    this->mlp_cov->train();
    this->mlp_color->train();
    if(this->appearance_dim > 0)
        this->embedding_appearance->train();
    if(this->use_feat_bank)
        this->mlp_feature_bank->train();
    if(this->appearance_dim > 0)
        this->mlp_apperance->train();

    if(use_coarse_anchor)
    {
        this->mlp_opacity_c->train();
        this->mlp_cov_c->train();
        this->mlp_color_c->train();
        if(this->use_feat_bank)
            this->mlp_feature_bank_c->train();
        if(this->appearance_dim > 0)
            this->mlp_apperance_c->train();        
    }
}

void GaussianModel::training_statis(
    torch::Tensor &viewspace_point_tensor,
    torch::Tensor &opacity,
    torch::Tensor &update_filter,
    torch::Tensor &offset_selection_mask,
    torch::Tensor &anchor_visible_mask)
{
    auto temp_opacity = opacity.clone().view(-1).detach();
    temp_opacity = torch::where(temp_opacity < 0, torch::zeros_like(temp_opacity), temp_opacity);

    temp_opacity = temp_opacity.view({-1, this->n_offsets});
    temp_opacity.sum(1, true);
    this->opacity_accum.index({anchor_visible_mask}) + temp_opacity.sum(1, true);
    this->opacity_accum.index_put_(
        {anchor_visible_mask},
        this->opacity_accum.index({anchor_visible_mask}) + temp_opacity.sum(1, true));

    this->anchor_demon.index_put_(
        {anchor_visible_mask},
        this->anchor_demon.index({anchor_visible_mask}) + 1);

    anchor_visible_mask = anchor_visible_mask.unsqueeze(1).repeat({1, this->n_offsets}).view(-1);
    auto combined_mask = torch::zeros_like(this->offset_gradient_accum, torch::TensorOptions().dtype(torch::kBool)).squeeze(1);
    combined_mask.index_put_({anchor_visible_mask}, offset_selection_mask);
    auto temp_mask = combined_mask.clone();
    combined_mask.index_put_({temp_mask}, update_filter);
    torch::Tensor grad_norm;
    if (use_coarse_anchor)
    {
        grad_norm = torch::frobenius_norm(viewspace_point_tensor.grad().index({torch::indexing::Slice(0, 
                                           update_filter.size(0))}).index({update_filter, torch::indexing::Slice(0, 2)}),
                                           -1,
                                           true);
    }
    else
        grad_norm = torch::frobenius_norm(viewspace_point_tensor.grad().index({update_filter, torch::indexing::Slice(0, 2)}),
                                            -1,
                                            true);
    this->offset_gradient_accum.index_put_(
        {combined_mask},
        this->offset_gradient_accum.index({combined_mask}) + grad_norm);
    this->offset_denom.index_put_(
        {combined_mask},
        this->offset_denom.index({combined_mask}) + 1);
}

void GaussianModel::prune_anchor(torch::Tensor &mask)
{
    auto valid_points_mask = ~mask;
    std::vector<torch::Tensor> optimizable_tensors(6);
    auto& param_groups = this->optimizer_->param_groups();
    auto& state = this->optimizer_->state();
    for (int group_idx = 0; group_idx < 6; ++group_idx) {
        auto& param = param_groups[group_idx].params()[0];
        auto key = c10::guts::to_string(param.unsafeGetTensorImpl());
        if (state.find(key) != state.end()) {
            auto& stored_state = static_cast<torch::optim::AdamParamState&>(*state[key]);
            auto new_state = std::make_unique<torch::optim::AdamParamState>();
            new_state->step(stored_state.step());
            new_state->exp_avg(stored_state.exp_avg().index({valid_points_mask}).clone());
            new_state->exp_avg_sq(stored_state.exp_avg_sq().index({valid_points_mask}).clone());

            state.erase(key);
            param = param.index({valid_points_mask}).requires_grad_();
            key = c10::guts::to_string(param.unsafeGetTensorImpl());
            state[key] = std::move(new_state);
            if(group_idx == 4)
            {
                auto scales = param;
                auto temp = scales.index({torch::indexing::Slice(), torch::indexing::Slice(3, torch::indexing::None)});
                temp = torch::clamp(temp, -std::numeric_limits<float>::max(), 0.05);
                scales.index_put_({torch::indexing::Slice(), torch::indexing::Slice(3, torch::indexing::None)}, temp);
                param = scales;
            }
            optimizable_tensors[group_idx] = param;
        }
        else {
            param = param.index({valid_points_mask}).requires_grad_();
            if(group_idx == 4)
            {
                auto scales = param;
                auto temp = scales.index({torch::indexing::Slice(), torch::indexing::Slice(3, torch::indexing::None)});
                temp = torch::clamp(temp, -std::numeric_limits<float>::max(), 0.05);
                scales.index_put_({torch::indexing::Slice(), torch::indexing::Slice(3, torch::indexing::None)}, temp);
                param = scales;
            }
            optimizable_tensors[group_idx] = param;
        }
    }

    this->_anchor = optimizable_tensors[0];
    this->_offset = optimizable_tensors[1];
    this->_anchor_feat = optimizable_tensors[2];
    this->_opacity = optimizable_tensors[3];
    this->_scaling = optimizable_tensors[4];
    this->_rotation = optimizable_tensors[5];

    GAUSSIAN_MODEL_TENSORS_TO_VEC

}
void GaussianModel::anchor_growing(torch::Tensor &grads, float threshold, torch::Tensor &offset_mask)
{
    auto init_length = this->get_anchor().size(0) * this->n_offsets;
    for (int i = 0; i < this->update_depth; i++)
    {
        float cur_threshold = threshold * (std::pow(std::floor(this->update_hierachy_factor / 2), i));
        auto candidate_mask = (grads >= cur_threshold);
        candidate_mask = torch::logical_and(candidate_mask, offset_mask);

        auto rand_mask = torch::rand_like(candidate_mask.to(torch::kFloat)) > (std::pow(0.5, i + 1));
        rand_mask = rand_mask.to(this->device_type_);
        candidate_mask = torch::logical_and(candidate_mask, rand_mask);

        auto length_inc = this->get_anchor().size(0) * this->n_offsets - init_length;
        if(length_inc == 0)
        {
            if(i > 0)
                continue;
        }
        else
            candidate_mask = torch::cat({candidate_mask,
                                         torch::zeros({length_inc}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA))},
                                         0);

        auto all_xyz = this->get_anchor().unsqueeze(1) + this->_offset * this->get_scaling().index(
            {torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 3)}).unsqueeze(1);

        int size_factor = std::floor(this->update_init_factor / std::pow(this->update_hierachy_factor, i));
        float cur_size = this->voxel_size * size_factor;

        auto grid_coords = torch::round(this->get_anchor() / cur_size).to(torch::kInt);

        auto selected_xyz = all_xyz.view({-1, 3}).index({candidate_mask});
        auto selected_grid_coords = torch::round(selected_xyz / cur_size).to(torch::kInt);

        auto results = at::unique_dim(selected_grid_coords, 0, true, true);
        auto selected_grid_coords_unique = std::get<0>(results);
        auto inverse_indices = std::get<1>(results);
        bool use_chunk = true;
        torch::Tensor remove_duplicates;
        if (use_chunk)
        {
            int64_t chunk_size = 4096;
            int64_t max_iters = std::floor(grid_coords.size(0) / chunk_size) + (grid_coords.size(0) % chunk_size != 0 ? 1 : 0);
            std::vector<torch::Tensor> remove_duplicates_list(max_iters);
            torch::Tensor cur_remove_duplicates_temp;
            for (int j = 0; j < max_iters; j++)
            {
                auto cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords.index(
                    {torch::indexing::Slice(j * chunk_size, (j + 1) * chunk_size), torch::indexing::Slice()})).all(-1).any(-1).view(-1);
                remove_duplicates_list.push_back(cur_remove_duplicates);
                if (j == 0)
                    cur_remove_duplicates_temp = cur_remove_duplicates;
                else
                    cur_remove_duplicates_temp = torch::logical_or(cur_remove_duplicates_temp, cur_remove_duplicates);
            }
            remove_duplicates = cur_remove_duplicates_temp;
        }
        else
            remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1);

        remove_duplicates = ~remove_duplicates;
        auto candidate_anchor = selected_grid_coords_unique.index({remove_duplicates}) * cur_size;

        if(candidate_anchor.size(0) > 0)
        {
            auto new_scaling = torch::ones_like(candidate_anchor).repeat({1, 2}).to(torch::kFloat).to(this->device_type_) * cur_size;
            new_scaling = torch::log(new_scaling);
            auto new_rotation = torch::zeros({candidate_anchor.size(0), 4}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
            new_rotation.index_put_({torch::indexing::Slice(), 0}, 1.0);

            auto new_opacities = general_utils::inverse_sigmoid(0.1f * torch::ones({candidate_anchor.size(0), 1},
                                 torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA)));
            auto new_feat = this->_anchor_feat.unsqueeze(1).repeat({1, this->n_offsets, 1})
                                    .view({-1, this->feat_dim}).index({candidate_mask});

            auto scatter_max_results = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand({-1, new_feat.size(1)}),
                                                   0, torch::nullopt, torch::nullopt);          
            new_feat = std::get<0>(scatter_max_results).index({remove_duplicates});

            auto new_offsets = torch::zeros_like(candidate_anchor).unsqueeze(1).repeat({1, this->n_offsets, 1}).to(torch::kFloat).to(torch::kCUDA);

            auto zeros_tensor = torch::zeros({new_opacities.size(0), 1},
                                          torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
            auto temp_anchor_demon = torch::cat({this->anchor_demon, zeros_tensor}, 0);
            this->anchor_demon = temp_anchor_demon;

            auto temp_opacity_accum= torch::cat({this->opacity_accum, torch::zeros({new_opacities.size(0), 1},
                                            torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA))},0);
            this->opacity_accum = temp_opacity_accum;

            c10::cuda::CUDACachingAllocator::emptyCache();
           
            std::vector<torch::Tensor> optimizable_tensors(6);
            std::vector<torch::Tensor> tensors_dict = {
                candidate_anchor,
                new_offsets,
                new_feat,
                new_opacities,
                new_scaling,
                new_rotation
            };
            auto& param_groups = this->optimizer_->param_groups();
            auto& state = this->optimizer_->state();
            for (int group_idx = 0; group_idx < 6; ++group_idx) {
                auto& group = param_groups[group_idx];
                assert(group.params().size() == 1);
                auto& extension_tensor = tensors_dict[group_idx];
                auto& param = group.params()[0];
                auto key = c10::guts::to_string(param.unsafeGetTensorImpl());
                if (state.find(key) != state.end()) {
                    auto& stored_state = static_cast<torch::optim::AdamParamState&>(*state[key]);
                    auto new_state = std::make_unique<torch::optim::AdamParamState>();
                    new_state->step(stored_state.step());
                    new_state->exp_avg(torch::cat({stored_state.exp_avg().clone(), torch::zeros_like(extension_tensor)}, /*dim=*/0));
                    new_state->exp_avg_sq(torch::cat({stored_state.exp_avg_sq().clone(), torch::zeros_like(extension_tensor)}, /*dim=*/0));

                    state.erase(key);
                    param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();
                    key = c10::guts::to_string(param.unsafeGetTensorImpl());
                    state[key] = std::move(new_state);

                    optimizable_tensors[group_idx] = param;
                }
                else {
                    param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();
                    optimizable_tensors[group_idx] = param;
                }
            }

            this->_anchor = optimizable_tensors[0];
            this->_offset = optimizable_tensors[1];
            this->_anchor_feat = optimizable_tensors[2];
            this->_opacity = optimizable_tensors[3];
            this->_scaling = optimizable_tensors[4];
            this->_rotation = optimizable_tensors[5];

            GAUSSIAN_MODEL_TENSORS_TO_VEC
        }
    }
}

void GaussianModel::adjust_anchor(
    int check_interval,
    float success_threshold,
    float grad_threshold,
    float min_opacity)
{
    auto grads = this->offset_gradient_accum / this->offset_denom;
    grads.index_put_({grads.isnan()}, 0.0f);
    auto grads_norm = torch::frobenius_norm(grads, /*dim=*/-1);
    auto offset_mask = (this->offset_denom > check_interval * success_threshold * 0.5).squeeze(/*dim=*/ 1);

    this->anchor_growing(grads_norm, grad_threshold, offset_mask);

    this->offset_denom.index_put_({offset_mask}, 0);
    auto padding_offset_demon = torch::zeros(
        {this->get_anchor().size(0) * this->n_offsets - this->offset_denom.size(0), 1},
        torch::TensorOptions().dtype(torch::kInt32).device(this->device_type_));
    this->offset_denom = torch::cat({this->offset_denom, padding_offset_demon}, /*dim=*/0);

    this->offset_gradient_accum.index_put_({offset_mask}, 0);
    auto padding_offset_gradient_accum = torch::zeros(
        {this->get_anchor().size(0) * this->n_offsets - this->offset_gradient_accum.size(0), 1},
        torch::TensorOptions().dtype(torch::kInt32).device(this->device_type_));
    this->offset_gradient_accum = torch::cat({this->offset_gradient_accum, padding_offset_gradient_accum}, /*dim=*/0);

    auto prune_mask = (this->opacity_accum < min_opacity * this->anchor_demon).squeeze(/*dim=*/1);
    auto anchors_mask = (this->anchor_demon > check_interval * success_threshold).squeeze(/*dim=*/1);
    prune_mask = torch::logical_and(prune_mask, anchors_mask);

    auto offset_denom_new = this->offset_denom.view({-1, this->n_offsets}).index({~prune_mask});
    offset_denom_new = offset_denom_new.view({-1, 1});
    this->offset_denom = offset_denom_new;

    auto offset_gradient_accum_new = this->offset_gradient_accum.view({-1, this->n_offsets}).index({~prune_mask});
    offset_gradient_accum_new = offset_gradient_accum_new.view({-1, 1});
    this->offset_gradient_accum = offset_gradient_accum_new;

    if (anchors_mask.sum().item<int>() > 0)
    {
        this->opacity_accum.index_put_(
            {anchors_mask},
            torch::zeros({anchors_mask.sum().item<int>(), 1},
                         torch::TensorOptions().dtype(torch::kFloat32).device(this->device_type_)));
        this->anchor_demon.index_put_(
            {anchors_mask},
            torch::zeros({anchors_mask.sum().item<int>(), 1},
                         torch::TensorOptions().dtype(torch::kFloat32).device(this->device_type_)));
    }

    auto temp_opacity_accum = this->opacity_accum.index({~prune_mask});
    this->opacity_accum = temp_opacity_accum;

    auto temp_anchor_demon = this->anchor_demon.index({~prune_mask});
    this->anchor_demon = temp_anchor_demon;

    if (prune_mask.size(0) > 0)
    {
        this->prune_anchor(prune_mask);
    }

    this->max_radii2D = torch::zeros((this->get_anchor().size(0)), torch::TensorOptions().device(this->device_type_));
}

torch::Tensor GaussianModel::inverse_sigmoid(torch::Tensor x)
{
    return torch::log(x / (1 - x));
}

void GaussianModel::densificationPostfix(
    torch::Tensor& new_anchor,
    torch::Tensor& new_offsets,
    torch::Tensor& new_feat,
    torch::Tensor& new_opacities,
    torch::Tensor& new_scaling,
    torch::Tensor& new_rotation)
{
    std::vector<torch::Tensor> optimizable_tensors(6);
    std::vector<torch::Tensor> tensors_dict = {
        new_anchor,
        new_offsets,
        new_feat,
        new_opacities,
        new_scaling,
        new_rotation
    };
    auto& param_groups = this->optimizer_->param_groups();
    auto& state = this->optimizer_->state();
    for (int group_idx = 0; group_idx < 6; ++group_idx) {
        auto& group = param_groups[group_idx];
        assert(group.params().size() == 1);
        auto& extension_tensor = tensors_dict[group_idx];
        auto& param = group.params()[0];
        auto key = c10::guts::to_string(param.unsafeGetTensorImpl());
        if (state.find(key) != state.end()) {
            auto& stored_state = static_cast<torch::optim::AdamParamState&>(*state[key]);
            auto new_state = std::make_unique<torch::optim::AdamParamState>();
            new_state->step(stored_state.step());
            new_state->exp_avg(torch::cat({stored_state.exp_avg().clone(), torch::zeros_like(extension_tensor)}, /*dim=*/0));
            new_state->exp_avg_sq(torch::cat({stored_state.exp_avg_sq().clone(), torch::zeros_like(extension_tensor)}, /*dim=*/0));
            state.erase(key);
            param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();
            key = c10::guts::to_string(param.unsafeGetTensorImpl());
            state[key] = std::move(new_state);

            optimizable_tensors[group_idx] = param;
        }
        else {
            param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();
            optimizable_tensors[group_idx] = param;
        }
    }

    this->_anchor = optimizable_tensors[0];
    this->_offset = optimizable_tensors[1];
    this->_anchor_feat = optimizable_tensors[2];
    this->_opacity = optimizable_tensors[3];
    this->_scaling = optimizable_tensors[4];
    this->_rotation = optimizable_tensors[5];

    GAUSSIAN_MODEL_TENSORS_TO_VEC

    this->max_radii2D = torch::zeros({this->get_anchor().size(0)}, torch::TensorOptions().device(device_type_));
}

void GaussianModel::densificationPostfixCoarse(
    torch::Tensor& new_anchor,
    torch::Tensor& new_offsets,
    torch::Tensor& new_feat,
    torch::Tensor& new_opacities,
    torch::Tensor& new_scaling,
    torch::Tensor& new_rotation)
{

    int start_idx = 0;

    if (use_feat_bank)
    {
        if (appearance_dim > 0)
        {
            start_idx = 12;
        }
        else
            throw std::runtime_error("if use feature bank ,appearance_dim shoud > 0!");
    }
    else if (appearance_dim > 0)
    {
        start_idx = 11;
    }
    else
    {
        start_idx = 9;
    }
    std::vector<torch::Tensor> optimizable_tensors(6);
    std::vector<torch::Tensor> tensors_dict = {
        new_anchor,
        new_offsets,
        new_feat,
        new_opacities,
        new_scaling,
        new_rotation
    };
    auto &param_groups = this->optimizer_->param_groups();
    auto& state = this->optimizer_->state();
    for (int group_idx = start_idx; group_idx < start_idx + 6; ++group_idx) {
        auto& group = param_groups[group_idx];
        assert(group.params().size() == 1);
        auto& extension_tensor = tensors_dict[group_idx-start_idx];
        auto& param = group.params()[0];
        auto key = c10::guts::to_string(param.unsafeGetTensorImpl());
        if (state.find(key) != state.end()) {
            auto& stored_state = static_cast<torch::optim::AdamParamState&>(*state[key]);
            auto new_state = std::make_unique<torch::optim::AdamParamState>();
            new_state->step(stored_state.step());
            new_state->exp_avg(torch::cat({stored_state.exp_avg().clone(), torch::zeros_like(extension_tensor)}, /*dim=*/0));
            new_state->exp_avg_sq(torch::cat({stored_state.exp_avg_sq().clone(), torch::zeros_like(extension_tensor)}, /*dim=*/0));
            state.erase(key);
            param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();
            key = c10::guts::to_string(param.unsafeGetTensorImpl());
            state[key] = std::move(new_state);

            optimizable_tensors[group_idx-start_idx] = param;
        }
        else {
            param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();
            optimizable_tensors[group_idx-start_idx] = param;
        }

    }

    this->_anchor_c = optimizable_tensors[0];
    this->_offset_c = optimizable_tensors[1];
    this->_anchor_feat_c = optimizable_tensors[2];
    this->_opacity_c = optimizable_tensors[3];
    this->_scaling_c = optimizable_tensors[4];
    this->_rotation_c = optimizable_tensors[5];

    GAUSSIAN_MODEL_TENSORS_TO_VEC_COARSE

    this->max_radii2D_c = torch::zeros({this->_anchor_c.size(0)}, torch::TensorOptions().device(device_type_));
}
