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

#include "include/gaussian_renderer.h"


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, bool>
GaussianRenderer::render(
    std::shared_ptr<GaussianKeyframe> viewpoint_camera,
    int image_height,
    int image_width,
    std::shared_ptr<GaussianModel> pc,
    GaussianPipelineParams &pipe,
    torch::Tensor &bg_color,
    torch::Tensor &override_color,
    torch::Tensor &visible_mask,
    bool retain_grad,
    float scaling_modifier,
    bool use_override_color)
{
    bool is_training = pc->get_color_mlp()->is_training();

    auto neural_gaussians = generate_neural_gaussians(
        viewpoint_camera,
        image_height,
        image_width,
        pc,
        visible_mask,
        is_training);
    auto xyz_ = std::get<0>(neural_gaussians);
    auto color_ = std::get<1>(neural_gaussians);
    auto opacity_ = std::get<2>(neural_gaussians);
    auto scaling_ = std::get<3>(neural_gaussians);
    auto rot_ = std::get<4>(neural_gaussians);

    torch::Tensor neural_opacity, mask;
    if (is_training) {
        neural_opacity = std::get<5>(neural_gaussians);
        mask = std::get<6>(neural_gaussians);
    }

    auto screenspace_points = torch::zeros_like(xyz_,
        torch::TensorOptions().dtype(pc->get_anchor().dtype()).requires_grad(true).device(torch::kCUDA));
    if(retain_grad)
    {
        try {
            screenspace_points.retain_grad();
        }
        catch (const std::exception& e) {
            ; 
        }
    }

    float tanfovx = std::tan(viewpoint_camera->FoVx_ * 0.5f);
    float tanfovy = std::tan(viewpoint_camera->FoVy_ * 0.5f);

    GaussianRasterizationSettings raster_settings(
        image_height,
        image_width,
        tanfovx,
        tanfovy,
        bg_color,
        scaling_modifier,
        viewpoint_camera->world_view_transform_,
        viewpoint_camera->full_proj_transform_,
        pc->active_sh_degree_,
        viewpoint_camera->camera_center_,
        false
    );

    GaussianRasterizer rasterizer(raster_settings);

    bool has_scales = true,
         has_rotations = true,
         has_color_precomp = true;
    auto means3D = xyz_;
    auto means2D = screenspace_points;
    auto colors_precomp = color_;
    auto opacity = opacity_;
    auto scales = scaling_;
    auto rotations = rot_;

    bool has_shs = false,
         has_cov3D_precomp = false;
    torch::Tensor shs = torch::empty({0});
    torch::Tensor cov3D_precomp = torch::empty({0});

    auto rasterizer_result = rasterizer.forward(
        means3D,
        means2D,
        opacity,
        has_shs,
        has_color_precomp,
        has_scales,
        has_rotations,
        has_cov3D_precomp,
        shs,
        colors_precomp,
        scales,
        rotations,
        cov3D_precomp
    );
    auto rendered_image = std::get<0>(rasterizer_result);
    auto radii = std::get<1>(rasterizer_result);

    return std::make_tuple(
        rendered_image,             
        screenspace_points, 
        radii > 0,          
        radii,               
        mask,               
        neural_opacity,     
        scaling_,          
        is_training         
    );
}

torch::Tensor GaussianRenderer::prefilter_voxel(
    std::shared_ptr<GaussianKeyframe> viewpoint_camera,
    int image_height,
    int image_width,
    std::shared_ptr<GaussianModel> pc,
    GaussianPipelineParams& pipe,
    torch::Tensor& bg_color,
    torch::Tensor& override_color,
    float scaling_modifier,
    bool use_override_color)
{
    auto screenspace_points = torch::zeros_like(pc->get_anchor(),
        torch::TensorOptions().dtype(pc->get_anchor().dtype()).requires_grad(true).device(torch::kCUDA));
    try {
        screenspace_points.retain_grad();
    }
    catch (const std::exception& e) {
        ; 
    }
    float tanfovx = std::tan(viewpoint_camera->FoVx_ * 0.5f);
    float tanfovy = std::tan(viewpoint_camera->FoVy_ * 0.5f);

    GaussianRasterizationSettings raster_settings(
        image_height,
        image_width,
        tanfovx,
        tanfovy,
        bg_color,
        scaling_modifier,
        viewpoint_camera->world_view_transform_,
        viewpoint_camera->full_proj_transform_,
        pc->active_sh_degree_,
        viewpoint_camera->camera_center_,
        false
    );
    GaussianRasterizer rasterizer(raster_settings);

    auto means3D = pc->get_anchor();
    
    bool has_scales = false,
         has_rotations = false,
         has_cov3D_precomp = false;
    torch::Tensor scales,
                  rotations,
                  cov3D_precomp;
    if (pipe.compute_cov3D_) {
        cov3D_precomp = pc->get_covariance(scaling_modifier);
        has_cov3D_precomp = true;
    }
    else {
        scales = pc->get_scaling();
        rotations = pc->get_rotation();
        has_scales = true;
        has_rotations = true;
    }   

    auto scale = scales.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)});

    auto radii_pure = rasterizer.visible_filter(
        means3D,
        has_scales,
        has_rotations,
        has_cov3D_precomp,
        scale,
        rotations,
        cov3D_precomp
    );
    return radii_pure > 0;
}

torch::Tensor positional_encoding(torch::Tensor x, int max_freq) {
    std::vector<torch::Tensor> encoding;
    for (int i = 0; i < max_freq; ++i) {
        double freq = std::pow(2.0, i);
        torch::Tensor sin_enc = torch::sin(freq * M_PI * x);
        torch::Tensor cos_enc = torch::cos(freq * M_PI * x);
        encoding.push_back(sin_enc);
        encoding.push_back(cos_enc);
    }
    encoding.push_back(x);
    return torch::cat(encoding, -1);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
GaussianRenderer::generate_neural_gaussians(
    std::shared_ptr<GaussianKeyframe> viewpoint_camera,
    int image_height,
    int image_width,
    std::shared_ptr<GaussianModel> pc,
    torch::Tensor &visible_mask,
    bool is_training)
{ 
    if (!visible_mask.defined()) {
        visible_mask = torch::ones({pc->get_anchor().size(0)},
            torch::TensorOptions().dtype(pc->get_anchor().dtype()).requires_grad(true).device(torch::kCUDA));
    }

    auto feat = pc->_anchor_feat.index({visible_mask});
    auto anchor = pc->get_anchor().index({visible_mask});
    auto grid_offsets = pc->_offset.index({visible_mask});
    auto grid_scaling = pc->get_scaling().index({visible_mask});
    auto ob_view = anchor - viewpoint_camera->camera_center_;
    auto ob_dist = torch::frobenius_norm(ob_view, /*dim=*/{1}, /*keepdim=*/true);
    ob_view = ob_view / ob_dist;

    if (pc->use_feat_bank){
        auto cat_view = torch::cat({ob_view, ob_dist}, /*dim=*/1);

        auto bank_weight = pc->get_featurebank_mlp()->forward(cat_view).unsqueeze(/*dim=*/1);

        feat = feat.unsqueeze(/*dim=*/-1);
        feat = feat.index({torch::indexing::Slice(), torch::indexing::Slice(None_, None_, 4), torch::indexing::Slice(None_, 1)}).repeat({1, 4, 1}) *
                   bank_weight.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(None_, 1)}) +
               feat.index({torch::indexing::Slice(), torch::indexing::Slice(None_, None_, 2), torch::indexing::Slice(None_, 1)}).repeat({1, 2, 1}) *
                   bank_weight.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, 2)}) +
               feat.index({torch::indexing::Slice(), torch::indexing::Slice(None_, None_, 1), torch::indexing::Slice(None_, 1)}) *
                   bank_weight.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(2, None_)});
        feat = feat.squeeze(/*dim=*/-1); 
    }

    auto cat_local_view = torch::cat({feat, ob_view, ob_dist}, 1);
    auto cat_local_view_wodist = torch::cat({feat, ob_view}, 1);
    torch::Tensor appearance;

    torch::Tensor appearance_feat;
    if (pc->appearance_dim > 0)
    {
        std::vector<float> pose = {static_cast<float>(viewpoint_camera->t_.x()), static_cast<float>(viewpoint_camera->t_.y()),
                                    static_cast<float>(viewpoint_camera->t_.z()), static_cast<float>(viewpoint_camera->R_quaternion_.w()),
                                    static_cast<float>(viewpoint_camera->R_quaternion_.x()), static_cast<float>(viewpoint_camera->R_quaternion_.y()),
                                    static_cast<float>(viewpoint_camera->R_quaternion_.z())};

        auto ob_pose = torch::from_blob(pose.data(), {7, 1},
                           torch::TensorOptions().dtype(torch::kFloat32)).to(ob_dist.device()).transpose(0, 1);

        torch::Tensor ob_pose_expanded = ob_pose.expand({cat_local_view.size(0), -1});
        

        appearance_feat = pc->mlp_apperance->forward(ob_pose_expanded);
    }

    torch::Tensor neural_opacity;
    if (pc->add_opacity_dist)
        neural_opacity = pc->get_opacity_mlp()->forward(cat_local_view);
    else
        neural_opacity = pc->get_opacity_mlp()->forward(cat_local_view_wodist);
    
    neural_opacity = neural_opacity.reshape({-1,1});
    auto mask = (neural_opacity > 0.0);
    mask = mask.view({-1});

    auto opacity = neural_opacity.index({mask});

    torch::Tensor color;
    if(pc->appearance_dim>0)
    {
        if(pc->add_color_dist )
            color = pc->get_color_mlp()->forward(torch::cat({cat_local_view, appearance_feat}, 1));
        else
            color = pc->get_color_mlp()->forward(torch::cat({cat_local_view_wodist, appearance_feat}, 1));
    }
    else
    {
        if(pc->add_color_dist )
            color = pc->get_color_mlp()->forward(cat_local_view);
        else
            color = pc->get_color_mlp()->forward(cat_local_view_wodist);
    }
    color = color.reshape({anchor.size(0) * pc->n_offsets, 3});

    torch::Tensor scale_rot;
    if(pc->add_cov_dist)
        scale_rot = pc->get_cov_mlp()->forward(cat_local_view);
    else
        scale_rot = pc->get_cov_mlp()->forward(cat_local_view_wodist);
    scale_rot = scale_rot.reshape({anchor.size(0) * pc->n_offsets, 7});
    
    auto offsets = grid_offsets.view({-1, 3});
    auto concatenated = torch::cat({grid_scaling, anchor}, -1);
    torch::Tensor concatenated_repeated;
    {
        auto sizes = concatenated.sizes();
        int n = sizes[0];
        int c = sizes[1];
        concatenated_repeated = concatenated.repeat({1, pc->n_offsets});
        concatenated_repeated = concatenated_repeated.view({n * pc->n_offsets, c});
    }
    
    auto concatenated_all = torch::cat({concatenated_repeated, color, scale_rot, offsets}, -1);
    auto masked = concatenated_all.index({mask});
    auto masked_split = masked.split({6, 3, 3, 7, 3}, -1);
    auto scaling_repeat = masked_split[0];
    auto repeat_anchor = masked_split[1];
    color = masked_split[2];
    scale_rot = masked_split[3];
    offsets = masked_split[4];
    auto scaling = scaling_repeat.index({torch::indexing::Slice(), torch::indexing::Slice(3, None_)}) *
              torch::sigmoid(scale_rot.index({torch::indexing::Slice(), torch::indexing::Slice(None_, 3)}));
    auto rot = torch::nn::functional::normalize(scale_rot.index({torch::indexing::Slice(),
                                                                 torch::indexing::Slice(3, 7)}));
    offsets = offsets * scaling_repeat.index({torch::indexing::Slice(), torch::indexing::Slice(None_, 3)});
    auto xyz = repeat_anchor + offsets;
    return std::make_tuple(xyz, color, opacity, scaling, rot, neural_opacity, mask );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GaussianRenderer::gaussians_project2_image(
    std::shared_ptr<GaussianKeyframe> viewpoint_camera,
    int image_height,
    int image_width,
    std::shared_ptr<GaussianModel> pc,
    GaussianPipelineParams &pipe,
    torch::Tensor &bg_color,
    torch::Tensor &override_color,
    torch::Tensor &visible_mask,
    float scaling_modifier,
    bool use_override_color)
{
    bool is_training = pc->get_color_mlp()->is_training();

    auto neural_gaussians = generate_neural_gaussians(
        viewpoint_camera,
        image_height,
        image_width,
        pc,
        visible_mask,
        is_training);
    auto xyz_ = std::get<0>(neural_gaussians);
    auto color_ = std::get<1>(neural_gaussians);
    auto opacity_ = std::get<2>(neural_gaussians);
    auto scaling_ = std::get<3>(neural_gaussians);
    auto rot_ = std::get<4>(neural_gaussians);

    auto screenspace_points = torch::zeros_like(xyz_,
        torch::TensorOptions().dtype(pc->get_anchor().dtype()).requires_grad(true).device(torch::kCUDA));

    float tanfovx = std::tan(viewpoint_camera->FoVx_ * 0.5f);
    float tanfovy = std::tan(viewpoint_camera->FoVy_ * 0.5f);

    GaussianRasterizationSettings raster_settings(
        image_height,
        image_width,
        tanfovx,
        tanfovy,
        bg_color,
        scaling_modifier,
        viewpoint_camera->world_view_transform_,
        viewpoint_camera->full_proj_transform_,
        pc->active_sh_degree_,
        viewpoint_camera->camera_center_,
        false
    );

    GaussianRasterizer rasterizer(raster_settings);

    bool has_scales = true,
         has_rotations = true,
         has_color_precomp = true;
    auto means3D = xyz_;
    auto means2D = screenspace_points;
    auto colors_precomp = color_;
    auto opacity = opacity_;
    auto scales = scaling_;
    auto rotations = rot_;

    bool has_shs = false,
         has_cov3D_precomp = false;
    torch::Tensor shs = torch::empty({0});
    torch::Tensor cov3D_precomp = torch::empty({0});

    auto result = rasterizer.project2_image(
        means3D,
        means2D,
        opacity,
        has_shs,
        has_color_precomp,
        has_scales,
        has_rotations,
        has_cov3D_precomp,
        shs,
        colors_precomp,
        scales,
        rotations,
        cov3D_precomp
    );
    auto points_image_2d = std::get<0>(result);
    auto radii = std::get<1>(result);
    auto color = std::get<2>(result);

    return std::make_tuple(
        points_image_2d, 
        radii,          
        color);   
}

