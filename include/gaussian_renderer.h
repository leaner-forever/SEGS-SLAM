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

#include <tuple>

#include <torch/torch.h>

#include "sh_utils.h"

#include "gaussian_parameters.h"
#include "gaussian_keyframe.h"
#include "gaussian_model.h"
#include "gaussian_rasterizer.h"

#include "tensor_utils.h"

#define None_  torch::indexing::None

class GaussianRenderer
{
public:
    static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                      torch::Tensor, torch::Tensor, torch::Tensor, bool>
    render(
        std::shared_ptr<GaussianKeyframe> viewpoint_camera,
        int image_height,
        int image_width,
        std::shared_ptr<GaussianModel> gaussians,
        GaussianPipelineParams &pipe,
        torch::Tensor &bg_color,
        torch::Tensor &override_color,
        torch::Tensor &visible_mask,
        bool retain_grad = false,
        float scaling_modifier = 1.0f,
        bool has_override_color = false);

    static torch::Tensor prefilter_voxel(
        std::shared_ptr<GaussianKeyframe> viewpoint_camera,
        int image_height,
        int image_width,
        std::shared_ptr<GaussianModel> gaussians,
        GaussianPipelineParams& pipe,
        torch::Tensor& bg_color,
        torch::Tensor& override_color,
        float scaling_modifier = 1.0f,
        bool has_override_color = false);

    static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    generate_neural_gaussians(
        std::shared_ptr<GaussianKeyframe> viewpoint_camera,
        int image_height,
        int image_width,
        std::shared_ptr<GaussianModel> gaussians,
        torch::Tensor &visible_mask,
        bool is_training = false);

    static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    generate_neural_gaussians_coarse(
        std::shared_ptr<GaussianKeyframe> viewpoint_camera,
        int image_height,
        int image_width,
        std::shared_ptr<GaussianModel> gaussians,
        torch::Tensor &visible_mask,
        bool is_training = false);

    static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> gaussians_project2_image(
        std::shared_ptr<GaussianKeyframe> viewpoint_camera,
        int image_height,
        int image_width,
        std::shared_ptr<GaussianModel> gaussians,
        GaussianPipelineParams& pipe,
        torch::Tensor& bg_color,
        torch::Tensor& override_color,
        torch::Tensor &visible_mask,
        float scaling_modifier = 1.0f,
        bool has_override_color = false);
};
