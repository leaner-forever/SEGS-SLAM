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

#include "include/gaussian_trainer.h"

GaussianTrainer::GaussianTrainer()
{}

void GaussianTrainer::trainingOnce(
    std::shared_ptr<GaussianScene> scene,
    std::shared_ptr<GaussianModel> gaussians,
    GaussianModelParams& dataset,
    GaussianOptimizationParams& opt,
    GaussianPipelineParams& pipe,
    torch::DeviceType device_type, 
    std::vector<int> testing_iterations,
    std::vector<int> saving_iterations,
    std::vector<int> checkpoint_iterations)
{
    int first_iter = 0;
    gaussians->trainingSetup(opt);

    std::vector<float> bg_color;
    if (dataset.white_background_)
        bg_color = {1.0f, 1.0f, 1.0f};
    else
        bg_color = {0.0f, 0.0f, 0.0f};
    torch::Tensor background = torch::tensor(bg_color, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    float ema_loss_for_log = 0.0f;
    first_iter += 1;
    for (int iteration = first_iter; iteration <= opt.iterations_; ++iteration) {
        auto iter_start_timing = std::chrono::steady_clock::now();

        gaussians->updateLearningRate(iteration);

        if (iteration % 1000 == 0)
            gaussians->oneUpShDegree();

        std::map<std::size_t, std::shared_ptr<GaussianKeyframe>> viewpoint_stack;
        if (viewpoint_stack.empty())
            viewpoint_stack = scene->keyframes();
        int random_cam_idx = std::rand() / ((RAND_MAX + 1u) / static_cast<int>(viewpoint_stack.size()));
        auto random_cam_it = viewpoint_stack.begin();
        for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
            ++random_cam_it;
        std::shared_ptr<GaussianKeyframe> viewpoint_cam = (*random_cam_it).second;

        auto override_color = torch::empty(0, torch::TensorOptions().device(torch::kCUDA));
        auto voxel_visible_mask = GaussianRenderer::prefilter_voxel(
            viewpoint_cam,
            viewpoint_cam->image_height_,
            viewpoint_cam->image_width_,
            gaussians,
            pipe,
            background,
            override_color
        );
        bool retain_grad = true;
        auto render_pkg = GaussianRenderer::render(
            viewpoint_cam,
            viewpoint_cam->image_height_,
            viewpoint_cam->image_width_,
            gaussians,
            pipe,
            background,
            override_color,
            voxel_visible_mask,
            retain_grad
        );
        auto image = std::get<0>(render_pkg);
        auto viewspace_point_tensor = std::get<1>(render_pkg);
        auto visibility_filter = std::get<2>(render_pkg);
        auto radii = std::get<3>(render_pkg);

        auto gt_image = viewpoint_cam->original_image_.cuda();
        auto Ll1 = loss_utils::l1_loss(image, gt_image);
        auto loss = (1.0 - opt.lambda_dssim_) * Ll1 + opt.lambda_dssim_ * (1.0 - loss_utils::ssim(image, gt_image));
        loss.backward();

        torch::cuda::synchronize();
        auto iter_end_timing = std::chrono::steady_clock::now();
        auto iter_time = std::chrono::duration_cast<std::chrono::milliseconds>(iter_end_timing - iter_start_timing).count();

        {
            torch::NoGradGuard no_grad;
            ema_loss_for_log = 0.4f * loss.item().toFloat() + 0.6 * ema_loss_for_log;
            trainingReport(
                iteration,
                opt.iterations_,
                Ll1,
                loss,
                ema_loss_for_log,
                loss_utils::l1_loss,
                iter_time,
                *gaussians,
                *scene,
                pipe,
                background
            );

            if (iteration < opt.iterations_) {
                gaussians->optimizer_->step();
                gaussians->optimizer_->zero_grad(true);
            }
        }
    }
}

void GaussianTrainer::trainingReport(
    int iteration,
    int num_iterations,
    torch::Tensor& Ll1,
    torch::Tensor& loss,
    float ema_loss_for_log,
    std::function<torch::Tensor(torch::Tensor&, torch::Tensor&)> l1_loss,
    int64_t elapsed_time,
    GaussianModel& gaussians,
    GaussianScene& scene,
    GaussianPipelineParams& pipe,
    torch::Tensor& background)
{
    std::cout << std::fixed << std::setprecision(8)
              << "Training iteration " << iteration << "/" << num_iterations
              << ", time elapsed:" << elapsed_time / 1000.0 << "s"
              << ", ema_loss:" << ema_loss_for_log
              << std::endl;
}