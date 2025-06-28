/**
 * This file is part of Photo-SLAM
 *
 * Copyright (C) 2023-2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2023-2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University of Science and Technology.
 *
 * Photo-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Photo-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Photo-SLAM.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "include/gaussian_mapper.h"

GaussianMapper::GaussianMapper(
    std::shared_ptr<ORB_SLAM3::System> pSLAM,
    std::filesystem::path gaussian_config_file_path,
    std::filesystem::path result_dir,
    int seed,
    torch::DeviceType device_type)
    : pSLAM_(pSLAM),
      initial_mapped_(false),
      interrupt_training_(false),
      stopped_(false),
      iteration_(0),
      ema_loss_for_log_(0.0f),
      SLAM_ended_(false),
      loop_closure_iteration_(false),
      min_num_initial_map_kfs_(15UL),
      large_rot_th_(1e-1f),
      large_trans_th_(1e-2f),
      training_report_interval_(0),
      vTimestamps(nullptr)
{
    std::srand(0);
    torch::manual_seed(0);
    torch::cuda::manual_seed(0);

    if (device_type == torch::kCUDA && torch::cuda::is_available()) {
        std::cout << "[Gaussian Mapper]CUDA available! Training on GPU." << std::endl;
        device_type_ = torch::kCUDA;
        model_params_.data_device_ = "cuda";
    }
    else {
        std::cout << "[Gaussian Mapper]Training on CPU." << std::endl;
        device_type_ = torch::kCPU;
        model_params_.data_device_ = "cpu";
    }

    result_dir_ = result_dir;
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    config_file_path_ = gaussian_config_file_path;
    readConfigFromFile(gaussian_config_file_path);

    std::vector<float> bg_color;
    if (model_params_.white_background_)
        bg_color = {1.0f, 1.0f, 1.0f};
    else
        bg_color = {0.0f, 0.0f, 0.0f};
    background_ = torch::tensor(bg_color,
                    torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    
    override_color_ = torch::empty(0, torch::TensorOptions().device(device_type_));

    gaussians_ = std::make_shared<GaussianModel>(model_params_);
    scene_ = std::make_shared<GaussianScene>(model_params_);

    if (!pSLAM) {
        return;
        gaussians_->hasSLAM = false;
    }
    gaussians_->hasSLAM = true;

    switch (pSLAM->getSensorType())
    {
    case ORB_SLAM3::System::MONOCULAR:
    case ORB_SLAM3::System::IMU_MONOCULAR:
    {
        this->sensor_type_ = MONOCULAR;
    }
    break;
    case ORB_SLAM3::System::STEREO:
    case ORB_SLAM3::System::IMU_STEREO:
    {
        this->sensor_type_ = STEREO;
        this->stereo_baseline_length_ = pSLAM->getSettings()->b();
        this->stereo_cv_sgm_ = cv::cuda::createStereoSGM(
            this->stereo_min_disparity_,
            this->stereo_num_disparity_);
        this->stereo_Q_ = pSLAM->getSettings()->Q().clone();
        stereo_Q_.convertTo(stereo_Q_, CV_32FC3, 1.0);
    }
    break;
    case ORB_SLAM3::System::RGBD:
    case ORB_SLAM3::System::IMU_RGBD:
    {
        this->sensor_type_ = RGBD;
    }
    break;
    default:
    {
        throw std::runtime_error("[Gaussian Mapper]Unsupported sensor type!");
    }
    break;
    }

    auto settings = pSLAM->getSettings();
    cv::Size SLAM_im_size = settings->newImSize();
    UndistortParams undistort_params(
        SLAM_im_size,
        settings->camera1DistortionCoef()
    );

    auto vpCameras = pSLAM->getAtlas()->GetAllCameras();
    for (auto& SLAM_camera : vpCameras) {
        Camera camera;
        camera.camera_id_ = SLAM_camera->GetId();
        if (SLAM_camera->GetType() == ORB_SLAM3::GeometricCamera::CAM_PINHOLE) {
            camera.setModelId(Camera::CameraModelType::PINHOLE);
            float SLAM_fx = SLAM_camera->getParameter(0);
            float SLAM_fy = SLAM_camera->getParameter(1);
            float SLAM_cx = SLAM_camera->getParameter(2);
            float SLAM_cy = SLAM_camera->getParameter(3);

            cv::Mat K = (
                cv::Mat_<float>(3, 3)
                    << SLAM_fx, 0.f, SLAM_cx,
                        0.f, SLAM_fy, SLAM_cy,
                        0.f, 0.f, 1.f
            );

            camera.width_ = undistort_params.old_size_.width;
            float x_ratio = static_cast<float>(camera.width_) / undistort_params.old_size_.width;

            camera.height_ = undistort_params.old_size_.height;
            float y_ratio = static_cast<float>(camera.height_) / undistort_params.old_size_.height;

            camera.num_gaus_pyramid_sub_levels_ = num_gaus_pyramid_sub_levels_;
            camera.gaus_pyramid_width_.resize(num_gaus_pyramid_sub_levels_);
            camera.gaus_pyramid_height_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                camera.gaus_pyramid_width_[l] = camera.width_ * this->kf_gaus_pyramid_factors_[l];
                camera.gaus_pyramid_height_[l] = camera.height_ * this->kf_gaus_pyramid_factors_[l];
            }

            camera.params_[0] = SLAM_fx * x_ratio;
            camera.params_[1] = SLAM_fy * y_ratio;
            camera.params_[2] = SLAM_cx * x_ratio;
            camera.params_[3] = SLAM_cy * y_ratio;

            cv::Mat K_new = (
                cv::Mat_<float>(3, 3)
                    << camera.params_[0], 0.f, camera.params_[2],
                        0.f, camera.params_[1], camera.params_[3],
                        0.f, 0.f, 1.f
            );

            if (this->sensor_type_ == MONOCULAR || this->sensor_type_ == RGBD)
                undistort_params.dist_coeff_.copyTo(camera.dist_coeff_);

            camera.initUndistortRectifyMapAndMask(K, SLAM_im_size, K_new, true);

            undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    camera.undistort_mask, device_type_);

            cv::Mat viewer_sub_undistort_mask;
            int viewer_image_height_ = camera.height_ * rendered_image_viewer_scale_;
            int viewer_image_width_ = camera.width_ * rendered_image_viewer_scale_;
            cv::resize(camera.undistort_mask, viewer_sub_undistort_mask,
                       cv::Size(viewer_image_width_, viewer_image_height_));
            viewer_sub_undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_sub_undistort_mask, device_type_);

            cv::Mat viewer_main_undistort_mask;
            int viewer_image_height_main_ = camera.height_ * rendered_image_viewer_scale_main_;
            int viewer_image_width_main_ = camera.width_ * rendered_image_viewer_scale_main_;
            cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                       cv::Size(viewer_image_width_main_, viewer_image_height_main_));
            viewer_main_undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_main_undistort_mask, device_type_);

            if (this->sensor_type_ == STEREO) {
                camera.stereo_bf_ = stereo_baseline_length_ * camera.params_[0];
                if (this->stereo_Q_.cols != 4) {
                    this->stereo_Q_ = cv::Mat(4, 4, CV_32FC1);
                    this->stereo_Q_.setTo(0.0f);
                    this->stereo_Q_.at<float>(0, 0) = 1.0f;
                    this->stereo_Q_.at<float>(0, 3) = -camera.params_[2];
                    this->stereo_Q_.at<float>(1, 1) = 1.0f;
                    this->stereo_Q_.at<float>(1, 3) = -camera.params_[3];
                    this->stereo_Q_.at<float>(2, 3) = camera.params_[0];
                    this->stereo_Q_.at<float>(3, 2) = 1.0f / stereo_baseline_length_;
                }
            }
        }
        else if (SLAM_camera->GetType() == ORB_SLAM3::GeometricCamera::CAM_FISHEYE) {
            camera.setModelId(Camera::CameraModelType::FISHEYE);
        }
        else {
            camera.setModelId(Camera::CameraModelType::INVALID);
        }

        if (!viewer_camera_id_set_) {
            viewer_camera_id_ = camera.camera_id_;
            viewer_camera_id_set_ = true;
        }
        this->scene_->addCamera(camera);
    }

    kdtreePointsfull.reset(new pcl::KdTreeFLANN<pcl::PointXYZRGB>());
    pointCloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());

}

void GaussianMapper::readConfigFromFile(std::filesystem::path cfg_path)
{
    cv::FileStorage settings_file(cfg_path.string().c_str(), cv::FileStorage::READ);
    if(!settings_file.isOpened()) {
       std::cerr << "[Gaussian Mapper]Failed to open settings file at: " << cfg_path << std::endl;
       exit(-1);
    }

    std::cout << "[Gaussian Mapper]Reading parameters from " << cfg_path << std::endl;
    std::unique_lock<std::mutex> lock(mutex_settings_);

    model_params_.sh_degree_ =
        settings_file["Model.sh_degree"].operator int();
    model_params_.resolution_ =
        settings_file["Model.resolution"].operator float();
    model_params_.white_background_ =
        (settings_file["Model.white_background"].operator int()) != 0;
    model_params_.eval_ =
        (settings_file["Model.eval"].operator int()) != 0;

    model_params_.feat_dim =
        settings_file["Model.feat_dim"].operator int();
    model_params_.n_offsets =
        settings_file["Model.n_offsets"].operator int();
    model_params_.voxel_size =
        settings_file["Model.voxel_size"].operator float();
    model_params_.update_depth =
        settings_file["Model.update_depth"].operator int();
    model_params_.update_init_factor =
        settings_file["Model.update_init_factor"].operator int();
    model_params_.update_hierachy_factor =
        settings_file["Model.update_hierachy_factor"].operator int();

    model_params_.use_feat_bank =
        (settings_file["Model.use_feat_bank"].operator int()) != 0;
    model_params_.appearance_dim =
        settings_file["Model.appearance_dim"].operator int();
    model_params_.lowpoly =
        (settings_file["Model.lowpoly"].operator int()) != 0;
    model_params_.ds =
        settings_file["Model.ds"].operator int();
    model_params_.ratio =
        settings_file["Model.ratio"].operator float();
    model_params_.undistorted =
        (settings_file["Model.undistorted"].operator int()) != 0;
    model_params_.embedding_dim =
        settings_file["Model.embedding_dim"].operator int();

    model_params_.add_opacity_dist =
        (settings_file["Model.add_opacity_dist"].operator int()) != 0;
    model_params_.add_cov_dist =
        (settings_file["Model.add_cov_dist"].operator int()) != 0;
    model_params_.add_color_dist =
        (settings_file["Model.add_color_dist"].operator int()) != 0;

    z_near_ =
        settings_file["Camera.z_near"].operator float();
    z_far_ =
        settings_file["Camera.z_far"].operator float();

    monocular_inactive_geo_densify_max_pixel_dist_ =
        settings_file["Monocular.inactive_geo_densify_max_pixel_dist"].operator float();
    stereo_min_disparity_ =
        settings_file["Stereo.min_disparity"].operator int();
    stereo_num_disparity_ =
        settings_file["Stereo.num_disparity"].operator int();
    RGBD_min_depth_ =
        settings_file["RGBD.min_depth"].operator float();
    RGBD_max_depth_ =
        settings_file["RGBD.max_depth"].operator float();

    inactive_geo_densify_ =
        (settings_file["Mapper.inactive_geo_densify"].operator int()) != 0;
    max_depth_cached_ =
        settings_file["Mapper.depth_cache"].operator int();
    min_num_initial_map_kfs_ = 
        static_cast<unsigned long>(settings_file["Mapper.min_num_initial_map_kfs"].operator int());
    new_keyframe_times_of_use_ = 
        settings_file["Mapper.new_keyframe_times_of_use"].operator int();
    local_BA_increased_times_of_use_ = 
        settings_file["Mapper.local_BA_increased_times_of_use"].operator int();
    loop_closure_increased_times_of_use_ = 
        settings_file["Mapper.loop_closure_increased_times_of_use_"].operator int();
    cull_keyframes_ =
        (settings_file["Mapper.cull_keyframes"].operator int()) != 0;
    large_rot_th_ =
        settings_file["Mapper.large_rotation_threshold"].operator float();
    large_trans_th_ =
        settings_file["Mapper.large_translation_threshold"].operator float();
    stable_num_iter_existence_ =
        settings_file["Mapper.stable_num_iter_existence"].operator int();

    pipe_params_.convert_SHs_ =
        (settings_file["Pipeline.convert_SHs"].operator int()) != 0;
    pipe_params_.compute_cov3D_ =
        (settings_file["Pipeline.compute_cov3D"].operator int()) != 0;

    do_gaus_pyramid_training_ =
        (settings_file["GausPyramid.do"].operator int()) != 0;
    num_gaus_pyramid_sub_levels_ =
        settings_file["GausPyramid.num_sub_levels"].operator int();
    int sub_level_times_of_use =
        settings_file["GausPyramid.sub_level_times_of_use"].operator int();
    kf_gaus_pyramid_times_of_use_.resize(num_gaus_pyramid_sub_levels_);
    kf_gaus_pyramid_factors_.resize(num_gaus_pyramid_sub_levels_);
    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
        kf_gaus_pyramid_times_of_use_[l] = sub_level_times_of_use;
        kf_gaus_pyramid_factors_[l] = std::pow(0.5f, num_gaus_pyramid_sub_levels_ - l);
    }

    keyframe_record_interval_ = 
        settings_file["Record.keyframe_record_interval"].operator int();
    all_keyframes_record_interval_ = 
        settings_file["Record.all_keyframes_record_interval"].operator int();
    record_rendered_image_ = 
        (settings_file["Record.record_rendered_image"].operator int()) != 0;
    record_ground_truth_image_ = 
        (settings_file["Record.record_ground_truth_image"].operator int()) != 0;
    record_loss_image_ = 
        (settings_file["Record.record_loss_image"].operator int()) != 0;
    record_debug_image_ = 
        (settings_file["Record.record_debug_image"].operator int()) != 0;
    training_report_interval_ = 
        settings_file["Record.training_report_interval"].operator int();
    record_loop_ply_ =
        (settings_file["Record.record_loop_ply"].operator int()) != 0;
    
    light_mode  =
        (settings_file["Mapper.light_mode"].operator int()) != 0;

    this->use_frequency_regularization = (settings_file["Mapper.use_frequency_regularization"].operator int()) != 0;
    this->use_multi_resolution = (settings_file["Mapper.use_multi_resolution"].operator int()) != 0;
    this->scale_num = settings_file["Mapper.scale_num"].operator int();
    this->frequency_regulization_until = settings_file["Mapper.frequency_regulization_until"].operator int();
    this->high_frequency_regularization_start = settings_file["Mapper.high_frequency_regularization_start"].operator int();
    this->lambda_frequency_high = settings_file["Mapper.lambda_frequency_high"].operator float();
    this->lambda_frequency_low = settings_file["Mapper.lambda_frequency_low"].operator float();

    opt_params_.iterations_ =
        settings_file["Optimization.max_num_iterations"].operator int();
    opt_params_.position_lr_init_ =
        settings_file["Optimization.position_lr_init"].operator float();
    opt_params_.position_lr_final_ =
        settings_file["Optimization.position_lr_final"].operator float();
    opt_params_.position_lr_delay_mult_ =
        settings_file["Optimization.position_lr_delay_mult"].operator float();
    opt_params_.position_lr_max_steps_ =
        settings_file["Optimization.position_lr_max_steps"].operator int();
    opt_params_.feature_lr_ =
        settings_file["Optimization.feature_lr"].operator float();
    opt_params_.opacity_lr_ =
        settings_file["Optimization.opacity_lr"].operator float();
    opt_params_.scaling_lr_ =
        settings_file["Optimization.scaling_lr"].operator float();
    opt_params_.rotation_lr_ =
        settings_file["Optimization.rotation_lr"].operator float();

    opt_params_.percent_dense_ =
        settings_file["Optimization.percent_dense"].operator float();
    opt_params_.lambda_dssim_ =
        settings_file["Optimization.lambda_dssim"].operator float();

    opt_params_.offset_lr_init =
        settings_file["Optimization.offset_lr_init"].operator float();
    opt_params_.offset_lr_final = settings_file["Optimization.offset_lr_final"].operator float();
    opt_params_.offset_lr_delay_mult =settings_file["Optimization.offset_lr_delay_mult"].operator float();
    opt_params_.offset_lr_max_steps =settings_file["Optimization.offset_lr_max_steps"].operator int();

    opt_params_.mlp_opacity_lr_init =
        settings_file["Optimization.mlp_opacity_lr_init"].operator float();
    opt_params_.mlp_opacity_lr_final = settings_file["Optimization.mlp_opacity_lr_final"].operator float();
    opt_params_.mlp_opacity_lr_delay_mult =settings_file["Optimization.mlp_opacity_lr_delay_mult"].operator float();
    opt_params_.mlp_opacity_lr_max_steps =settings_file["Optimization.mlp_opacity_lr_max_steps"].operator int();

    opt_params_.mlp_cov_lr_init =settings_file["Optimization.mlp_cov_lr_init"].operator float();
    opt_params_.mlp_cov_lr_final =settings_file["Optimization.mlp_cov_lr_final"].operator float();
    opt_params_.mlp_cov_lr_delay_mult =settings_file["Optimization.mlp_cov_lr_delay_mult"].operator float();
    opt_params_.mlp_cov_lr_max_steps =settings_file["Optimization.mlp_cov_lr_max_steps"].operator int();

    opt_params_.mlp_color_lr_init =settings_file["Optimization.mlp_color_lr_init"].operator float();
    opt_params_.mlp_color_lr_final =settings_file["Optimization.mlp_color_lr_final"].operator float();
    opt_params_.mlp_color_lr_delay_mult =settings_file["Optimization.mlp_color_lr_delay_mult"].operator float();
    opt_params_.mlp_color_lr_max_steps =settings_file["Optimization.mlp_color_lr_max_steps"].operator int();

    opt_params_.mlp_featurebank_lr_init =settings_file["Optimization.mlp_featurebank_lr_init"].operator float();
    opt_params_.mlp_featurebank_lr_final =settings_file["Optimization.mlp_featurebank_lr_final"].operator float();
    opt_params_.mlp_featurebank_lr_delay_mult =settings_file["Optimization.mlp_featurebank_lr_delay_mult"].operator float();
    opt_params_.mlp_featurebank_lr_max_steps =settings_file["Optimization.mlp_featurebank_lr_max_steps"].operator int();

    opt_params_.appearance_lr_init =settings_file["Optimization.appearance_lr_init"].operator float();
    opt_params_.appearance_lr_final =settings_file["Optimization.appearance_lr_final"].operator float();
    opt_params_.appearance_lr_delay_mult =settings_file["Optimization.appearance_lr_delay_mult"].operator float();
    opt_params_.appearance_lr_max_steps =settings_file["Optimization.appearance_lr_max_steps"].operator int();

    opt_params_.start_stat =settings_file["Optimization.start_stat"].operator int();
    opt_params_.update_from =settings_file["Optimization.update_from"].operator int();
    opt_params_.update_interval =settings_file["Optimization.update_interval"].operator int();
    opt_params_.update_until =settings_file["Optimization.update_until"].operator int();
    opt_params_.min_opacity =settings_file["Optimization.min_opacity"].operator float();
    opt_params_.success_threshold =settings_file["Optimization.success_threshold"].operator float();


    prune_big_point_after_iter_ =
        settings_file["Optimization.prune_big_point_after_iter"].operator int();
    densify_min_opacity_ =
        settings_file["Optimization.densify_min_opacity"].operator float();
    opt_params_.densify_grad_threshold =
        settings_file["Optimization.densify_grad_threshold"].operator float();

    model_params_.use_coarse_anchor =
        (settings_file["Model.use_coarse_anchor"].operator int()) != 0;
    model_params_.feat_dim_coarse =
        settings_file["Model.feat_dim_coarse"].operator int();
    model_params_.n_offsets_coarse =
        settings_file["Model.n_offsets_coarse"].operator int();
    model_params_.coarse_voxel_size =
        settings_file["Model.coarse_voxel_size"].operator float();
    model_params_.appearance_dim_coarse =
        settings_file["Model.appearance_dim_coarse"].operator int();

    opt_params_.anchor_lr_init_coarse =
        settings_file["Optimization.anchor_lr_init_coarse"].operator float();
    opt_params_.anchor_lr_final_coarse =
        settings_file["Optimization.anchor_lr_final_coarse"].operator float();
    opt_params_.anchor_lr_delay_mult_coarse =
        settings_file["Optimization.anchor_lr_delay_mult_coarse"].operator float();
    opt_params_.anchor_lr_max_steps_coarse =
        settings_file["Optimization.anchor_lr_max_steps_coarse"].operator int();
    opt_params_.feature_lr_coarse =
        settings_file["Optimization.feature_lr_coarse"].operator float();
    opt_params_.opacity_lr_coarse =
        settings_file["Optimization.opacity_lr_coarse"].operator float();
    opt_params_.scaling_lr_coarse =
        settings_file["Optimization.scaling_lr_coarse"].operator float();
    opt_params_.rotation_lr_coarse =
        settings_file["Optimization.rotation_lr_coarse"].operator float();

    opt_params_.offset_lr_init_coarse =
        settings_file["Optimization.offset_lr_init_coarse"].operator float();
    opt_params_.offset_lr_final_coarse = settings_file["Optimization.offset_lr_final_coarse"].operator float();
    opt_params_.offset_lr_delay_mult_coarse =settings_file["Optimization.offset_lr_delay_mult_coarse"].operator float();
    opt_params_.offset_lr_max_steps_coarse =settings_file["Optimization.offset_lr_max_steps_coarse"].operator int();

    opt_params_.mlp_opacity_lr_init_coarse =
        settings_file["Optimization.mlp_opacity_lr_init_coarse"].operator float();
    opt_params_.mlp_opacity_lr_final_coarse = settings_file["Optimization.mlp_opacity_lr_final_coarse"].operator float();
    opt_params_.mlp_opacity_lr_delay_mult_coarse =settings_file["Optimization.mlp_opacity_lr_delay_mult_coarse"].operator float();
    opt_params_.mlp_opacity_lr_max_steps_coarse =settings_file["Optimization.mlp_opacity_lr_max_steps_coarse"].operator int();

    opt_params_.mlp_cov_lr_init_coarse =settings_file["Optimization.mlp_cov_lr_init_coarse"].operator float();
    opt_params_.mlp_cov_lr_final_coarse =settings_file["Optimization.mlp_cov_lr_final_coarse"].operator float();
    opt_params_.mlp_cov_lr_delay_mult_coarse =settings_file["Optimization.mlp_cov_lr_delay_mult_coarse"].operator float();
    opt_params_.mlp_cov_lr_max_steps_coarse =settings_file["Optimization.mlp_cov_lr_max_steps_coarse"].operator int();

    opt_params_.mlp_color_lr_init_coarse =settings_file["Optimization.mlp_color_lr_init_coarse"].operator float();
    opt_params_.mlp_color_lr_final_coarse =settings_file["Optimization.mlp_color_lr_final_coarse"].operator float();
    opt_params_.mlp_color_lr_delay_mult_coarse =settings_file["Optimization.mlp_color_lr_delay_mult_coarse"].operator float();
    opt_params_.mlp_color_lr_max_steps_coarse =settings_file["Optimization.mlp_color_lr_max_steps_coarse"].operator int();

    opt_params_.mlp_featurebank_lr_init_coarse =settings_file["Optimization.mlp_featurebank_lr_init_coarse"].operator float();
    opt_params_.mlp_featurebank_lr_final_coarse =settings_file["Optimization.mlp_featurebank_lr_final_coarse"].operator float();
    opt_params_.mlp_featurebank_lr_delay_mult_coarse =settings_file["Optimization.mlp_featurebank_lr_delay_mult_coarse"].operator float();
    opt_params_.mlp_featurebank_lr_max_steps_coarse =settings_file["Optimization.mlp_featurebank_lr_max_steps_coarse"].operator int();

    opt_params_.appearance_lr_init_coarse =settings_file["Optimization.appearance_lr_init_coarse"].operator float();
    opt_params_.appearance_lr_final_coarse =settings_file["Optimization.appearance_lr_final_coarse"].operator float();
    opt_params_.appearance_lr_delay_mult_coarse =settings_file["Optimization.appearance_lr_delay_mult_coarse"].operator float();
    opt_params_.appearance_lr_max_steps_coarse =settings_file["Optimization.appearance_lr_max_steps_coarse"].operator int();

    rendered_image_viewer_scale_ =
        settings_file["GaussianViewer.image_scale"].operator float();
    rendered_image_viewer_scale_main_ =
        settings_file["GaussianViewer.image_scale_main"].operator float();

    args_logger(model_params_, opt_params_, pipe_params_);
    std::cout << "use_coarse_anchor" <<  model_params_.use_coarse_anchor
                << "feat_dim_coarse" <<  model_params_.feat_dim_coarse
                << "n_offsets_coarse" <<  model_params_.n_offsets_coarse
                << "coarse_voxel_size" <<  model_params_.coarse_voxel_size 
                << "appearance_dim_coarse" <<  model_params_.appearance_dim_coarse
                << std::endl;

    std::cout << "use_frequency_regularization:" <<  use_frequency_regularization
                << " use_multi_resolution:" <<  use_multi_resolution
                << " scale_num:" <<  scale_num
                << " frequency_regulization_until:" <<  frequency_regulization_until 
                << " high_frequency_regularization_start:" <<  high_frequency_regularization_start
                << " lambda_frequency_high:" <<  lambda_frequency_high
                << " lambda_frequency_low:" <<  lambda_frequency_low
                << " scales:";
    scales.resize(scale_num, 0);
    for (int i = 0; i < scale_num ; i++)
    {
        scales[i] = float(1.0 / pow(2, i));
        std::cout << scales[i] << ",";
    }
    std::cout << std::endl;
}

void GaussianMapper::run()
{
    while (!isStopped()) {
        if (hasMetInitialMappingConditions()) {
            pSLAM_->getAtlas()->clearMappingOperation();

            auto pMap = pSLAM_->getAtlas()->GetCurrentMap();
            std::vector<ORB_SLAM3::KeyFrame*> vpKFs;
            std::vector<ORB_SLAM3::MapPoint*> vpMPs;
            {
                std::unique_lock<std::mutex> lock_map(pMap->mMutexMapUpdate);
                vpKFs = pMap->GetAllKeyFrames();
                vpMPs = pMap->GetAllMapPoints();
                for (const auto& pMP : vpMPs){
                    Point3D point3D;
                    auto pos = pMP->GetWorldPos();
                    point3D.xyz_(0) = pos.x();
                    point3D.xyz_(1) = pos.y();
                    point3D.xyz_(2) = pos.z();
                    auto color = pMP->GetColorRGB();
                    point3D.color_(0) = color(0);
                    point3D.color_(1) = color(1);
                    point3D.color_(2) = color(2);
                    scene_->cachePoint3D(pMP->mnId, point3D);
                }
                for (const auto& pKF : vpKFs){
                    std::shared_ptr<GaussianKeyframe> new_kf = std::make_shared<GaussianKeyframe>(pKF->mnId, getIteration());
                    new_kf->zfar_ = z_far_;
                    new_kf->znear_ = z_near_;
                    new_kf->frameID = pKF->frameID;
                    auto pose = pKF->GetPose();
                    new_kf->setPose(
                        pose.unit_quaternion().cast<double>(),
                        pose.translation().cast<double>());
                    cv::Mat imgRGB_undistorted, imgAux_undistorted;
                    try {
                        if(use_undistorted_image)
                        {  
                            Camera& camera = scene_->cameras_.at(pKF->mpCamera->GetId());
                            imgRGB_undistorted = pKF->undistortedRGB;
                            cv::Mat imgAux = pKF->imgAuxiliary;
                            if (this->sensor_type_ == RGBD)
                                camera.undistortImage(imgAux, imgAux_undistorted);
                            else
                                imgAux_undistorted = imgAux;                            

                            new_kf->setCameraParams(camera, imgRGB_undistorted.size);
                            new_kf->original_image_ =
                                tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
                            new_kf->img_filename_ = pKF->mNameFile;
                            new_kf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
                            new_kf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
                            new_kf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
                        }
                        else
                        {
                            Camera& camera = scene_->cameras_.at(pKF->mpCamera->GetId());
                            new_kf->setCameraParams(camera);

                            cv::Mat imgRGB = pKF->imgLeftRGB;
                            if (this->sensor_type_ == STEREO)
                                imgRGB_undistorted = imgRGB;
                            else
                                camera.undistortImage(imgRGB, imgRGB_undistorted);
                            cv::Mat imgAux = pKF->imgAuxiliary;
                            if (this->sensor_type_ == RGBD)
                                camera.undistortImage(imgAux, imgAux_undistorted);
                            else
                                imgAux_undistorted = imgAux;

                            new_kf->original_image_ =
                                tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
                            new_kf->img_filename_ = pKF->mNameFile;
                            new_kf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
                            new_kf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
                            new_kf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
                        }
                    }
                    catch (std::out_of_range) {
                        throw std::runtime_error("[GaussianMapper::run]KeyFrame Camera not found!");
                    }
                    new_kf->computeTransformTensors();
                    scene_->addKeyframe(new_kf, &kfid_shuffled_);

                    increaseKeyframeTimesOfUse(new_kf, newKeyframeTimesOfUse());

                    std::vector<float> pixels;
                    std::vector<float> pointsLocal;
                    pKF->GetKeypointInfo(pixels, pointsLocal);
                    new_kf->kps_pixel_ = std::move(pixels);
                    new_kf->kps_point_local_ = std::move(pointsLocal);
                    new_kf->img_undist_ = imgRGB_undistorted;
                    new_kf->img_auxiliary_undist_ = imgAux_undistorted;
                }
            }

            if (isdoingGausPyramidTraining()){
                for (auto& kfit : scene_->keyframes()) {
                    auto pkf = kfit.second;
                    if (device_type_ == torch::kCUDA) {
                        cv::cuda::GpuMat img_gpu;
                        img_gpu.upload(pkf->img_undist_);
                        pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                        for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                            cv::cuda::GpuMat img_resized;
                            cv::cuda::resize(img_gpu, img_resized,
                                            cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                            pkf->gaus_pyramid_original_image_[l] =
                                tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
                        }
                    }
                    else {
                        pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                        for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                            cv::Mat img_resized;
                            cv::resize(pkf->img_undist_, img_resized,
                                    cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                            pkf->gaus_pyramid_original_image_[l] =
                                tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
                        }
                    }
                }
            }

            {
                std::cout << "gaussians_->setApperance();"<< std::endl; 
                gaussians_->setApperance();
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
                gaussians_->createFromPcd(scene_->cached_point_cloud_, scene_->cameras_extent_);
                std::unique_lock<std::mutex> lock(mutex_settings_);
                gaussians_->trainingSetup(opt_params_);
            }

            trainForOneIteration();

            initial_mapped_ = true;
            break;
        }
        else if (pSLAM_->isShutDown()) {
            break;
        }
        else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    int SLAM_stop_iter = 0;
    while (!isStopped()) {
        if (hasMetIncrementalMappingConditions()) {
            combineMappingOperations();
            if (cull_keyframes_)
                cullKeyframes();
                
        }
        trainForOneIteration();

        if (pSLAM_->isShutDown() && !SLAM_ended_ ) {
            SLAM_ended_ = true;
            std::cout << "SLAM finished at" << getIteration() << std::endl;

            if(this->poseSaved)
            {
                std::vector<ORB_SLAM3::KeyFrame*> vpKFs = pSLAM_->getAtlas()->GetAllKeyFrames();
                std::size_t nkfs = scene_->keyframes().size();
                auto kfit = scene_->keyframes().begin();
                std::unordered_map<std::size_t, ORB_SLAM3::KeyFrame*> kfMap;
                for (const auto& pKF : vpKFs) {
                    kfMap[pKF->frameID] = pKF;
                }

                int update_poses = 0;
                for (std::size_t i = 0; i < nkfs; ++i)
                {
                    auto mnId = (*kfit).second->frameID;
                    if (this->sensor_type_ == MONOCULAR)
                    {
                        if(vTimestamps != nullptr){
                            double timeStamp = (*vTimestamps)[mnId];
                            auto found = this->poseT_.find(timeStamp);
                            if (found != this->poseT_.end())
                            {
                                auto Tw2c = found->second;
                                Eigen::Vector3d pose_trans(Tw2c[1], Tw2c[2], Tw2c[3]);
                                Eigen::Quaterniond pose_rot(Tw2c[7], Tw2c[4], Tw2c[5], Tw2c[6]);
                                Sophus::SE3d pose(pose_rot, pose_trans);
                                Sophus::SE3d Tc2w = pose.inverse();
                                auto c2w_q = Tc2w.unit_quaternion();
                                auto c2w_t = Tc2w.translation();

                                (*kfit).second->setPose(c2w_q, c2w_t);
                                (*kfit).second->computeTransformTensors();

                                update_poses++;
                            }
                        }
                        else{
                            auto found = this->pose_.find(mnId);
                            if (found != this->pose_.end())
                            {
                                auto Tw2c = found->second;
                                Eigen::Vector3d pose_trans(Tw2c[1], Tw2c[2], Tw2c[3]);
                                Eigen::Quaterniond pose_rot(Tw2c[7], Tw2c[4], Tw2c[5], Tw2c[6]);
                                Sophus::SE3d pose(pose_rot, pose_trans);
                                Sophus::SE3d Tc2w = pose.inverse();
                                auto c2w_q = Tc2w.unit_quaternion();
                                auto c2w_t = Tc2w.translation();

                                (*kfit).second->setPose(c2w_q, c2w_t);
                                (*kfit).second->computeTransformTensors();

                                update_poses++;
                            }
                        }
                    }

                    if (this->sensor_type_ != MONOCULAR){
                    auto found = this->pose_.find(mnId);
                    if (found != this->pose_.end())
                    {
                        auto Tw2c = found->second;
                        Eigen::Vector3d pose_trans(Tw2c[1], Tw2c[2], Tw2c[3]);
                        Eigen::Quaterniond pose_rot(Tw2c[7], Tw2c[4], Tw2c[5], Tw2c[6]);
                        Sophus::SE3d pose(pose_rot, pose_trans);
                        Sophus::SE3d Tc2w = pose.inverse();
                        auto c2w_q = Tc2w.unit_quaternion();
                        auto c2w_t = Tc2w.translation();

                        (*kfit).second->setPose(c2w_q, c2w_t);
                        (*kfit).second->computeTransformTensors();

                        update_poses++;
                    }
                    }

                    ++kfit;
                }
                std::cout << "update pose: " << update_poses << std::endl;
            }
            if(light_mode)
                break;
        }

        if (getIteration() >= opt_params_.iterations_)
            break;
    }

    if(light_mode){
        std::cout << "[light mode]: Tail gaussian optimization " << std::endl;
        int densify_interval = densifyInterval();
        int n_delay_iters = densify_interval * 0.8;
        while (getIteration() - SLAM_stop_iter <= n_delay_iters || getIteration() % densify_interval <= n_delay_iters || isKeepingTraining()) {
            trainForOneIteration();
            densify_interval = densifyInterval();
            n_delay_iters = densify_interval * 0.8;
        }
    }

    std::cout << std::endl << "anchors:" <<  gaussians_->get_anchor().size(0) << std::endl;
    std::ofstream(result_dir_ / "gaussians_num.txt") << gaussians_->get_anchor().size(0);
    renderAndRecordAllKeyframes("_shutdown");
    savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");
    writeKeyframeUsedTimes(result_dir_ / "used_times", "final");
    
    std::string result = result_dir_ / (std::to_string(getIteration()) + "_shutdown");
    std::cout << ":" << std::endl;
    std::cout << result << std::endl;

    std::vector<ORB_SLAM3::KeyFrame*> allFs = pSLAM_->GetAllFrames();
    std::cout << " allFs" << allFs.size() << std::endl;
    
    signalStop();
}

void GaussianMapper::trainColmap()
{
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
        gaussians_->createFromPcd(scene_->cached_point_cloud_, scene_->cameras_extent_);
        std::unique_lock<std::mutex> lock(mutex_settings_);
        gaussians_->trainingSetup(opt_params_);
        this->initial_mapped_ = true;
    }

    while (!isStopped()) {
        trainForOneIteration();

        if (getIteration() >= opt_params_.iterations_)
            break;
    }

    renderAndRecordAllKeyframes("_shutdown");
    savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");
    writeKeyframeUsedTimes(result_dir_ / "used_times", "final");

    signalStop();
}


void GaussianMapper::trainForOneIteration()
{
    increaseIteration(1);
    auto iter_start_timing = std::chrono::steady_clock::now();
    std::shared_ptr<GaussianKeyframe> viewpoint_cam = useOneRandomSlidingWindowKeyframe();
    
    if (!viewpoint_cam)
    {
        increaseIteration(-1);
        return;
    }

    writeKeyframeUsedTimes(result_dir_ / "used_times");

    int training_level = num_gaus_pyramid_sub_levels_;
    int image_height, image_width;
    torch::Tensor gt_image, mask;
    if (isdoingGausPyramidTraining()){
        training_level = viewpoint_cam->getCurrentGausPyramidLevel();
        if (training_level == num_gaus_pyramid_sub_levels_) {
            image_height = viewpoint_cam->image_height_;
            image_width = viewpoint_cam->image_width_;
            gt_image = viewpoint_cam->original_image_.cuda();
            mask = undistort_mask_[viewpoint_cam->camera_id_];
        }
        else {
            image_height = viewpoint_cam->gaus_pyramid_height_[training_level];
            image_width = viewpoint_cam->gaus_pyramid_width_[training_level];
            gt_image = viewpoint_cam->gaus_pyramid_original_image_[training_level].cuda();
            try {
                mask = scene_->cameras_.at(viewpoint_cam->camera_id_).gaus_pyramid_undistort_mask_[training_level];
            } catch (const std::out_of_range& e) {
                std::cerr << "Out of range error: " << e.what() << std::endl;
            }

        }
    }

    std::unique_lock<std::mutex> lock_render(mutex_render_);

    gaussians_->updateLearningRate(getIteration());


    gaussians_->setAnchorFeatureLearningRate(anchorFeatureLearningRate());
    gaussians_->setOpacityLearningRate(opacityLearningRate());
    gaussians_->setScalingLearningRate(scalingLearningRate());
    gaussians_->setRotationLearningRate(rotationLearningRate());

    auto prefilter_start_timing = std::chrono::steady_clock::now();
    if (!isdoingGausPyramidTraining()){
        image_height = viewpoint_cam->image_height_;
        image_width = viewpoint_cam->image_width_;
    }
    auto voxel_visible_mask = GaussianRenderer::prefilter_voxel(
        viewpoint_cam,
        image_height,
        image_width,
        gaussians_,
        pipe_params_,
        background_,
        override_color_
    );
    auto prefilter_end_timing = std::chrono::steady_clock::now();
    auto render_start_timing = std::chrono::steady_clock::now();
    bool retain_grad = (getIteration() < opt_params_.update_until && getIteration() >= 0);
    auto render_pkg = GaussianRenderer::render(
        viewpoint_cam,
        image_height,
        image_width,
        gaussians_,
        pipe_params_,
        background_,
        override_color_,
        voxel_visible_mask,
        retain_grad
    );
    auto render_end_timing = std::chrono::steady_clock::now();
    auto rendered_image = std::get<0>(render_pkg);
    auto viewspace_point_tensor = std::get<1>(render_pkg);
    auto visibility_filter = std::get<2>(render_pkg);
    auto radii = std::get<3>(render_pkg);
    auto offset_selection_mask = std::get<4>(render_pkg);
    auto opacity = std::get<5>(render_pkg);
    auto scaling = std::get<6>(render_pkg);
    auto is_training = std::get<7>(render_pkg);

    mask = undistort_mask_[viewpoint_cam->camera_id_];
    torch::Tensor masked_image = rendered_image;

    auto backward_start_timing = std::chrono::steady_clock::now();
    if (!isdoingGausPyramidTraining()){
        gt_image = viewpoint_cam->original_image_.cuda();
    }

    torch::Tensor mask_rgb = (gt_image != 0.0f).any(-1);
    mask_rgb = mask_rgb.to(torch::kFloat32).unsqueeze(-1);

    masked_image = masked_image * mask_rgb;
    rendered_image = rendered_image * mask_rgb;
    gt_image = gt_image * mask_rgb;

    auto Ll1 = loss_utils::l1_loss(rendered_image, gt_image);
    float lambda_dssim = lambdaDssim();
    auto scaling_reg = scaling.prod(1).mean();
    auto loss = (1.0 - lambda_dssim) * Ll1 +
                lambda_dssim * (1.0 - loss_utils::ssim(masked_image, gt_image, device_type_)) + 0.01 * scaling_reg; 

    if(this->use_frequency_regularization)
    {
        if(getIteration() < frequency_regulization_until)
        {
            auto freq_loss_low = loss_utils::low_freq_loss(rendered_image, gt_image);
            loss = loss + lambda_frequency_low * freq_loss_low;
        }

        if(getIteration() < frequency_regulization_until && getIteration() > high_frequency_regularization_start)
        {
            if(use_multi_resolution)
                loss = loss + lambda_frequency_high * loss_utils::multi_scale_loss(rendered_image, gt_image, scales);
            else
                loss = loss + lambda_frequency_high * loss_utils::high_frequency_loss(rendered_image, gt_image);
        }        
    }

    auto freq_loss = loss_utils::high_frequency_loss(rendered_image, gt_image);
    auto freq_loss_low = loss_utils::low_freq_loss(rendered_image, gt_image);

    loss.backward();
    auto backward_end_timing = std::chrono::steady_clock::now();

    torch::cuda::synchronize();
    auto synchronize_end_timing = std::chrono::steady_clock::now();

    {
        torch::NoGradGuard no_grad;
        ema_loss_for_log_ = 0.4f * loss.item().toFloat() + 0.6 * ema_loss_for_log_;

        auto densification_start_timing = std::chrono::steady_clock::now();
        if (getIteration() < opt_params_.update_until && getIteration() >opt_params_.start_stat)
        {
            gaussians_->training_statis(viewspace_point_tensor, opacity,
                                        visibility_filter, offset_selection_mask, voxel_visible_mask);
        
            if(getIteration()>opt_params_.update_from && getIteration()%opt_params_.update_interval==0)
                gaussians_->adjust_anchor(
                    opt_params_.update_interval, 
                    opt_params_.success_threshold, 
                    opt_params_.densify_grad_threshold,
                    opt_params_.min_opacity);
        }
        else if(getIteration() == opt_params_.update_until)
        {
            std::cout << std::endl
                      << "densification ended: anchors:" << gaussians_->get_anchor().size(0) << std::endl;
        }
        auto densification_end_timing = std::chrono::steady_clock::now();

        auto iter_end_timing = std::chrono::steady_clock::now();
        auto iter_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        iter_end_timing - iter_start_timing).count();

        elapsed_time_from_start = elapsed_time_from_start + iter_time;
        if (getIteration() % 10 == 0)
        {
            int bar_width = 10;
            float progress = (float)getIteration() / opt_params_.iterations_;

            float estimated_total_time = elapsed_time_from_start / progress;
            float remaining_seconds = estimated_total_time - elapsed_time_from_start;

            int elapsed_minutes = static_cast<int>(elapsed_time_from_start / 1000.0) / 60;
            int elapsed_seconds_display = static_cast<int>(elapsed_time_from_start / 1000.0) % 60;

            int remaining_minutes = static_cast<int>(remaining_seconds / 1000.0) / 60;
            int remaining_seconds_display = static_cast<int>(remaining_seconds / 1000.0) % 60;

            std::cout << "\033[1;34mTrain:\033[0m[";
            int pos = bar_width * progress;
            for (int i = 0; i < bar_width; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "]" << int(progress * 100.0) << "% [" << getIteration() << "/" << opt_params_.iterations_
                      << ", L=" << std::fixed << std::setprecision(4) << ema_loss_for_log_
                      << ", " << std::setfill('0') << std::setw(2) << elapsed_minutes << ":" << std::setw(2) << elapsed_seconds_display
                      << "<" << std::setfill('0') << std::setw(2) << remaining_minutes << ":" << std::setw(2) << remaining_seconds_display
                      << " " << std::setfill('0') << std::setw(2) << iter_time << "ms"
                      << " FL=" <<  freq_loss.item().toFloat() 
                      <<"/" << freq_loss_low.item().toFloat() << "]"
                      << "\r";

            if (getIteration() < opt_params_.iterations_)
                std::cout.flush();
        }
        if ((all_keyframes_record_interval_ && getIteration() % all_keyframes_record_interval_ == 0)
            )
        {
            renderAndRecordAllKeyframes();
        }

        if (loop_closure_iteration_)
            loop_closure_iteration_ = false;

        if (getIteration() < opt_params_.iterations_) {
            gaussians_->optimizer_->step();
            gaussians_->optimizer_->zero_grad(true);
        }
    }
}

bool GaussianMapper::isStopped()
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    return this->stopped_;
}

void GaussianMapper::signalStop(const bool going_to_stop)
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    this->stopped_ = going_to_stop;
}

bool GaussianMapper::hasMetInitialMappingConditions()
{
    if (!pSLAM_->isShutDown() &&
        pSLAM_->GetNumKeyframes() >= min_num_initial_map_kfs_ &&
        pSLAM_->getAtlas()->hasMappingOperation())
        return true;

    bool conditions_met = false;
    return conditions_met;
}

bool GaussianMapper::hasMetIncrementalMappingConditions()
{
    if (!pSLAM_->isShutDown() &&
        pSLAM_->getAtlas()->hasMappingOperation())
        return true;

    bool conditions_met = false;
    return conditions_met;
}
void GaussianMapper::combineMappingOperations()
{
    while (pSLAM_->getAtlas()->hasMappingOperation()) {
        ORB_SLAM3::MappingOperation opr =
            pSLAM_->getAtlas()->getAndPopMappingOperation();

        switch (opr.meOperationType)
        {

        case ORB_SLAM3::MappingOperation::OprType::LocalMappingBA:
        {
            auto& associated_kfs = opr.associatedKeyFrames();

            for (auto& kf : associated_kfs) {
                auto kfid = std::get<0>(kf);
                std::shared_ptr<GaussianKeyframe> pkf = scene_->getKeyframe(kfid);
                if (pkf) {
                    auto& pose = std::get<2>(kf);
                    pkf->setPose(
                        pose.unit_quaternion().cast<double>(),
                        pose.translation().cast<double>());
                    pkf->computeTransformTensors();

                    increaseKeyframeTimesOfUse(pkf, local_BA_increased_times_of_use_);
                }
                else {
                    handleNewKeyframe(kf);

                    if (this->sensor_type_ == MONOCULAR)
                    {
                        std::vector<ORB_SLAM3::KeyFrame*> vpKFs = pSLAM_->getAtlas()->GetAllKeyFrames();
                        std::size_t nkfs = scene_->keyframes().size();
                        auto kfit = scene_->keyframes().begin();
                        bool frameFinded = false;
                        std::unordered_map<std::size_t, ORB_SLAM3::KeyFrame*> kfMap;
                        for (const auto& pKF : vpKFs) {
                            kfMap[pKF->frameID] = pKF;
                        }

                        for (std::size_t i = 0; i < nkfs; ++i)
                        {
                            auto mnId = (*kfit).second->frameID;
                            auto found = kfMap.find(mnId);
                            if (found != kfMap.end()) {
                                auto pose = found->second->GetPose();
                                (*kfit).second->setPose(
                                    pose.unit_quaternion().cast<double>(),
                                    pose.translation().cast<double>());
                                (*kfit).second->computeTransformTensors();
                            }
                            ++kfit;
                        }
                    }
                    else{
                        std::vector<ORB_SLAM3::KeyFrame*> vpKFs = pSLAM_->getAtlas()->GetAllKeyFrames();
                        std::size_t nkfs = scene_->keyframes().size();
                        auto kfit = scene_->keyframes().begin();
                        bool frameFinded = false;
                        std::unordered_map<std::size_t, ORB_SLAM3::KeyFrame*> kfMap;
                        for (const auto& pKF : vpKFs) {
                            kfMap[pKF->frameID] = pKF;
                        }

                        for (std::size_t i = 0; i < nkfs; ++i)
                        {
                            auto mnId = (*kfit).second->frameID;
                            auto found = kfMap.find(mnId);
                            if (found != kfMap.end()) {
                                auto pose = found->second->GetPose();
                                (*kfit).second->setPose(
                                    pose.unit_quaternion().cast<double>(),
                                    pose.translation().cast<double>());
                                (*kfit).second->computeTransformTensors();
                            }
                            else;
                            ++kfit;
                        }
                    }
                }
            }

            auto& associated_points = opr.associatedMapPoints();
            auto& points = std::get<0>(associated_points);
            auto& colors = std::get<1>(associated_points);

            if (initial_mapped_ && points.size() >= 30) {
                torch::NoGradGuard no_grad;
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                gaussians_->increasePcd(points, colors, getIteration());
            }
        }
        break;

        case ORB_SLAM3::MappingOperation::OprType::LoopClosingBA:
        {
            loop_closure_iteration_ = true;
        }
        break;

        case ORB_SLAM3::MappingOperation::OprType::ScaleRefinement:
        {
            std::cout << "[Gaussian Mapper]Scale refinement Detected. Transforming all kfs and points..."
                      << std::endl;

            float s = opr.mfScale;
            Sophus::SE3f& T = opr.mT;
            if (initial_mapped_) {
                {
                    std::unique_lock<std::mutex> lock_render(mutex_render_);
                    gaussians_->applyScaledTransformation(s, T);
                }
                scene_->applyScaledTransformation(s, T);
            }
            else { 
                for (auto& pt : scene_->cached_point_cloud_) {
                    auto& pt_xyz = pt.second.xyz_;
                    pt_xyz *= s;
                    pt_xyz = T.cast<double>() * pt_xyz;
                }

                for (auto& kfit : scene_->keyframes()) {
                    std::shared_ptr<GaussianKeyframe> pkf = kfit.second;
                    Sophus::SE3f Twc = pkf->getPosef().inverse();
                    Twc.translation() *= s;
                    Sophus::SE3f Tyc = T * Twc;
                    Sophus::SE3f Tcy = Tyc.inverse();
                    pkf->setPose(Tcy.unit_quaternion().cast<double>(), Tcy.translation().cast<double>());
                    pkf->computeTransformTensors();
                }
            }
        }
        break;

        default:
        {
            throw std::runtime_error("MappingOperation type not supported!");
        }
        break;
        }
    }
}

void GaussianMapper::handleNewKeyframe(
    std::tuple<unsigned long,
               unsigned long,
               Sophus::SE3f,
               cv::Mat,
               bool,
               cv::Mat,
               std::string,
               unsigned long,
               double,
               cv::Mat> &kf)
{
    std::shared_ptr<GaussianKeyframe> pkf =
        std::make_shared<GaussianKeyframe>(std::get<0>(kf), getIteration());
    pkf->zfar_ = z_far_;
    pkf->znear_ = z_near_;
    pkf->frameID = std::get<7>(kf);
    pkf->TimeStamp = std::get<8>(kf);
    auto& pose = std::get<2>(kf);
    pkf->setPose(
        pose.unit_quaternion().cast<double>(),
        pose.translation().cast<double>());
    cv::Mat imgRGB_undistorted, imgAux_undistorted, imgRGB_undistorted_edge, edge_mask;

    try {
        if(use_undistorted_image)
        {
            Camera& camera = scene_->cameras_.at(std::get<1>(kf));
            imgRGB_undistorted = std::get<9>(kf);

            cv::Mat imgAux = std::get<5>(kf);
            if (this->sensor_type_ == RGBD)
                camera.undistortImage(imgAux, imgAux_undistorted);
            else
                imgAux_undistorted = imgAux; 
            pkf->setCameraParams(camera, imgRGB_undistorted.size);
            pkf->original_image_ =
                tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
            pkf->img_filename_ = std::get<6>(kf);
            pkf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
            pkf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
            pkf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
        }
        else
        {
        Camera& camera = scene_->cameras_.at(std::get<1>(kf));
        pkf->setCameraParams(camera);

        cv::Mat imgRGB = std::get<3>(kf);
        if (this->sensor_type_ == STEREO)
            imgRGB_undistorted = imgRGB;
        else
            camera.undistortImage(imgRGB, imgRGB_undistorted);
        cv::Mat imgAux = std::get<5>(kf);
        if (this->sensor_type_ == RGBD)
            camera.undistortImage(imgAux, imgAux_undistorted);
        else
            imgAux_undistorted = imgAux;

        pkf->original_image_ =
            tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
        pkf->img_filename_ = std::get<8>(kf);
        pkf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
        pkf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
        pkf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
        }
    }
    catch (std::out_of_range) {
        throw std::runtime_error("[GaussianMapper::combineMappingOperations]KeyFrame Camera not found!");
    }
    pkf->computeTransformTensors();
    scene_->addKeyframe(pkf, &kfid_shuffled_);

    increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());

    pkf->img_undist_ = imgRGB_undistorted;
    pkf->img_auxiliary_undist_ = imgAux_undistorted;

    if(isdoingGausPyramidTraining()){
        if (device_type_ == torch::kCUDA) {
            cv::cuda::GpuMat img_gpu;
            img_gpu.upload(pkf->img_undist_);
            pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                cv::cuda::GpuMat img_resized;
                cv::cuda::resize(img_gpu, img_resized,
                                    cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                pkf->gaus_pyramid_original_image_[l] =
                    tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
            }
        }
        else {
            pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                cv::Mat img_resized;
                cv::resize(pkf->img_undist_, img_resized,
                            cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                pkf->gaus_pyramid_original_image_[l] =
                    tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
            }
        }
    }
}

void GaussianMapper::handleNewKeyframe(
    std::tuple<unsigned long ,
               unsigned long ,
               Sophus::SE3f ,
               cv::Mat ,
               bool ,
               cv::Mat ,
               std::vector<float>,
               std::vector<float>,
               std::string,
               float ,
               float ,
               float ,
               float ,
               unsigned long,
               double,
               cv::Mat> &kf)
{
    std::shared_ptr<GaussianKeyframe> pkf =
        std::make_shared<GaussianKeyframe>(std::get<0>(kf), getIteration());
    pkf->zfar_ = z_far_;
    pkf->znear_ = z_near_;
    pkf->frameID = std::get<13>(kf);
    pkf->TimeStamp = std::get<14>(kf);
    auto& pose = std::get<2>(kf);
    pkf->setPose(
        pose.unit_quaternion().cast<double>(),
        pose.translation().cast<double>());
    cv::Mat imgRGB_undistorted, imgAux_undistorted, imgRGB_undistorted_edge, edge_mask;
    try {
        if(use_undistorted_image)
        {
            
            Camera& camera = scene_->cameras_.at(std::get<1>(kf));
            imgRGB_undistorted = std::get<15>(kf);
            cv::Mat imgAux = std::get<5>(kf);
            if (this->sensor_type_ == RGBD)
                camera.undistortImage(imgAux, imgAux_undistorted);
            else
                imgAux_undistorted = imgAux; 
            pkf->setCameraParams(camera, imgRGB_undistorted.size);
            pkf->original_image_ =
                tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
            pkf->img_filename_ = std::get<8>(kf);
            pkf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
            pkf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
            pkf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;  
        }
        else
        {
        Camera& camera = scene_->cameras_.at(std::get<1>(kf));
        pkf->setCameraParams(camera);

        cv::Mat imgRGB = std::get<3>(kf);
        if (this->sensor_type_ == STEREO)
            imgRGB_undistorted = imgRGB;
        else
            camera.undistortImage(imgRGB, imgRGB_undistorted);
        cv::Mat imgAux = std::get<5>(kf);
        if (this->sensor_type_ == RGBD)
            camera.undistortImage(imgAux, imgAux_undistorted);
        else
            imgAux_undistorted = imgAux;

        pkf->original_image_ =
            tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
        pkf->img_filename_ = std::get<8>(kf);
        pkf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
        pkf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
        pkf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
        }
    }
    catch (std::out_of_range) {
        throw std::runtime_error("[GaussianMapper::combineMappingOperations]KeyFrame Camera not found!");
    }
    pkf->computeTransformTensors();
    scene_->addKeyframe(pkf, &kfid_shuffled_);

    increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());

    pkf->img_undist_ = imgRGB_undistorted;
    pkf->img_auxiliary_undist_ = imgAux_undistorted;
    pkf->kps_pixel_ = std::move(std::get<6>(kf));
    pkf->kps_point_local_ = std::move(std::get<7>(kf));
    if (isdoingInactiveGeoDensify())
        increasePcdByKeyframeInactiveGeoDensify(pkf);

    if (device_type_ == torch::kCUDA) {
        cv::cuda::GpuMat img_gpu;
        img_gpu.upload(pkf->img_undist_);
        pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
        for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            cv::cuda::GpuMat img_resized;
            cv::cuda::resize(img_gpu, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
            pkf->gaus_pyramid_original_image_[l] =
                tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
        }
    }
    else {
        pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
        for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            cv::Mat img_resized;
            cv::resize(pkf->img_undist_, img_resized,
                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
            pkf->gaus_pyramid_original_image_[l] =
                tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
        }
    }
}

void GaussianMapper::updateAllPoses(
    std::map<std::size_t, std::shared_ptr<GaussianKeyframe>> &keyframes,
    const std::vector<ORB_SLAM3::KeyFrame *> &vpKFs)
{
    std::unordered_map<std::size_t, ORB_SLAM3::KeyFrame*> kfMap;
    for (const auto& pKF : vpKFs) {
        kfMap[pKF->frameID] = pKF;
    }

    for (auto& kfPair : keyframes) {
        auto mnId = kfPair.second->frameID;
        auto found = kfMap.find(mnId);
        if (found != kfMap.end()) {
            auto pose = found->second->GetPose();
            kfPair.second->setPose(
                pose.unit_quaternion().cast<double>(),
                pose.translation().cast<double>());
            kfPair.second->computeTransformTensors();
        }
    }
}

void GaussianMapper::generateKfidRandomShuffle()
{
    if (scene_->keyframes().empty())
        return;

    std::size_t nkfs = scene_->keyframes().size();
    kfid_shuffle_.resize(nkfs);
    std::iota(kfid_shuffle_.begin(), kfid_shuffle_.end(), 0);
    std::mt19937 g(rd_());
    std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

    kfid_shuffled_ = true;
}

std::shared_ptr<GaussianKeyframe>
GaussianMapper::useOneRandomSlidingWindowKeyframe()
{
    if (scene_->keyframes().empty())
        return nullptr;

    if (!kfid_shuffled_)
        generateKfidRandomShuffle();

    std::shared_ptr<GaussianKeyframe> viewpoint_cam = nullptr;
    int random_cam_idx;

    if (kfid_shuffled_) {
        int start_shuffle_idx = kfid_shuffle_idx_;
        do {
            ++kfid_shuffle_idx_;
            if (kfid_shuffle_idx_ >= kfid_shuffle_.size())
                kfid_shuffle_idx_ = 0;
            if (kfid_shuffle_idx_ == start_shuffle_idx)
                for (auto& kfit : scene_->keyframes())
                    increaseKeyframeTimesOfUse(kfit.second, 1);
            random_cam_idx = kfid_shuffle_[kfid_shuffle_idx_];
            auto random_cam_it = scene_->keyframes().begin();
            for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
                ++random_cam_it;
            viewpoint_cam = (*random_cam_it).second;
        } while (viewpoint_cam->remaining_times_of_use_ <= 0);
    }
    auto viewpoint_fid = viewpoint_cam->fid_;
    if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
        kfs_used_times_[viewpoint_fid] = 1;
    else
        ++kfs_used_times_[viewpoint_fid];
    
    --(viewpoint_cam->remaining_times_of_use_);
    return viewpoint_cam;
}

std::shared_ptr<GaussianKeyframe>
GaussianMapper::useOneRandomKeyframe()
{
    if (scene_->keyframes().empty())
        return nullptr;

    int nkfs = static_cast<int>(scene_->keyframes().size());
    int random_cam_idx = std::rand() / ((RAND_MAX + 1u) / nkfs);
    auto random_cam_it = scene_->keyframes().begin();
    for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
        ++random_cam_it;
    std::shared_ptr<GaussianKeyframe> viewpoint_cam = (*random_cam_it).second;

    auto viewpoint_fid = viewpoint_cam->fid_;
    if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
        kfs_used_times_[viewpoint_fid] = 1;
    else
        ++kfs_used_times_[viewpoint_fid];

    return viewpoint_cam;
}

void GaussianMapper::increaseKeyframeTimesOfUse(
    std::shared_ptr<GaussianKeyframe> pkf,
    int times)
{
    pkf->remaining_times_of_use_ += times;
}

void GaussianMapper::cullKeyframes()
{
    std::unordered_set<unsigned long> kfids =
        pSLAM_->getAtlas()->GetCurrentKeyFrameIds();
    std::vector<unsigned long> kfids_to_erase;
    std::size_t nkfs = scene_->keyframes().size();
    kfids_to_erase.reserve(nkfs);
    for (auto& kfit : scene_->keyframes()) {
        unsigned long kfid = kfit.first;
        if (kfids.find(kfid) == kfids.end()) {
            kfids_to_erase.emplace_back(kfid);
        }
    }

    for (auto& kfid : kfids_to_erase) {
        scene_->keyframes().erase(kfid);
    }
}
void GaussianMapper::increasePcdByKeyframeInactiveGeoDensify(
    std::shared_ptr<GaussianKeyframe> pkf)
{
    torch::NoGradGuard no_grad;

    Sophus::SE3f Twc = pkf->getPosef().inverse();

    switch (this->sensor_type_)
    {
    case MONOCULAR:
    {
        assert(pkf->kps_pixel_.size() % 2 == 0);
        int N = pkf->kps_pixel_.size() / 2;
        torch::Tensor kps_pixel_tensor = torch::from_blob(
            pkf->kps_pixel_.data(), {N, 2},
            torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);
        torch::Tensor kps_point_local_tensor = torch::from_blob(
            pkf->kps_point_local_.data(), {N, 3},
            torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);
        torch::Tensor kps_has3D_tensor = torch::where(
            kps_point_local_tensor.index({torch::indexing::Slice(), 2}) > 0.0f, true, false);

        cv::cuda::GpuMat rgb_gpu;
        rgb_gpu.upload(pkf->img_undist_);
        torch::Tensor colors = tensor_utils::cvGpuMat2TorchTensor_Float32(rgb_gpu);
        colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();

        auto result =
            monocularPinholeInactiveGeoDensifyBySearchingNeighborhoodKeypoints(
                kps_pixel_tensor, kps_has3D_tensor, kps_point_local_tensor, colors,
                monocular_inactive_geo_densify_max_pixel_dist_, pkf->intr_, pkf->image_width_);
        torch::Tensor& points3D_valid = std::get<0>(result);
        torch::Tensor& colors_valid = std::get<1>(result);
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        }
        else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, 0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid}, 0);
        }
    }
    break;
    case STEREO:
    {
        cv::cuda::GpuMat rgb_left_gpu, rgb_right_gpu;
        cv::cuda::GpuMat gray_left_gpu, gray_right_gpu;

        rgb_left_gpu.upload(pkf->img_undist_);
        rgb_right_gpu.upload(pkf->img_auxiliary_undist_);

        cv::cuda::cvtColor(rgb_left_gpu, gray_left_gpu, cv::COLOR_RGB2GRAY);
        cv::cuda::cvtColor(rgb_right_gpu, gray_right_gpu, cv::COLOR_RGB2GRAY);

        gray_left_gpu.convertTo(gray_left_gpu, CV_8UC1, 255.0);
        gray_right_gpu.convertTo(gray_right_gpu, CV_8UC1, 255.0);

        cv::cuda::GpuMat cv_disp;
        stereo_cv_sgm_->compute(gray_left_gpu, gray_right_gpu, cv_disp);
        cv_disp.convertTo(cv_disp, CV_32F, 1.0 / 16.0);

        cv::cuda::GpuMat cv_points3D;
        cv::cuda::reprojectImageTo3D(cv_disp, cv_points3D, stereo_Q_, 3);

        torch::Tensor disp = tensor_utils::cvGpuMat2TorchTensor_Float32(cv_disp);
        disp = disp.flatten(0, 1).contiguous();
        torch::Tensor points3D = tensor_utils::cvGpuMat2TorchTensor_Float32(cv_points3D);
        points3D = points3D.permute({1, 2, 0}).flatten(0, 1).contiguous();
        torch::Tensor colors = tensor_utils::cvGpuMat2TorchTensor_Float32(rgb_left_gpu);
        colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();
    
        torch::Tensor point_valid_flags = torch::full(
            {disp.size(0)}, false, torch::TensorOptions().dtype(torch::kBool).device(device_type_));
        int nkps_twice = pkf->kps_pixel_.size();
        int width = pkf->image_width_;
        for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
            int idx = static_cast<int>(pkf->kps_pixel_[kpidx]) + static_cast<int>(pkf->kps_pixel_[kpidx + 1]) * width;
            point_valid_flags[idx] = true;
        }
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(disp > static_cast<float>(stereo_cv_sgm_->getMinDisparity()), true, false));
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(disp < static_cast<float>(stereo_cv_sgm_->getNumDisparities()), true, false));

        torch::Tensor points3D_valid = points3D.index({point_valid_flags});
        torch::Tensor colors_valid = colors.index({point_valid_flags});

        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);

        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        }
        else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, 0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid}, 0);
        }
    }
    break;
    case RGBD:
    {
        cv::cuda::GpuMat img_rgb_gpu, img_depth_gpu;
        img_rgb_gpu.upload(pkf->img_undist_);
        img_depth_gpu.upload(pkf->img_auxiliary_undist_);

        torch::Tensor rgb = tensor_utils::cvGpuMat2TorchTensor_Float32(img_rgb_gpu);
        rgb = rgb.permute({1, 2, 0}).flatten(0, 1).contiguous();
        torch::Tensor depth = tensor_utils::cvGpuMat2TorchTensor_Float32(img_depth_gpu);
        depth = depth.flatten(0, 1).contiguous();

        torch::Tensor point_valid_flags = torch::full(
            {depth.size(0)}, false, torch::TensorOptions().dtype(torch::kBool).device(device_type_));
        int nkps_twice = pkf->kps_pixel_.size();
        int width = pkf->image_width_;
        for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
            int idx = static_cast<int>(pkf->kps_pixel_[kpidx]) + static_cast<int>(pkf->kps_pixel_[kpidx + 1]) * width;
            point_valid_flags[idx] = true;
        }
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(depth > RGBD_min_depth_, true, false));
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(depth < RGBD_max_depth_, true, false));
        torch::Tensor points3D_valid;
        Camera& camera = scene_->cameras_.at(pkf->camera_id_);
        switch (camera.model_id_)
        {
        case Camera::PINHOLE:
        {
            points3D_valid = reprojectDepthPinhole(
                depth, point_valid_flags, pkf->intr_, pkf->image_width_);
        }
        break;
        case Camera::FISHEYE:
        {
            throw std::runtime_error("[Gaussian Mapper]Fisheye cameras are not supported currently!");
        }
        break;
        default:
        {
            throw std::runtime_error("[Gaussian Mapper]Invalid camera model!");
        }
        break;
        }
        points3D_valid = points3D_valid.index({point_valid_flags});
        torch::Tensor colors_valid = points3D_valid;

        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);

        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        }
        else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, 0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid}, 0);
        }
    }
    break;
    default:
    {
        throw std::runtime_error("[Gaussian Mapper]Unsupported sensor type!");
    }
    break;
    }

    pkf->done_inactive_geo_densify_ = true;
    ++depth_cached_;

    if (depth_cached_ >= max_depth_cached_) {
        depth_cached_ = 0;
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        gaussians_->increasePcd(depth_cache_points_, depth_cache_colors_, getIteration());
    }
}

void GaussianMapper::recordKeyframeRendered(
        torch::Tensor &rendered,
        torch::Tensor &ground_truth,
        torch::Tensor &image_points,
        torch::Tensor &image_radii,
        unsigned long kfid,
        std::filesystem::path result_img_dir,
        std::filesystem::path result_gt_dir,
        std::filesystem::path result_loss_dir,
        std::filesystem::path image_points_dir,
        std::filesystem::path image_radii_dir,
        std::string name_suffix)
{
    if (record_rendered_image_) {
        auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered);
        cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
        image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_img_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + ".jpg"), image_cv);
    }

    if (record_ground_truth_image_) {
        auto gt_image_cv = tensor_utils::torchTensor2CvMat_Float32(ground_truth);
        cv::cvtColor(gt_image_cv, gt_image_cv, CV_RGB2BGR);
        gt_image_cv.convertTo(gt_image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_gt_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + ".jpg"), gt_image_cv);
    }

    if (record_loss_image_) {
        torch::Tensor loss_tensor = torch::abs(rendered - ground_truth);
        auto loss_image_cv = tensor_utils::torchTensor2CvMat_Float32(loss_tensor);
        cv::cvtColor(loss_image_cv, loss_image_cv, CV_RGB2BGR);
        loss_image_cv.convertTo(loss_image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_loss_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + "_loss.jpg"), loss_image_cv);
    }
}

void GaussianMapper::renderAndRecordKeyframe(
    std::shared_ptr<GaussianKeyframe> pkf,
    float &dssim,
    float &psnr,
    float &psnr_gs,
    double &render_time,
    std::filesystem::path result_img_dir,
    std::filesystem::path result_gt_dir,
    std::filesystem::path result_loss_dir,
    std::filesystem::path image_points_dir,
    std::filesystem::path image_radii_dir,
    std::string name_suffix)
{
    auto voxel_visible_mask = GaussianRenderer::prefilter_voxel(
        pkf,
        pkf->image_height_,
        pkf->image_width_,
        gaussians_,
        pipe_params_,
        background_,
        override_color_
    );
    auto start_timing = std::chrono::steady_clock::now();
    bool retain_grad = (getIteration() < opt_params_.update_until && getIteration() >= 0);
    auto render_pkg = GaussianRenderer::render(
        pkf,
        pkf->image_height_,
        pkf->image_width_,
        gaussians_,
        pipe_params_,
        background_,
        override_color_,
        voxel_visible_mask,
        retain_grad);
    auto rendered_image = std::get<0>(render_pkg);
    torch::Tensor masked_image = rendered_image;
    torch::cuda::synchronize();
    auto end_timing = std::chrono::steady_clock::now();
    auto render_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_timing - start_timing).count();
    render_time = 1e-6 * render_time_ns;
    auto gt_image = pkf->original_image_;    

    torch::Tensor mask_rgb = (gt_image != 0.0f).any(-1);       
    mask_rgb = mask_rgb.to(torch::kFloat32).unsqueeze(-1);        

    masked_image = rendered_image * mask_rgb;
    gt_image = gt_image * mask_rgb;
    dssim = loss_utils::ssim(masked_image, gt_image, device_type_).item().toFloat();
    psnr = loss_utils::psnr(masked_image, gt_image).item().toFloat();
    psnr_gs = loss_utils::psnr_gaussian_splatting(masked_image, gt_image).item().toFloat();

    if (record_rendered_image_) {
        auto image_cv = tensor_utils::torchTensor2CvMat_Float32(masked_image);
        cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
        image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_img_dir / (std::to_string(getIteration()) + "_" + std::to_string(pkf->imageid_) + name_suffix + ".jpg"), image_cv);
    }

    if (record_ground_truth_image_) {
        auto gt_image_cv = tensor_utils::torchTensor2CvMat_Float32(gt_image);
        cv::cvtColor(gt_image_cv, gt_image_cv, CV_RGB2BGR);
        gt_image_cv.convertTo(gt_image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_gt_dir / (std::to_string(getIteration()) + "_" + std::to_string(pkf->imageid_) + name_suffix + ".jpg"), gt_image_cv);
    }

    if (record_loss_image_) {
        torch::Tensor loss_tensor = torch::abs(masked_image - gt_image);
        auto loss_image_cv = tensor_utils::torchTensor2CvMat_Float32(loss_tensor);
        cv::cvtColor(loss_image_cv, loss_image_cv, CV_RGB2BGR);
        loss_image_cv.convertTo(loss_image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_loss_dir / (std::to_string(getIteration()) + "_" + std::to_string(pkf->imageid_) + name_suffix + "_loss.jpg"), loss_image_cv);
    }

    if (record_debug_image_)
    {
        auto project_pkg = GaussianRenderer::gaussians_project2_image(
            pkf,
            pkf->image_height_,
            pkf->image_width_,
            gaussians_,
            pipe_params_,
            background_,
            override_color_,
            voxel_visible_mask);
        auto points_image_2d = std::get<0>(project_pkg);
        auto radii = std::get<1>(project_pkg);
        auto color = std::get<2>(project_pkg);

        auto points_cpu = points_image_2d.detach().cpu();
        auto radii_cpu = radii.detach().cpu();
        auto color_cpu = color.detach().cpu();

        auto image_points = torch::zeros({pkf->image_height_, pkf->image_width_, 3},
                                torch::TensorOptions().dtype(color.dtype()).requires_grad(false).device(torch::kCUDA));

        int num_threads = 100; 
        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < points_image_2d.size(0); i++)
        {
            int x = points_cpu[i][0].item<int>();
            int y = points_cpu[i][1].item<int>();
            if (x >= 0 && x < pkf->image_width_ && y >= 0 && y < pkf->image_height_) {
                image_points.index({y, x, 0}) = color_cpu[i][0].clone();
                image_points.index({y, x, 1}) = color_cpu[i][1].clone();
                image_points.index({y, x, 2}) = color_cpu[i][2].clone();
            }
        }

        TORCH_CHECK(points_cpu.dim() == 2 && points_cpu.size(1) == 2, "points_image_2d should have shape (N, 2)");
        TORCH_CHECK(radii_cpu.dim() == 1, "radii should have shape (N,)");
        TORCH_CHECK(color_cpu.dim() == 2 && color_cpu.size(1) == 3, "color should have shape (N, 3)");

        cv::Mat image_radii(pkf->image_height_, pkf->image_width_, CV_8UC3, cv::Scalar(0, 0, 0));
        int num_gaussians = points_cpu.size(0);
        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < num_gaussians; ++i) {
            int x = points_cpu[i][0].item<int>();
            int y = points_cpu[i][1].item<int>();

            int radius = radii_cpu[i].item<int>();

            cv::Scalar ellipse_color(
                color_cpu[i][2].item<float>() * 255,
                color_cpu[i][1].item<float>() * 255,
                color_cpu[i][0].item<float>() * 255
            );
            cv::ellipse(image_radii, cv::Point(x, y), cv::Size(radius, radius), 0, 0, 360, ellipse_color, -1);
        }

        auto image_cpu = image_points.detach().cpu();
        cv::Mat image_mat(pkf->image_height_, pkf->image_width_, CV_32FC3, image_cpu.data_ptr<float>());

        image_mat.convertTo(image_mat, CV_8UC3, 255.0);
        cv::imwrite(image_points_dir / (std::to_string(getIteration()) + "_" + std::to_string(pkf->imageid_) + name_suffix + ".jpg"), image_mat);

        cv::imwrite(image_radii_dir / (std::to_string(getIteration()) + "_" + std::to_string(pkf->imageid_) + name_suffix + ".jpg"), image_radii);
        std::cout << "render And Record one frame" << std::endl;
    }
}

void GaussianMapper::renderAndRecordAllKeyframes(
    std::string name_suffix)
{
    std::filesystem::path result_dir = result_dir_ / (std::to_string(getIteration()) + name_suffix);
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

    std::filesystem::path image_dir = result_dir / "image";
    if (record_rendered_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_dir);

    std::filesystem::path image_gt_dir = result_dir / "image_gt";
    if (record_ground_truth_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_gt_dir);

    std::filesystem::path image_loss_dir = result_dir / "image_loss";
    if (record_loss_image_) {
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_loss_dir);
    }

    std::filesystem::path image_points_dir = result_dir / "image_points";
    std::filesystem::path image_radii_dir = result_dir / "image_radii";
    if (record_debug_image_)
    {
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_points_dir);
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_radii_dir);
    }

    std::filesystem::path render_time_path = result_dir / "render_time.txt";
    std::ofstream out_time(render_time_path);
    out_time << "##[Gaussian Mapper]Render time statistics: keyframe id, time(milliseconds)" << std::endl;
    
    std::filesystem::path dssim_path = result_dir / "dssim.txt";
    std::ofstream out_dssim(dssim_path);
    out_dssim << "##[Gaussian Mapper]keyframe id, dssim" << std::endl;

    std::filesystem::path psnr_path = result_dir / "psnr.txt";
    std::ofstream out_psnr(psnr_path);
    out_psnr << "##[Gaussian Mapper]keyframe id, psnr" << std::endl;

    std::filesystem::path psnr_gs_path = result_dir / "psnr_gaussian_splatting.txt";
    std::ofstream out_psnr_gs(psnr_gs_path);
    out_psnr_gs << "##[Gaussian Mapper]keyframe id, psnr_gaussian_splatting" << std::endl;

    gaussians_->eval();
    std::size_t nkfs = scene_->keyframes().size();
    auto kfit = scene_->keyframes().begin();
    float dssim, psnr, psnr_gs;
    float dssim_avg, psnr_avg, psnr_gs_avg = 0.0;
    double render_time;
    for (std::size_t i = 0; i < nkfs; ++i) {
        (*kfit).second->imageid_ = (*kfit).second->fid_;
        renderAndRecordKeyframe((*kfit).second, dssim, psnr, psnr_gs, render_time, image_dir, image_gt_dir, image_loss_dir, image_points_dir, image_radii_dir);
        out_time << (*kfit).first << " " << std::fixed << std::setprecision(8) << render_time << std::endl;

        out_dssim   << (*kfit).first << " " << std::fixed << std::setprecision(10) << dssim   << std::endl;
        out_psnr    << (*kfit).first << " " << std::fixed << std::setprecision(10) << psnr    << std::endl;
        out_psnr_gs << (*kfit).first << " " << std::fixed << std::setprecision(10) << psnr_gs << std::endl;

        ++kfit;
        psnr_avg += psnr/ float(nkfs);
        dssim_avg += dssim;
        psnr_gs_avg += psnr_gs;
    }
    dssim_avg = dssim_avg / float(nkfs);
    psnr_gs_avg = psnr_gs_avg / float(nkfs);


    std::cout << "[Render metric evaluation progress] at "<< getIteration() << " Iteration:" << std::endl;
    std::cout << "\tPSNR:: " << psnr_avg << ',' << nkfs << std::endl;
    std::cout << "\tSSIM:: " << dssim_avg << std::endl;
    std::cout << "\tPSNR_GS:: \033[1;32m" << psnr_gs_avg << "\033[0m" << std::endl;
    gaussians_->train();
}

void GaussianMapper::renderAndRecordAllframes(std::vector<ORB_SLAM3::KeyFrame*> vpFs)
{
    std::filesystem::path result_dir = result_dir_ / (std::to_string(getIteration()) + "_images");
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

    std::filesystem::path image_dir = result_dir / "all_image";
    if (record_rendered_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_dir);

    std::filesystem::path image_gt_dir = result_dir / "all_image_gt";
    if (record_ground_truth_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_gt_dir);

    std::filesystem::path image_loss_dir = result_dir / "all_image_loss";
    if (record_loss_image_) {
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_loss_dir);
    }

    std::filesystem::path image_points_dir = result_dir / "image_points";
    std::filesystem::path image_radii_dir = result_dir / "image_radii";
    if (record_debug_image_)
    {
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_points_dir);
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_radii_dir);
    }

    std::filesystem::path dssim_path = result_dir / "dssim.txt";
    std::ofstream out_dssim(dssim_path);
    out_dssim << "##[Gaussian Mapper]keyframe id, dssim" << std::endl;

    std::filesystem::path psnr_gs_path = result_dir / "psnr_gaussian_splatting.txt";
    std::ofstream out_psnr_gs(psnr_gs_path);
    out_psnr_gs << "##[Gaussian Mapper]keyframe id, psnr_gaussian_splatting" << std::endl;

    std::filesystem::path psnr_compare_path = result_dir / "psnr_compare.txt";
    std::ofstream out_psnr_compare(psnr_compare_path);
    out_psnr_compare << "##[Gaussian Mapper]frame id, keyframe id, psnr_gaussian_splatting" << std::endl;

    std::filesystem::path traj_path = result_dir / "AllCameraTrajectory_TUM.txt";
    std::ofstream out_traj(traj_path);
    out_traj << "##[Gaussian Mapper]traj in Tum" << std::endl;

    std::size_t nkfs = scene_->keyframes().size();
    auto kfit = scene_->keyframes().begin();
    std::deque<unsigned long> vec_kfID(nkfs, 0);
    std::deque<Sophus::SE3d> vec_kfPose(nkfs);
    for (std::size_t i = 0; i < nkfs; ++i) {
        vec_kfID[i] = (*kfit).second->frameID;
        vec_kfPose[i] = (*kfit).second->getPose();
        ++kfit;
    }
    gaussians_->eval();
    float dssim, psnr, psnr_gs;
    std::vector<float> dssim_vec(vpFs.size(), 0.0f);
    std::vector<float> psnr_gs_vec(vpFs.size(), 0.0f);
    double render_time;
    int idx;
    for (idx = 0; idx < vpFs.size(); idx++)
    {
        auto pKF = vpFs[idx];

        std::shared_ptr<GaussianKeyframe> new_kf = std::make_shared<GaussianKeyframe>(pKF->mnId, getIteration());
        auto pose = pKF->mTcw_eval;
        new_kf->setPose(
            pose.unit_quaternion().cast<double>(),
            pose.translation().cast<double>());

        cv::Mat imgRGB_undistorted, imgAux_undistorted;
        try {
            Camera& camera = scene_->cameras_.at(pKF->mpCamera->GetId());
            new_kf->setCameraParams(camera);

            cv::Mat imgRGB = pKF->imgLeftRGB;
            if (this->sensor_type_ == STEREO)
                imgRGB_undistorted = imgRGB;
            else
                camera.undistortImage(imgRGB, imgRGB_undistorted);
            cv::Mat imgAux = pKF->imgAuxiliary;
            if (this->sensor_type_ == RGBD)
                camera.undistortImage(imgAux, imgAux_undistorted);
            else
                imgAux_undistorted = imgAux;

            new_kf->original_image_ =
                tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
            new_kf->img_filename_ = pKF->mNameFile;
        }
        catch (std::out_of_range) {
            throw std::runtime_error("[GaussianMapper::run]KeyFrame Camera not found!");
        }
        new_kf->computeTransformTensors();
        new_kf->imageid_ = idx;
        new_kf->fid_ = pKF->mnId;
        if (new_kf->fid_ > scene_->keyframes().size())
            new_kf->fid_ = scene_->keyframes().size() - 1;

        renderAndRecordKeyframe(new_kf, dssim, psnr, psnr_gs, render_time, image_dir, image_gt_dir, image_loss_dir, image_points_dir, image_radii_dir);

        out_dssim   << idx << " " << std::fixed << std::setprecision(10) << dssim   << std::endl;
        out_psnr_gs << idx << " " << std::fixed << std::setprecision(10) << psnr_gs << std::endl;

        dssim_vec[idx] = dssim;
        psnr_gs_vec[idx] = psnr_gs;

        {
            float progress = static_cast<float>(idx + 1) / vpFs.size();
            int barWidth = 25;
            std::cout << "[";
            int pos = barWidth * progress;
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << int(progress * 100.0) << " % " << "pKF->mnId: " << pKF->mnId << "\r";
            std::cout.flush();
        }

        Eigen::Vector3d twc = pose.translation().cast<double>();
        auto q = pose.unit_quaternion().cast<double>();
        if (pKF->frameID == vec_kfID.front())
        {
            auto kfPose = vec_kfPose.front();
            Eigen::Vector3d twc_kf = kfPose.translation().cast<double>();
            auto q_kf = kfPose.unit_quaternion().cast<double>();

            std::cout << setprecision(6) << pKF->frameID << " " << setprecision(9)
                    << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << std::endl
                     << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
            
            out_psnr_compare << pKF->frameID  << " " << pKF->mnId  << " " << std::fixed << std::setprecision(10) << psnr_gs << std::endl;
            vec_kfID.pop_front();
            vec_kfPose.pop_front();
        }
            out_traj << setprecision(6) << 111 << " " << setprecision(9)
                    << twc(0) << " " << twc(1) << " " << twc(2) << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;    }
    float dssim_avg_value = std::accumulate(dssim_vec.begin(), dssim_vec.end(), 0.0f) / vpFs.size();
    float psnr_gs_avg_value = std::accumulate(psnr_gs_vec.begin(), psnr_gs_vec.end(), 0.0f) / vpFs.size();

    std::cout << "[Render metric evaluation progress] at "<< getIteration() << " Iteration: "  << idx  << std::endl;
    std::cout << "\tSSIM:: " << dssim_avg_value << std::endl;
    std::cout << "\tPSNR_GS:: \033[1;32m" << psnr_gs_avg_value << "\033[0m" << std::endl;
}

void GaussianMapper::savePly(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    keyframesToJson(result_dir);
    saveModelParams(result_dir);

    std::filesystem::path ply_dir = result_dir / "point_cloud";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

    ply_dir = ply_dir / ("iteration_" + std::to_string(getIteration()));
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

    gaussians_->savePly(ply_dir / "point_cloud.ply");

    std::filesystem::path mlp_dir = ply_dir / "mlp";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(mlp_dir)

    gaussians_->save_mlp_checkpoints(mlp_dir);
}

void GaussianMapper::keyframesToJson(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

    std::filesystem::path result_path = result_dir / "cameras.json";
    std::ofstream out_stream;
    out_stream.open(result_path);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open json file at " + result_path.string());

    Json::Value json_root;
    Json::StreamWriterBuilder builder;
    const std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());

    int i = 0;
    for (const auto& kfit : scene_->keyframes()) {
        const auto pkf = kfit.second;
        Eigen::Matrix4f Rt;
        Rt.setZero();
        Eigen::Matrix3f R = pkf->R_quaternion_.toRotationMatrix().cast<float>();
        Rt.topLeftCorner<3, 3>() = R;
        Eigen::Vector3f t = pkf->t_.cast<float>();
        Rt.topRightCorner<3, 1>() = t;
        Rt(3, 3) = 1.0f;

        Eigen::Matrix4f Twc = Rt.inverse();
        Eigen::Vector3f pos = Twc.block<3, 1>(0, 3);
        Eigen::Matrix3f rot = Twc.block<3, 3>(0, 0);

        Json::Value json_kf;
        json_kf["id"] = static_cast<Json::Value::UInt64>(pkf->fid_);
        json_kf["img_name"] = pkf->img_filename_;
        json_kf["width"] = pkf->image_width_;
        json_kf["height"] = pkf->image_height_;

        json_kf["position"][0] = pos.x();
        json_kf["position"][1] = pos.y();
        json_kf["position"][2] = pos.z();

        json_kf["rotation"][0][0] = rot(0, 0);
        json_kf["rotation"][0][1] = rot(0, 1);
        json_kf["rotation"][0][2] = rot(0, 2);
        json_kf["rotation"][1][0] = rot(1, 0);
        json_kf["rotation"][1][1] = rot(1, 1);
        json_kf["rotation"][1][2] = rot(1, 2);
        json_kf["rotation"][2][0] = rot(2, 0);
        json_kf["rotation"][2][1] = rot(2, 1);
        json_kf["rotation"][2][2] = rot(2, 2);

        json_kf["fy"] = graphics_utils::fov2focal(pkf->FoVy_, pkf->image_height_);
        json_kf["fx"] = graphics_utils::fov2focal(pkf->FoVx_, pkf->image_width_);

        json_root[i] = Json::Value(json_kf);
        ++i;
    }

    writer->write(json_root, &out_stream);
}

void GaussianMapper::saveModelParams(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    std::filesystem::path result_path = result_dir / "cfg_args";
    std::ofstream out_stream;
    out_stream.open(result_path);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open file at " + result_path.string());

    out_stream << "Namespace("
               << "eval=" << (model_params_.eval_ ? "True" : "False") << ", "
               << "images=" << "\'" << model_params_.images_ << "\', "
               << "model_path=" << "\'" << model_params_.model_path_.string() << "\', "
               << "resolution=" << model_params_.resolution_ << ", "
               << "sh_degree=" << model_params_.sh_degree_ << ", "
               << "source_path=" << "\'" << model_params_.source_path_.string() << "\', "
               << "white_background=" << (model_params_.white_background_ ? "True" : "False") << ", "
               << ")";

    out_stream.close();
}

void GaussianMapper::writeKeyframeUsedTimes(std::filesystem::path result_dir, std::string name_suffix)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    std::filesystem::path result_path = result_dir / ("keyframe_used_times" + name_suffix + ".txt");
    std::ofstream out_stream;
    out_stream.open(result_path, std::ios::app);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open json at " + result_path.string());

    out_stream << "##[Gaussian Mapper]Iteration " << getIteration() << " keyframe id, used times, remaining times:\n";
    for (const auto& used_times_it : kfs_used_times_)
        out_stream << used_times_it.first << " "
                   << used_times_it.second << " "
                   << scene_->keyframes().at(used_times_it.first)->remaining_times_of_use_
                   << "\n";
    out_stream << "##=========================================" <<std::endl;

    out_stream.close();
}

int GaussianMapper::getIteration()
{
    std::unique_lock<std::mutex> lock(mutex_status_);
    return iteration_;
}
void GaussianMapper::increaseIteration(const int inc)
{
    std::unique_lock<std::mutex> lock(mutex_status_);
    iteration_ += inc;
}

float GaussianMapper::positionLearningRateInit()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.position_lr_init_;
}
float GaussianMapper::featureLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.feature_lr_;
}

float GaussianMapper::anchorFeatureLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.feature_lr_;
}
float GaussianMapper::opacityLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.opacity_lr_;
}
float GaussianMapper::scalingLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.scaling_lr_;
}
float GaussianMapper::rotationLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.rotation_lr_;
}
float GaussianMapper::percentDense()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.percent_dense_;
}
float GaussianMapper::lambdaDssim()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.lambda_dssim_;
}
float GaussianMapper::densifyGradThreshold()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.densify_grad_threshold;
}
int GaussianMapper::densifyInterval()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.update_interval;
}
int GaussianMapper::newKeyframeTimesOfUse()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return new_keyframe_times_of_use_;
}
int GaussianMapper::stableNumIterExistence()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return stable_num_iter_existence_;
}
bool GaussianMapper::isKeepingTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return keep_training_;
}
bool GaussianMapper::isdoingGausPyramidTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return do_gaus_pyramid_training_;
}
bool GaussianMapper::isdoingInactiveGeoDensify()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return inactive_geo_densify_;
}
void GaussianMapper::setFeatureLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.feature_lr_ = lr;
}
void GaussianMapper::setOpacityLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.opacity_lr_ = lr;
}
void GaussianMapper::setScalingLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.scaling_lr_ = lr;
}
void GaussianMapper::setRotationLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.rotation_lr_ = lr;
}
void GaussianMapper::setPercentDense(const float percent_dense)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.percent_dense_ = percent_dense;
    gaussians_->setPercentDense(percent_dense);
}
void GaussianMapper::setLambdaDssim(const float lambda_dssim)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.lambda_dssim_ = lambda_dssim;
}

void GaussianMapper::setDensifyGradThreshold(const float th)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.densify_grad_threshold = th;
}

void GaussianMapper::setNewKeyframeTimesOfUse(const int times)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    new_keyframe_times_of_use_ = times;
}

void GaussianMapper::setKeepTraining(const bool keep)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    keep_training_ = keep;
}

void GaussianMapper::args_logger(GaussianModelParams model_params,
                                 GaussianOptimizationParams opt_params,
                                 GaussianPipelineParams pipe_params)
{
    std::cout << "\033[1;32m[Gaussian Param]\033[0m args: "
              << "add_color_dist = " << model_params.add_color_dist
              << ", add_cov_dist = " << model_params.add_cov_dist
              << ", add_opacity_dist = " << model_params.add_opacity_dist
              << ", appearance_dim = " << model_params.appearance_dim
              << ", appearance_lr_delay_mult = " << opt_params.appearance_lr_delay_mult
              << ", appearance_lr_final = " << opt_params.appearance_lr_final
              << ", appearance_lr_init = " << opt_params.appearance_lr_init
              << ", appearance_lr_max_steps = " << opt_params.appearance_lr_max_steps
              << ", compute_cov3D = " << pipe_params.compute_cov3D_
              << ", convert_SHs = " << pipe_params.convert_SHs_
              << ", data_device_ = " << model_params.data_device_
              << ", densify_grad_threshold = " << opt_params.densify_grad_threshold
              << ", ds = " << model_params.ds
              << ", feat_dim = " << model_params.feat_dim
              << ", feature_lr_ = " << opt_params.feature_lr_
              << ", iterations = " << opt_params.iterations_
              << ", lambda_dssim_ = " << opt_params.lambda_dssim_
              << ", lowpoly = " << model_params.lowpoly
              << ", min_opacity = " << opt_params.min_opacity
              << ", mlp_color_lr_delay_mult = " << opt_params.mlp_color_lr_delay_mult
              << ", mlp_color_lr_final = " << opt_params.mlp_color_lr_final
              << ", mlp_color_lr_init = " << opt_params.mlp_color_lr_init
              << ", mlp_color_lr_max_steps = " << opt_params.mlp_color_lr_max_steps
              << ", mlp_cov_lr_delay_mult = " << opt_params.mlp_cov_lr_delay_mult
              << ", mlp_cov_lr_final = " << opt_params.mlp_cov_lr_final
              << ", mlp_cov_lr_init = " << opt_params.mlp_cov_lr_init
              << ", mlp_cov_lr_max_steps = " << opt_params.mlp_cov_lr_max_steps
              << ", mlp_featurebank_lr_delay_mult = " << opt_params.mlp_featurebank_lr_delay_mult
              << ", mlp_featurebank_lr_final = " << opt_params.mlp_featurebank_lr_final
              << ", mlp_featurebank_lr_init = " << opt_params.mlp_featurebank_lr_init
              << ", mlp_featurebank_lr_max_steps = " << opt_params.mlp_featurebank_lr_max_steps
              << ", mlp_opacity_lr_delay_mult = " << opt_params.mlp_opacity_lr_delay_mult
              << ", mlp_opacity_lr_final = " << opt_params.mlp_opacity_lr_final
              << ", mlp_opacity_lr_init = " << opt_params.mlp_opacity_lr_init
              << ", mlp_opacity_lr_max_steps = " << opt_params.mlp_opacity_lr_max_steps
              << ", n_offsets = " << model_params.n_offsets
              << ", offset_lr_delay_mult = " << opt_params.offset_lr_delay_mult
              << ", offset_lr_final = " << opt_params.offset_lr_final
              << ", offset_lr_init = " << opt_params.offset_lr_init
              << ", offset_lr_max_steps = " << opt_params.offset_lr_max_steps
              << ", opacity_lr_ = " << opt_params.opacity_lr_
              << ", percent_dense_ = " << opt_params.percent_dense_
              << ", position_lr_delay_mult_ = " << opt_params.position_lr_delay_mult_
              << ", position_lr_final_ = " << opt_params.position_lr_final_
              << ", position_lr_init_ = " << opt_params.position_lr_init_
              << ", position_lr_max_steps_ = " << opt_params.position_lr_max_steps_
              << ", ratio = " << model_params.ratio
              << ", resolution_ = " << model_params.resolution_
              << ", rotation_lr_ = " << opt_params.rotation_lr_
              << ", scaling_lr_ = " << opt_params.scaling_lr_
              << ", sh_degree_ = " << model_params.sh_degree_
              << ", start_stat = " << opt_params.start_stat
              << ", success_threshold = " << opt_params.success_threshold
              << ", undistorted = " << model_params.undistorted
              << ", update_depth = " << model_params.update_depth
              << ", update_from = " << opt_params.update_from
              << ", update_hierachy_factor = " << model_params.update_hierachy_factor
              << ", update_init_factor = " << model_params.update_init_factor
              << ", update_interval = " << opt_params.update_interval
              << ", update_until = " << opt_params.update_until
              << ", use_feat_bank = " << model_params.use_feat_bank
              << ", voxel_size = " << model_params.voxel_size
              << ", white_background_ = " << model_params.white_background_
              << std::endl;
}

void GaussianMapper::LoadTrajectory(const std::string &filePath) {
    std::ifstream file(filePath);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filePath << std::endl;
        return;
    }

    std::string line;
    int index = 0;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::vector<double> row;
        double value;

        while (ss >> value) {
            row.push_back(value);
        }

        if (!row.empty()) {
            pose_[index] = row;
            ++index;
        }
    }

    file.close();
}

cv::Mat GaussianMapper::renderFromPose(
    const Sophus::SE3f &Tcw,
    const int width,
    const int height,
    const bool main_vision)
{
    if (!initial_mapped_ || getIteration() <= 0)
        return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
    std::shared_ptr<GaussianKeyframe> pkf = std::make_shared<GaussianKeyframe>();
    pkf->zfar_ = z_far_;
    pkf->znear_ = z_near_;
    pkf->setPose(
        Tcw.unit_quaternion().cast<double>(),
        Tcw.translation().cast<double>());
    try {
        Camera& camera = scene_->cameras_.at(viewer_camera_id_);
        pkf->setCameraParams(camera);
        pkf->computeTransformTensors();
    }
    catch (std::out_of_range) {
        throw std::runtime_error("[GaussianMapper::renderFromPose]KeyFrame Camera not found!");
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, bool> render_pkg;
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        auto voxel_visible_mask = GaussianRenderer::prefilter_voxel(
            pkf,
            height,
            width,
            gaussians_,
            pipe_params_,
            background_,
            override_color_
        );
        bool retain_grad = false;
        render_pkg = GaussianRenderer::render(
            pkf,
            height,
            width,
            gaussians_,
            pipe_params_,
            background_,
            override_color_,
            voxel_visible_mask,
            retain_grad
        );
    }

    torch::Tensor masked_image;
    if (main_vision)
        masked_image = std::get<0>(render_pkg) * viewer_main_undistort_mask_[pkf->camera_id_];
    else
        masked_image = std::get<0>(render_pkg) * viewer_sub_undistort_mask_[pkf->camera_id_];
    return tensor_utils::torchTensor2CvMat_Float32(masked_image);
}