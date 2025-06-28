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

#include <torch/torch.h>

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <thread>
#include <filesystem>
#include <memory>

#include <opencv2/core/core.hpp>

#include "ORB-SLAM3/include/System.h"

#include "include/gaussian_mapper.h"
#include "viewer/imgui_viewer.h"
#include "example_utils.h"

void LoadImages(const std::filesystem::path &pathImageDir, std::vector<std::string> &vstrImageFilenamesRGB,
                std::vector<std::string> &vstrImageFilenamesD);
void saveTrackingTime(std::vector<float> &vTimesTrack, const std::string &strSavePath);
void saveGpuPeakMemoryUsage(std::filesystem::path pathSave);

int main(int argc, char **argv)
{
    auto start_timing = std::chrono::steady_clock::now();
    if (argc != 6 && argc != 7 && argc != 8)
    {
        std::cerr << std::endl
                  << "Usage: " << argv[0]
                  << " path_to_vocabulary"                   /*1*/
                  << " path_to_ORB_SLAM3_settings"           /*2*/
                  << " path_to_gaussian_mapping_settings"    /*3*/
                  << " path_to_sequence"                     /*4*/
                  << " path_to_trajectory_output_directory/" /*5*/
                  << " (optional)no_viewer"                  /*6*/
                  << " (optional)undistorted_image"          /*7*/
                  << std::endl;
        return 1;
    }
    bool use_viewer = true;
    if (argc == 7)
        use_viewer = (std::string(argv[6]) == "no_viewer" ? false : true);

    bool has_undistorted_image = false;
    if (argc == 8)
    {
        use_viewer = (std::string(argv[6]) == "no_viewer" ? false : true); 
        has_undistorted_image = (std::string(argv[7]) == "undistorted_image" ? true : false);     
    }
    if (argc == 7)
    {
        has_undistorted_image = (std::string(argv[6]) == "undistorted_image" ? true : false);     
        use_viewer = (std::string(argv[6]) == "no_viewer" ? false : true);    
    }

    std::string output_directory = std::string(argv[5]);
    if (output_directory.back() != '/')
        output_directory += "/";
    std::filesystem::path output_dir(output_directory);

    // Retrieve paths to images
    std::vector<std::string> vstrImageFilenamesRGB;
    std::vector<std::string> vstrImageFilenamesD;
    std::string strImageDir = std::string(argv[4]);
    std::filesystem::path pathImageDir(strImageDir);
    pathImageDir /= "results";
    LoadImages(pathImageDir, vstrImageFilenamesRGB, vstrImageFilenamesD);

    // Check consistency in the number of images
    int nImages = vstrImageFilenamesRGB.size();
    if (vstrImageFilenamesRGB.empty())
    {
        std::cerr << std::endl << "No images found in provided path." << std::endl;
        return 1;
    }
    else if (vstrImageFilenamesD.size() != vstrImageFilenamesRGB.size())
    {
        std::cerr << std::endl << "Different number of images for rgb and depth." << std::endl;
        return 1;
    }

    // Device
    torch::DeviceType device_type;
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else
    {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    std::shared_ptr<ORB_SLAM3::System> pSLAM =
        std::make_shared<ORB_SLAM3::System>(
            argv[1], argv[2], ORB_SLAM3::System::RGBD);
    float imageScale = pSLAM->GetImageScale();

    // Create GaussianMapper
    std::filesystem::path gaussian_cfg_path(argv[3]);
    std::shared_ptr<GaussianMapper> pGausMapper =
        std::make_shared<GaussianMapper>(
            pSLAM, gaussian_cfg_path, output_dir, 0, device_type);
    std::thread training_thd(&GaussianMapper::run, pGausMapper.get());
    pGausMapper->use_undistorted_image = has_undistorted_image;

    // Create Gaussian Viewer
    std::thread viewer_thd;
    std::shared_ptr<ImGuiViewer> pViewer;
    if (use_viewer)
    {
        pViewer = std::make_shared<ImGuiViewer>(pSLAM, pGausMapper);
        viewer_thd = std::thread(&ImGuiViewer::run, pViewer.get());
    }

    // Vector for tracking time statistics
    std::vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    std::cout << std::endl << "-------" << std::endl;
    std::cout << "Start processing sequence ..." << std::endl;
    std::cout << "Images in the sequence: " << nImages << std::endl << std::endl;

    // Main loop
    cv::Mat imRGB, imD;
    for (int ni = 0; ni < nImages; ni++)
    {
        if (pSLAM->isShutDown())
            break;
        // Read image and depthmap from file
        imRGB = cv::imread(vstrImageFilenamesRGB[ni], cv::IMREAD_UNCHANGED);
        cv::cvtColor(imRGB, imRGB, CV_BGR2RGB);
        imD = cv::imread(vstrImageFilenamesD[ni], cv::IMREAD_UNCHANGED);
        double tframe = ni;
        std::cout << ni << " " << imRGB.type() << " " << imD.type()<< std::endl;

        cv::Mat undistorted_img;
        if (has_undistorted_image)
        {
            undistorted_img = imRGB;
        }
        else
            undistorted_img = imRGB;

        if (imRGB.empty())
        {
            std::cerr << std::endl << "Failed to load image at: "
                      << vstrImageFilenamesRGB[ni] << std::endl;
            return 1;
        }
        if (imD.empty())
        {
            std::cerr << std::endl << "Failed to load image at: "
                      << vstrImageFilenamesD[ni] << std::endl;
            return 1;
        }

        if (imageScale != 1.f)
        {
            int width = imRGB.cols * imageScale;
            int height = imRGB.rows * imageScale;
            cv::resize(imRGB, imRGB, cv::Size(width, height));
            cv::resize(imD, imD, cv::Size(width, height));
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // Pass the image to the SLAM system
        if (has_undistorted_image)
        {
            pSLAM->TrackRGBD(imRGB, imD, tframe, std::vector<ORB_SLAM3::IMU::Point>(), vstrImageFilenamesRGB[ni], undistorted_img);
        }
        else
            pSLAM->TrackRGBD(imRGB, imD, tframe, std::vector<ORB_SLAM3::IMU::Point>(), vstrImageFilenamesRGB[ni]);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        vTimesTrack[ni] = ttrack;
    }

    pSLAM->SaveTrajectoryTUM((output_dir / "CameraTrajectory_TUM_bf.txt").string());
    std::cout << "SaveTrajectoryTUM" << std::endl;
    // 写个读取函数,把轨迹和mnid绑定哈希表
    example_utils::LoadTrajectory((output_dir / "CameraTrajectory_TUM_bf.txt").string(), pGausMapper->pose_);
    pGausMapper->poseSaved =true;

    // Stop all threads
    pSLAM->Shutdown();
    training_thd.join();
    if (use_viewer)
        viewer_thd.join();

    auto end_timing = std::chrono::steady_clock::now();
    auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_timing - start_timing).count();
    std::ofstream out(output_dir / "RunnningTime.txt");
    out << "Totoal running Time (ms): " << runtime << std::endl;
    out.close();

    // GPU peak usage
    saveGpuPeakMemoryUsage(output_dir / "GpuPeakUsageMB.txt");

    // Tracking time statistics
    saveTrackingTime(vTimesTrack, (output_dir / "TrackingTime.txt").string());

    // Save camera trajectory
    pSLAM->SaveTrajectoryTUM((output_dir / "CameraTrajectory_TUM.txt").string());
    pSLAM->SaveKeyFrameTrajectoryTUM((output_dir / "KeyFrameTrajectory_TUM.txt").string());
    pSLAM->SaveTrajectoryEuRoC((output_dir / "CameraTrajectory_EuRoC.txt").string());
    pSLAM->SaveKeyFrameTrajectoryEuRoC((output_dir / "KeyFrameTrajectory_EuRoC.txt").string());
    pSLAM->SaveTrajectoryKITTI((output_dir / "CameraTrajectory_KITTI.txt").string());

    {        
        //0719 现在能确定训练时使用的位姿求逆与SLAM结束后保存的位姿相同
        std::filesystem::path result_dir = output_dir / (std::to_string(pGausMapper->getIteration()) + "_images");
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

        std::filesystem::path image_dir = result_dir / "all_image";
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_dir);

        std::filesystem::path image_gt_dir = result_dir / "all_image_gt";
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_gt_dir);

        std::filesystem::path image_loss_dir = result_dir / "all_image_loss";
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_loss_dir);

        std::filesystem::path image_points_dir = result_dir / "image_points";
            CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_points_dir);

        std::filesystem::path image_radii_dir = result_dir / "image_radii";
            CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_radii_dir);

        std::filesystem::path kf_image_dir = result_dir / "kf_image";
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(kf_image_dir);

        std::filesystem::path kf_image_gt_dir = result_dir / "kf_image_gt";
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(kf_image_gt_dir);

        std::filesystem::path kf_image_undistorted_dir = result_dir / "kf_image_undistorted";
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(kf_image_undistorted_dir);

        std::filesystem::path kf_image_points_dir = result_dir / "kf_image_points";
            CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(kf_image_points_dir);

        std::filesystem::path kf_image_radii_dir = result_dir / "kf_image_radii";
            CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(kf_image_radii_dir);

        std::filesystem::path metric_save_path = output_dir / "eval_metric.txt";
        std::ofstream out_metric(metric_save_path);

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

        std::filesystem::path kf_path = result_dir / "AllKeyframeTrajectory_TUM.txt";
        std::ofstream out_kf(kf_path);
        out_kf << "##[Gaussian Mapper]traj in Tum" << std::endl;

        std::size_t nkfs = pGausMapper->scene_->keyframes().size();
        auto kfit = pGausMapper->scene_->keyframes().begin();
        std::deque<unsigned long> vec_kfID(nkfs, 0);
        std::deque<Sophus::SE3d> vec_kfPose(nkfs);
        std::deque<unsigned long> vec_kfid(nkfs, 0);
        std::deque<unsigned long> vec_kfts(nkfs, 0);
        for (std::size_t i = 0; i < nkfs; ++i) {
            vec_kfID[i] = (*kfit).second->frameID;
            vec_kfPose[i] = (*kfit).second->getPose();
            vec_kfid[i] = (*kfit).second->fid_;
            vec_kfts[i] = (*kfit).second->TimeStamp;//时间戳不全，前面少了
            auto trans = (*kfit).second->getPose().inverse().translation().cast<double>();
            auto rot = (*kfit).second->getPose().inverse().unit_quaternion().cast<double>();
            out_kf << setprecision(6) << (*kfit).second->TimeStamp << " " << setprecision(9)
                     << trans(0) << " " << trans(1) << " " << trans(2) << " " << rot.x() << " " << rot.y() << " " << rot.z() << " " << rot.w() << std::endl;
            ++kfit;
        }

        std::vector<std::vector<double>> Traj;
        // LoadTrajectory((output_dir / "CameraTrajectory_TUM.txt").string(), Traj);
        example_utils::LoadTrajectory((output_dir / "CameraTrajectory_TUM_bf.txt").string(), Traj);

        // auto vpFs = pSLAM_->GetAllFrames();
        // auto vpFs = pSLAM_->GetAllFramesPose();
        // auto vpAllFs = pSLAM_->GetAllFramesPose();
        auto allkeyframes = pGausMapper->pSLAM_->getAtlas()->GetAllKeyFrames();
        std::deque<unsigned long> vec_allkeyframesID(nkfs, 0);
        std::deque<Sophus::SE3f> vec_allkeyframesPose(nkfs);   
        for (std::size_t i = 0; i < allkeyframes.size(); ++i) {
            vec_allkeyframesID[i] = allkeyframes[i]->frameID;
            vec_allkeyframesPose[i] = allkeyframes[i]->GetPose();

            // auto trans = (*kfit).second->getPose().inverse().translation().cast<double>();
            // auto rot = (*kfit).second->getPose().inverse().unit_quaternion().cast<double>();
            // out_kf << setprecision(6) << (*kfit).second->TimeStamp << " " << setprecision(9)
            //          << trans(0) << " " << trans(1) << " " << trans(2) << " " << rot.x() << " " << rot.y() << " " << rot.z() << " " << rot.w() << std::endl;
            // ++kfit;
        }   

        pGausMapper->gaussians_->eval();
        float dssim, psnr, psnr_gs;
        std::vector<float> dssim_vec;
        std::vector<float> dssim_Kf_vec;
        std::vector<float> psnr_gs_vec;
        std::vector<float> psnr_gsKf_vec;
        double render_time;
        int idx, idx_scaffold;
        idx_scaffold = 0;
        unsigned long keyframe_id = vec_kfID.front();
        for (idx = 0; idx < Traj.size(); idx++)
        {
            // auto pKF = vpFs[idx];

            std::shared_ptr<GaussianKeyframe> new_kf = std::make_shared<GaussianKeyframe>(idx, pGausMapper->getIteration());
            // Pose
            // auto pose = pKF->mTcw_eval.inverse();
            Eigen::Vector3d pose_trans(Traj[idx][1], Traj[idx][2], Traj[idx][3]);
            Eigen::Quaterniond pose_rot(Traj[idx][7], Traj[idx][4], Traj[idx][5], Traj[idx][6]);
            Sophus::SE3d pose(pose_rot, pose_trans);
            Sophus::SE3d c2w = pose.inverse();
            auto c2w_q = c2w.unit_quaternion();
            auto c2w_t = c2w.translation();
            // auto pose = pKF->GetPose();
            new_kf->setPose(c2w_q, c2w_t);
            // auto pF = vpAllFs[idx];   
            // auto Twc = pF->GetPose();
            // // new_kf->setPose(
            // //     Twc.unit_quaternion().cast<double>(),
            // //     Twc.translation().cast<double>());

            // std::cout << "Twc " << setprecision(4) << Twc.translation()(0) - pose.translation()(0)
            //           << " " << Twc.translation()(1) - pose.translation()(1) << " " << Twc.translation()(2) - pose.translation()(2)
            //           << " " << Twc.unit_quaternion().x() - pose.unit_quaternion().x() << " " << Twc.unit_quaternion().y() - pose.unit_quaternion().y()
            //           << " " << Twc.unit_quaternion().z() - pose.unit_quaternion().z() << " " << Twc.unit_quaternion().w() - pose.unit_quaternion().w() << std::endl;
            // std::cout << "pose " << setprecision(4) << pose.translation()(0) << " " << pose.translation()(1) << " " << pose.translation()(2) << " "
            //           << pose.unit_quaternion().x() << " " << pose.unit_quaternion().y() << " "
            //           << pose.unit_quaternion().z() << " " << pose.unit_quaternion().w() << std::endl;

            cv::Mat imgRGB_undistorted, imgAux_undistorted;
            try {
                if(has_undistorted_image)
                {
                    // Camera
                    Camera& camera = pGausMapper->scene_->cameras_.begin()->second;

                    // Image (left if STEREO)
                    int img_name = example_utils::extractFrameNumber(vstrImageFilenamesRGB[idx]);
                    if(double(img_name) != Traj[idx][0])
                    {
                        std::cout << "mismatch img_name " << img_name << " and idx " << Traj[idx][0] << std::endl;
                    }
                    cv::Mat imgRGB = cv::imread(vstrImageFilenamesRGB[idx], cv::IMREAD_UNCHANGED);
                    cv::cvtColor(imgRGB, imgRGB, CV_BGR2RGB);
                    imgRGB_undistorted = imgRGB;//undistortedRGB; //only for RGBD


                    if (imgRGB_undistorted.type() == CV_8UC3)
                        imgRGB_undistorted.convertTo(imgRGB_undistorted, CV_32FC3, 1.0 / 255.0);
                    else if (imgRGB_undistorted.type() == CV_16UC3)
                        imgRGB_undistorted.convertTo(imgRGB_undistorted, CV_32FC3, 1.0 / 65535.0);
                    else if (imgRGB_undistorted.type() == CV_16FC3 || imgRGB_undistorted.type() == CV_64FC3)
                        imgRGB_undistorted.convertTo(imgRGB_undistorted, CV_32FC3, 1.0);

                    new_kf->setCameraParams(camera, imgRGB_undistorted.size);
                    new_kf->original_image_ =
                        tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, pGausMapper->device_type_);
                    new_kf->img_filename_ = img_name;
                }
                else{
                // Camera
                // Camera& camera = pGausMapper->scene_->cameras_.at(pKF->mpCamera->GetId());
                Camera &camera = pGausMapper->scene_->cameras_.begin()->second;
                new_kf->setCameraParams(camera);

                // Image (left if STEREO)
                int img_name = example_utils::extractFrameNumber(vstrImageFilenamesRGB[idx]);
                if(double(img_name) != Traj[idx][0])
                {
                    std::cout << "mismatch img_name " << img_name << " and idx " << Traj[idx][0] << std::endl;
                }
                cv::Mat imgRGB = cv::imread(vstrImageFilenamesRGB[idx], cv::IMREAD_UNCHANGED);
                cv::cvtColor(imgRGB, imgRGB, CV_BGR2RGB);

                if (imgRGB.type() == CV_8UC3)
                    imgRGB.convertTo(imgRGB, CV_32FC3, 1.0 / 255.0);
                else if (imgRGB.type() == CV_16UC3)
                    imgRGB.convertTo(imgRGB, CV_32FC3, 1.0 / 65535.0);
                else if (imgRGB.type() == CV_16FC3 || imgRGB.type() == CV_64FC3)
                    imgRGB.convertTo(imgRGB, CV_32FC3, 1.0);

                // cv::Mat imgRGB = pKF->imgLeftRGB;
                if (pGausMapper->sensor_type_ == STEREO)
                    imgRGB_undistorted = imgRGB;
                else
                    camera.undistortImage(imgRGB, imgRGB_undistorted);


                new_kf->original_image_ =
                    tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, pGausMapper->device_type_);
                new_kf->img_filename_ = img_name;
                }
            }
            catch (std::out_of_range) {
                throw std::runtime_error("[GaussianMapper::run]KeyFrame Camera not found!");
            }
            new_kf->computeTransformTensors();
            new_kf->imageid_ = idx;
            // new_kf->fid_ = pKF->mnId; // 关注下appearance embedding在测试集上怎么弄
            // if (new_kf->fid_ > pGausMapper->scene_->keyframes().size())
            //     new_kf->fid_ = pGausMapper->scene_->keyframes().size() - 1;
            // For test-set visualizations, we choose index to best fit a target image (e.g. Figure 8) or set it to an arbitrary value

            for (std::size_t j = 0; j < allkeyframes.size(); ++j) {
                if (idx == vec_allkeyframesID[j])
                {
                    auto kfPose = vec_allkeyframesPose[j];
                    Eigen::Vector3d twc_kf = kfPose.translation().cast<double>();
                    auto q_kf = kfPose.unit_quaternion().cast<double>();

                    // std::cout << setprecision(6) << idx << " " << setprecision(9)
                    //         << c2w_t(0) - twc_kf(0) << " " << c2w_t(1) - twc_kf(1) << " " << c2w_t(2) - twc_kf(2) << " " << c2w_q.x() - q_kf.x() << std::endl
                    //         << " " << c2w_q.y() - q_kf.y() << " " << c2w_q.z() - q_kf.z() << " " << c2w_q.w() - q_kf.w() << std::endl;
                }//完全一样
            }

            //compare pose
            if (vec_kfID.size() > 0){
            if (idx == vec_kfID.front())
            {
                keyframe_id = vec_kfid.front();
                auto kfPose = vec_kfPose.front();
                Eigen::Vector3d twc_kf = kfPose.translation().cast<double>();
                auto q_kf = kfPose.unit_quaternion().cast<double>();

                if (std::abs(c2w_t(0) - twc_kf(0))> 0.0001 )
                {
                    std::cout << setprecision(6) << idx << " " << setprecision(9)
                            << c2w_t(0) - twc_kf(0) << " " << c2w_t(1) - twc_kf(1) << " " << c2w_t(2) - twc_kf(2) << " " << c2w_q.x() - q_kf.x() << std::endl
                            << " " << c2w_q.y() - q_kf.y() << " " << c2w_q.z() - q_kf.z() << " " << c2w_q.w() - q_kf.w() << std::endl;                    
                }

                vec_kfPose.pop_front();
                vec_kfid.pop_front();
                // new_kf->fid_ = keyframe_id;
            }
            // else{
                // idx_scaffold++;
                // new_kf->fid_ = (idx_scaffold - 1) % nkfs;
            // }
            }
            new_kf->fid_ = keyframe_id;//这是我优化后的
            if (vec_kfID.size() > 0){
            if (idx != vec_kfID.front())
            {
                pGausMapper->renderAndRecordKeyframe(new_kf, dssim, psnr, psnr_gs, render_time, image_dir, image_gt_dir, image_loss_dir, image_points_dir, image_radii_dir);

                out_dssim   << idx << " " << std::fixed << std::setprecision(10) << dssim   << std::endl;
                out_psnr_gs << idx << " " << std::fixed << std::setprecision(10) << psnr_gs << std::endl;

                dssim_vec.push_back(dssim);
                psnr_gs_vec.push_back(psnr_gs);
            }
            }
            else{
                pGausMapper->renderAndRecordKeyframe(new_kf, dssim, psnr, psnr_gs, render_time, image_dir, image_gt_dir, image_loss_dir, image_points_dir, image_radii_dir);

                out_dssim   << idx << " " << std::fixed << std::setprecision(10) << dssim   << std::endl;
                out_psnr_gs << idx << " " << std::fixed << std::setprecision(10) << psnr_gs << std::endl;

                dssim_vec.push_back(dssim);
                psnr_gs_vec.push_back(psnr_gs);                
            }

            //progress bar
            {
                float progress = static_cast<float>(idx + 1) / Traj.size();
                int barWidth = 25;
                std::cout << "[";
                int pos = barWidth * progress;
                for (int i = 0; i < barWidth; ++i) {
                    if (i < pos) std::cout << "=";
                    else if (i == pos) std::cout << ">";
                    else std::cout << " ";
                }
                std::cout << "] " << int(progress * 100.0) << " % " << "id " << idx << "| " << new_kf->fid_ << "\r";
                std::cout.flush();
            }

            if (vec_kfID.size() > 0){
            if (idx == vec_kfID.front())
            {
                pGausMapper->renderAndRecordKeyframe(new_kf, dssim, psnr, psnr_gs, render_time, kf_image_dir, kf_image_gt_dir, kf_image_undistorted_dir, kf_image_points_dir, kf_image_radii_dir);

                out_psnr_compare << idx << " " << vec_kfID.front() << " " << std::fixed << std::setprecision(10) << psnr_gs << std::endl;
                vec_kfID.pop_front();
                psnr_gsKf_vec.push_back(psnr_gs);
                dssim_Kf_vec.push_back(dssim);
            }
            }

            //save traj of all frame 
            // Eigen::Vector3d twc = pose.translation().cast<double>();
            // auto q = pose.unit_quaternion().cast<double>();
            out_traj << std::fixed << std::setprecision(6) << Traj[idx][0] << " " << setprecision(9)
                    << pose_trans(0) << " " << pose_trans(1) << " " << pose_trans(2) << " " 
                    << pose_rot.x() << " " << pose_rot.y() << " " << pose_rot.z() << " " << pose_rot.w() << std::endl;
        }
        float dssim_avg_value = (std::accumulate(dssim_vec.begin(), dssim_vec.end(), 0.0f)+
                               std::accumulate(dssim_Kf_vec.begin(), dssim_Kf_vec.end(), 0.0f)) / Traj.size();
        float ssim_Kf_avg = std::accumulate(dssim_Kf_vec.begin(), dssim_Kf_vec.end(), 0.0f) / dssim_Kf_vec.size();
        float ssim_test_avg = std::accumulate(dssim_vec.begin(), dssim_vec.end(), 0.0f) / dssim_vec.size();

        float psnr_gs_avg_value = (std::accumulate(psnr_gs_vec.begin(), psnr_gs_vec.end(), 0.0f)+
                               std::accumulate(psnr_gsKf_vec.begin(), psnr_gsKf_vec.end(), 0.0f)) / Traj.size();
        float psnr_gsKf_avg = std::accumulate(psnr_gsKf_vec.begin(), psnr_gsKf_vec.end(), 0.0f) / psnr_gsKf_vec.size();
        float psnr_test_avg = std::accumulate(psnr_gs_vec.begin(), psnr_gs_vec.end(), 0.0f) / psnr_gs_vec.size();

        std::cout << "[Render metric evaluation progress] at " << pGausMapper->getIteration()
                  << " Nums: " << psnr_gs_vec.size() << "/" << psnr_gsKf_vec.size() << "/" << idx << std::endl;
        std::cout << "\tPSNR_AVG:: \033[1;32m" << psnr_gs_avg_value << "\033[0m" 
                  <<"\tSSIM_AVG:: \033[1;32m" << dssim_avg_value << "\033[0m" << std::endl;
        std::cout << "\tPSNR_KF:: \033[1;32m" << psnr_gsKf_avg << "\033[0m"
                  << "\tSSIM_KF:: \033[1;32m" << ssim_Kf_avg << "\033[0m" << std::endl;
        std::cout << "\tPSNR in test:: \033[1;32m" << psnr_test_avg << "\033[0m"
                  << "\tSSIM in test:: \033[1;32m" << ssim_test_avg << "\033[0m" << std::endl;

        out_metric << "psnr_test: " << psnr_test_avg << std::endl
                   << "ssim_test: " << ssim_test_avg << std::endl
                   << "psnr_train: " << psnr_gsKf_avg << std::endl
                   << "ssim_train: " << ssim_Kf_avg << std::endl;
    }

    return 0;
}

void LoadImages(const std::filesystem::path &pathImageDir, std::vector<std::string> &vstrImageFilenamesRGB,
                std::vector<std::string> &vstrImageFilenamesD)
{
    for (const auto& imagePath : std::filesystem::directory_iterator(pathImageDir))
    {
        std::string name = imagePath.path().filename().string();
        if (name.rfind("frame", 0) == 0)
            vstrImageFilenamesRGB.push_back(imagePath.path().string());
        else if (name.rfind("depth", 0) == 0)
            vstrImageFilenamesD.push_back(imagePath.path().string());
        std::sort(vstrImageFilenamesRGB.begin(), vstrImageFilenamesRGB.end());
        std::sort(vstrImageFilenamesD.begin(), vstrImageFilenamesD.end());
    }
}

void saveTrackingTime(std::vector<float> &vTimesTrack, const std::string &strSavePath)
{
    std::ofstream out;
    out.open(strSavePath.c_str());
    std::size_t nImages = vTimesTrack.size();
    float totaltime = 0;
    for (int ni = 0; ni < nImages; ni++)
    {
        out << std::fixed << std::setprecision(4)
            << vTimesTrack[ni] << std::endl;
        totaltime += vTimesTrack[ni];
    }

    // std::sort(vTimesTrack.begin(), vTimesTrack.end());
    // out << "-------" << std::endl;
    // out << std::fixed << std::setprecision(4)
    //     << "median tracking time: " << vTimesTrack[nImages / 2] << std::endl;
    // out << std::fixed << std::setprecision(4)
    //     << "mean tracking time: " << totaltime / nImages << std::endl;

    out.close();
}

void saveGpuPeakMemoryUsage(std::filesystem::path pathSave)
{
    namespace c10Alloc = c10::cuda::CUDACachingAllocator;
    c10Alloc::DeviceStats mem_stats = c10Alloc::getDeviceStats(0);

    c10Alloc::Stat reserved_bytes = mem_stats.reserved_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)];
    float max_reserved_MB = reserved_bytes.peak / (1024.0 * 1024.0);

    c10Alloc::Stat alloc_bytes = mem_stats.allocated_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)];
    float max_alloc_MB = alloc_bytes.peak / (1024.0 * 1024.0);

    std::ofstream out(pathSave);
    out << "Peak reserved (MB): " << max_reserved_MB << std::endl;
    out << "Peak allocated (MB): " << max_alloc_MB << std::endl;
    out.close();
}