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

#pragma once

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/cudaimgproc.hpp>
#include <torch/torch.h>
#include "third_party/tinyply/tinyply.h"

#include "third_party/tinyply/tinyply.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace tensor_utils
{

inline void deleter(void* arg) {}


inline torch::Tensor cvMat2TorchTensor_Float32(
    cv::Mat& mat,
    torch::DeviceType device_type)
{
    torch::Tensor mat_tensor, tensor;

    switch (mat.channels())
    {
    case 1:
    {
        mat_tensor = torch::from_blob(mat.data, {mat.rows, mat.cols});
        tensor = mat_tensor.clone().to(device_type);
    }
    break;

    case 3:
    {
        mat_tensor = torch::from_blob(mat.data, {mat.rows, mat.cols, mat.channels()});
        tensor = mat_tensor.clone().to(device_type);
        tensor = tensor.permute({2, 0, 1});
    }
    break;
    
    default:
        std::cerr << "The mat has unsupported number of channels!" << std::endl;
    break;
    }

    return tensor.contiguous();
}

inline cv::Mat torchTensor2CvMat_Float32(torch::Tensor& tensor)
{
    cv::Mat mat;
    torch::Tensor mat_tensor = tensor.clone();

    switch (mat_tensor.ndimension())
    {
    case 2:
    {
        mat = cv::Mat(mat_tensor.size(0),
                      mat_tensor.size(1),
                      CV_32FC1,
                      mat_tensor.data_ptr<float>());
    }
    break;

    case 3:
    {
        mat_tensor = mat_tensor.detach().permute({1, 2, 0}).contiguous();
        mat_tensor = mat_tensor.to(torch::kCPU);
        mat = cv::Mat(mat_tensor.size(0),
                      mat_tensor.size(1),
                      CV_32FC3,
                      mat_tensor.data_ptr<float>());
    }
    break;
    
    default:
        std::cerr << "The tensor has unsupported number of dimensions!" << std::endl;
    break;
    }

    return mat.clone();
}

inline torch::Tensor cvGpuMat2TorchTensor_Float32(cv::cuda::GpuMat& mat)
{
    torch::Tensor mat_tensor, tensor;
    int64_t step = mat.step / sizeof(float);

    switch (mat.channels())
    {
    case 1:
    {
        std::vector<int64_t> strides = {step, 1};
        mat_tensor = torch::from_blob(
            mat.data,
            {mat.rows, mat.cols},
            strides,
            deleter,
            torch::TensorOptions().device(torch::kCUDA));
        tensor = mat_tensor.clone();
    }
    break;

    case 3:
    {
        std::vector<int64_t> strides = {step, static_cast<int64_t>(mat.channels()), 1};
        mat_tensor = torch::from_blob(
            mat.data,
            {mat.rows, mat.cols, mat.channels()},
            strides,
            deleter,
            torch::TensorOptions().device(torch::kCUDA));
        tensor = mat_tensor.clone().permute({2, 0, 1});
    }
    break;
    
    default:
        std::cerr << "The mat has unsupported number of channels!" << std::endl;
    break;
    }

    return tensor.contiguous();
}

inline cv::cuda::GpuMat torchTensor2CvGpuMat_Float32(torch::Tensor& tensor)
{
    cv::cuda::GpuMat mat;
    torch::Tensor mat_tensor = tensor.clone();

    switch (mat_tensor.ndimension())
    {
    case 2:
    {
        mat = cv::cuda::GpuMat(mat_tensor.size(0),
                               mat_tensor.size(1),
                               CV_32FC1,
                               mat_tensor.data_ptr<float>());
    }
    break;

    case 3:
    {
        mat_tensor = mat_tensor.detach().permute({1, 2, 0}).contiguous();
        mat = cv::cuda::GpuMat(mat_tensor.size(0),
                               mat_tensor.size(1),
                               CV_32FC3,
                               mat_tensor.data_ptr<float>());
    }
    break;

    default:
        std::cerr << "The tensor has unsupported number of channels!" << std::endl;
    break;
    }

    return mat.clone();
}

inline torch::Tensor EigenMatrix2TorchTensor(
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix,
    torch::DeviceType device_type = torch::kCUDA)
{
    auto eigen_matrix_T = eigen_matrix;
    eigen_matrix_T.transposeInPlace();
    torch::Tensor tensor = torch::from_blob(
        eigen_matrix_T.data(),
        {eigen_matrix.rows(), eigen_matrix.cols()},
        torch::TensorOptions().dtype(torch::kFloat)
    ).clone();

    tensor = tensor.to(device_type);
    return tensor;
}

inline torch::Tensor MatrixInverse2TorchTensor(
    torch::Tensor world_view_transform_,
    torch::DeviceType device_type = torch::kCUDA)
{

    Eigen::Matrix4f eigen_matrix;
    auto accessor = world_view_transform_.accessor<float, 2>();
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            eigen_matrix(j, i) = accessor[i][j];
        }
    }

    Eigen::Matrix4f eigen_inverse_matrix = eigen_matrix.inverse();

    auto eigen_matrix_T = eigen_inverse_matrix;
    eigen_matrix_T.transposeInPlace();
    torch::Tensor tensor = torch::from_blob(
        eigen_matrix_T.data(),
        {eigen_matrix.rows(), eigen_matrix.cols()},
        torch::TensorOptions().dtype(torch::kFloat)
    ).clone();

    tensor = tensor.to(device_type);
    return tensor;
}

inline void saveTensor2Txt(const torch::Tensor& tensor, const std::string& filename, int precision) {
    auto tensor_cpu = tensor.clone().to(torch::kCPU);

    auto sizes = tensor_cpu.sizes();

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file " << filename << " for writing." << std::endl;
        return;
    }

    file << std::fixed << std::setprecision(precision);

    if (tensor_cpu.dim() == 1) {
        if (tensor_cpu.scalar_type() == torch::kFloat32) {
            std::vector<float> data(tensor_cpu.numel());
            std::memcpy(data.data(), tensor_cpu.data_ptr(), data.size() * sizeof(float));
            for (size_t i = 0; i < sizes[0]; ++i) {
                file << data[i] << "\n";
            }
        } else if (tensor_cpu.scalar_type() == torch::kInt32) {
            std::vector<int> data(tensor_cpu.numel());
            std::memcpy(data.data(), tensor_cpu.data_ptr(), data.size() * sizeof(int));
            for (size_t i = 0; i < sizes[0]; ++i) {
                file << data[i] << "\n";
            }
        } else if (tensor_cpu.scalar_type() == torch::kBool) {
            std::vector<bool> data(tensor_cpu.numel());
            auto tensor_accessor = tensor_cpu.accessor<bool, 1>();
            for (size_t i = 0; i < sizes[0]; ++i) {
                data[i] = tensor_accessor[i];
            }
            for (size_t i = 0; i < sizes[0]; ++i) {
                file << data[i] << "\n";
            }
        } else {
            std::cerr << "Unsupported tensor data type: " << tensor_cpu.scalar_type() << std::endl;
            file.close();
            return;
        }
    } else if (tensor_cpu.dim() == 2) {
        if (tensor_cpu.scalar_type() == torch::kFloat32) {
            std::vector<float> data(tensor_cpu.numel());
            std::memcpy(data.data(), tensor_cpu.data_ptr(), data.size() * sizeof(float));
            for (size_t i = 0; i < sizes[0]; ++i) {
                for (size_t j = 0; j < sizes[1]; ++j) {
                    file << data[i * sizes[1] + j];
                    if (j < sizes[1] - 1) {
                        file << " ";
                    }
                }
                file << "\n";
            }
        } else if (tensor_cpu.scalar_type() == torch::kInt32) {
            std::vector<int> data(tensor_cpu.numel());
            std::memcpy(data.data(), tensor_cpu.data_ptr(), data.size() * sizeof(int));
            for (size_t i = 0; i < sizes[0]; ++i) {
                for (size_t j = 0; j < sizes[1]; ++j) {
                    file << data[i * sizes[1] + j];
                    if (j < sizes[1] - 1) {
                        file << " ";
                    }
                }
                file << "\n";
            }
        } else if (tensor_cpu.scalar_type() == torch::kBool) {
            std::vector<bool> data(tensor_cpu.numel());
            auto tensor_accessor = tensor_cpu.accessor<bool, 2>();
            for (size_t i = 0; i < sizes[0]; ++i) {
                for (size_t j = 0; j < sizes[1]; ++j) {
                    data[i * sizes[1] + j] = tensor_accessor[i][j];
                }
            }
            for (size_t i = 0; i < sizes[0]; ++i) {
                for (size_t j = 0; j < sizes[1]; ++j) {
                    file << data[i * sizes[1] + j];
                    if (j < sizes[1] - 1) {
                        file << " ";
                    }
                }
                file << "\n";
            }
        } else {
            std::cerr << "Unsupported tensor data type: " << tensor_cpu.scalar_type() << std::endl;
            file.close();
            return;
        }
    } else {
        std::cerr << "Unsupported tensor dimensions: " << tensor_cpu.dim() << std::endl;
        file.close();
        return;
    }

    file.close();

    std::cout << "Tensor data saved to " << filename << std::endl;
}

inline torch::Tensor loadTensorFromPly(std::filesystem::path ply_path, std::string vertex)
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

    std::shared_ptr<tinyply::PlyData> offset_denom;

    try { offset_denom = ply_file.request_properties_from_element("vertex", { vertex }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    ply_file.read(instream_binary);

    if (offset_denom)  std::cout << "\tRead " << offset_denom->count  << " total " << vertex  << std::endl;

    const std::size_t n_offset_denom_bytes = offset_denom->buffer.size_bytes();
    std::vector<float> offset_denom_vector(offset_denom->count * 1);
    std::memcpy(offset_denom_vector.data(), offset_denom->buffer.get(), n_offset_denom_bytes);

    torch::Tensor offset_denom_ = torch::from_blob(
                                    offset_denom_vector.data(), {int(offset_denom->count)},
                                    torch::TensorOptions().dtype(torch::kBool))
                                    .to(torch::kCUDA);

    return offset_denom_;
}


inline torch::Tensor loadTensorFromTxt(const std::string& file_path, int rows, int cols, torch::Dtype dtype) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + file_path);
    }

    torch::Tensor tensor;
    if (dtype == torch::kBool)
    {
        std::vector<int> data;
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            int value;
            while (iss >> value) {
                data.push_back(value);
            }
        }

        file.close();

        torch::Tensor tensor_int;
        if (cols == 0)
            tensor_int = torch::from_blob(data.data(), {rows}, torch::TensorOptions().dtype(torch::kInt32)).to(torch::kCUDA).clone();
        else if (cols > 0 )
            tensor_int = torch::from_blob(data.data(), {rows, cols}, torch::TensorOptions().dtype(torch::kInt32)).to(torch::kCUDA).clone();
        else 
            throw std::runtime_error("cols shoud > 0!");
        tensor = tensor_int.to(dtype);
    }
    else if (dtype == torch::kFloat32)
    {

        std::vector<float> data;
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            float value;
            while (iss >> value) {
                data.push_back(value);
            }
        }

        file.close();

        if (cols == 0)
            tensor = torch::from_blob(data.data(), {rows}, dtype).to(torch::kCUDA).clone();
        else if (cols > 0 )
            tensor = torch::from_blob(data.data(), {rows, cols}, dtype).to(torch::kCUDA).clone();
        else 
            throw std::runtime_error("cols shoud > 0!");
    }
    else 
    throw std::runtime_error("unsupported tensor type!");
    
    return tensor;
}

}
