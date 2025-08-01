cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(photo-slam LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    # set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

set(CAFFE2_USE_CUDNN on)

set(Boost_LIBRARY_DIR /usr/lib/x86_64-linux-gnu/)
# set(PCL_DIR /home/lzy/dependency/pcl/build)
find_package(PCL REQUIRED)
# Uncomment the following line if you want to solve CMake Warning "Failed to compute shorthash for libnvrtc.so"
# set(PYTHON_EXECUTABLE "/usr/bin/python3")
# set(Torch_DIR /home/lzy/workingspace/Photo-SLAM/Torch/libtorch/share/cmake/Torch)
set(Torch_DIR /home/lzy/dependency/libtorch/share/cmake/Torch)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread")

set(TorchScatter_DIR /usr/local/share/cmake/TorchScatter)
find_package(TorchScatter REQUIRED)
MESSAGE("TorchScatter:")
message(STATUS "TorchScatter include dir: ${TorchScatter_INCLUDE_DIR}")
message(STATUS "TorchScatter library: ${TorchScatter_LIBRARIES}")
message(STATUS "TorchScatter def: ${TorchScatter_DEFINITIONS}")

# set(ENV{CUDA_LAUNCH_BLOCKING} 1)
# set(ENV{TORCH_USE_CUDA_DSA} 1)

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(jsoncpp REQUIRED)

find_package(OpenGL REQUIRED)
find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)

find_package(realsense2)

set(ORB_SLAM3_SOURCE_DIR "${PROJECT_SOURCE_DIR}/ORB-SLAM3")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OpenMP found")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${ORB_SLAM3_SOURCE_DIR}
    ${ORB_SLAM3_SOURCE_DIR}/include
    ${ORB_SLAM3_SOURCE_DIR}/include/CameraModels
    ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/Sophus
    ${EIGEN3_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    ${TorchScatter_INCLUDE_DIRS})

##################################################################################
##  Build the gaussian splatting component libraries to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(simple_knn SHARED
    third_party/simple-knn/simple_knn.cu
    third_party/simple-knn/simple_knn.h
    third_party/simple-knn/spatial.cu
    third_party/simple-knn/spatial.h)
# target_compile_features(simple_knn PUBLIC cxx_std_17)
target_link_libraries(simple_knn "${TORCH_LIBRARIES}")

add_library(cuda_rasterizer SHARED
    include/operate_points.h
    src/operate_points.cu
    include/rasterize_points.h
    src/rasterize_points.cu
    include/stereo_vision.h
    src/stereo_vision.cu
    cuda_rasterizer/auxiliary.h
    cuda_rasterizer/backward.cu
    cuda_rasterizer/backward.h
    cuda_rasterizer/config.h
    cuda_rasterizer/forward.cu
    cuda_rasterizer/forward.h
    cuda_rasterizer/operate_points.h
    cuda_rasterizer/rasterizer.h
    cuda_rasterizer/rasterizer_impl.cu
    cuda_rasterizer/rasterizer_impl.h
    cuda_rasterizer/stereo_vision.h)
set_target_properties(cuda_rasterizer PROPERTIES CUDA_ARCHITECTURES "75;86")
# target_compile_features(cuda_rasterizer PUBLIC cxx_std_17)
target_include_directories(cuda_rasterizer PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(cuda_rasterizer
    glm::glm
    "${TORCH_LIBRARIES}"
    Eigen3::Eigen)

##################################################################################
##  Build the ImGui library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(imgui SHARED
    viewer/imgui/imconfig.h
    viewer/imgui/imgui_demo.cpp
    viewer/imgui/imgui_draw.cpp
    viewer/imgui/imgui_impl_glfw.cpp
    viewer/imgui/imgui_impl_glfw.h
    viewer/imgui/imgui_impl_opengl3_loader.h
    viewer/imgui/imgui_impl_opengl3.cpp
    viewer/imgui/imgui_impl_opengl3.h
    viewer/imgui/imgui_internal.h
    viewer/imgui/imgui_tables.cpp
    viewer/imgui/imgui_widgets.cpp
    viewer/imgui/imgui.cpp
    viewer/imgui/imgui.h
    viewer/imgui/imstb_rectpack.h
    viewer/imgui/imstb_textedit.h
    viewer/imgui/imstb_truetype.h)
target_link_libraries(imgui 
# glfw 
OpenGL::GL
${GLFW3_LIBRARY}
)

##################################################################################
##  Build the gaussian mapper library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(gaussian_mapper SHARED
    third_party/tinyply/tinyply.h
    third_party/tinyply/tinyply.cpp
    include/gaussian_keyframe.h
    include/gaussian_model.h
    include/gaussian_parameters.h
    include/gaussian_rasterizer.h
    include/gaussian_renderer.h
    include/gaussian_scene.h
    include/gaussian_trainer.h
    include/gaussian_mapper.h
    include/general_utils.h
    include/graphics_utils.h
    include/loss_utils.h
    include/sh_utils.h
    include/tensor_utils.h
    include/camera.h
    include/point_cloud.h
    include/point2d.h
    include/point3d.h
    include/types.h
    include/embedding.h
    include/mlp.h
    src/gaussian_keyframe.cpp
    src/gaussian_model.cpp
    src/gaussian_parameters.cpp
    src/gaussian_rasterizer.cpp
    src/gaussian_renderer.cpp
    src/gaussian_scene.cpp
    src/gaussian_trainer.cpp
    src/gaussian_mapper.cpp
    src/embedding.cpp
    src/mlp.cpp)
target_link_libraries(gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
    ${OpenCV_LIBRARIES}
    jsoncpp_lib
    "${TORCH_LIBRARIES}"
    Eigen3::Eigen
    simple_knn
    cuda_rasterizer
    ${PCL_LIBRARIES}
    /usr/local/lib/libtorchscatter.so
    ${OpenMP_CXX_FLAGS})

##################################################################################
##  Build the viewer library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(gaussian_viewer SHARED
    viewer/drawer_utils.h
    viewer/imgui_viewer.cpp
    viewer/imgui_viewer.h
    viewer/map_drawer.cpp
    viewer/map_drawer.h)
target_link_libraries(gaussian_viewer
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
    gaussian_mapper
    imgui
    ${OpenCV_LIBRARIES}
    jsoncpp_lib
    "${TORCH_LIBRARIES}"
    glm::glm
    glfw
    OpenGL::GL)

##################################################################################
##  Build the test examples to ${PROJECT_SOURCE_DIR}/bin
##################################################################################

# This is a C++ libtorch implementation of gaussian-splatting (https://github.com/graphdeco-inria/gaussian-splatting)
add_executable(train_colmap examples/train_colmap.cpp)
target_link_libraries(train_colmap
    gaussian_viewer
    gaussian_mapper
    # /usr/local/lib/libtorchscatter.so
    )

# add_executable(view_result examples/view_result.cpp)
# target_link_libraries(view_result
#     gaussian_viewer
#     gaussian_mapper
#     # ${TorchScatter_LIBRARIES}
#     )

##################################################################################
##  Build the mapping examples to ${PROJECT_SOURCE_DIR}/bin
##################################################################################

# Replica Monocular
add_executable(replica_mono 
    examples/replica_mono.cpp
    examples/example_utils.h)
target_link_libraries(replica_mono
    gaussian_viewer    
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

# Replica RGBD
add_executable(replica_rgbd 
    examples/replica_rgbd.cpp
    examples/example_utils.h)
target_link_libraries(replica_rgbd
    gaussian_viewer    
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

# TUM Monocular
add_executable(tum_mono 
    examples/tum_mono.cpp
    examples/example_utils.h)
target_link_libraries(tum_mono
    gaussian_viewer    
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

# TUM RGBD
add_executable(tum_rgbd 
    examples/tum_rgbd.cpp
    examples/example_utils.h)
target_link_libraries(tum_rgbd
    gaussian_viewer    
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

# # Scannetpp RGBD
# add_executable(scannetpp_rgbd examples/scannetpp_rgbd.cpp)
# target_link_libraries(scannetpp_rgbd
#     gaussian_viewer    
#     gaussian_mapper
#     ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

# EuRoC Stereo
add_executable(euroc_stereo examples/euroc_stereo.cpp)
target_link_libraries(euroc_stereo
    gaussian_viewer    
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

# Scannet Mono //效果很差
add_executable(scannet_mono examples/scannet_mono.cpp)
target_link_libraries(scannet_mono
    gaussian_viewer    
    gaussian_mapper
    "${TORCH_LIBRARIES}"
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

# # Scannet RGBD
add_executable(scannet_rgbd examples/scannet_rgbd.cpp)
target_link_libraries(scannet_rgbd
    gaussian_viewer    
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)


# # KITTI Mono
# add_executable(kitti_mono examples/kitti_mono.cpp)
# target_link_libraries(kitti_mono
#     gaussian_viewer    
#     gaussian_mapper
#     ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

# # KITTI Stereo
# add_executable(kitti_stereo examples/kitti_stereo.cpp)
# target_link_libraries(kitti_stereo
#     gaussian_viewer    
#     gaussian_mapper
#     ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)

# ##################################################################################
# ##  Build the mapping examples to ${PROJECT_SOURCE_DIR}/bin
# ##################################################################################

# # Intel Realsense
# if(realsense2_FOUND)
#     add_executable(realsense_rgbd examples/realsense_rgbd.cpp)
#     target_include_directories(realsense_rgbd PUBLIC
#         ${realsense_INCLUDE_DIR})
#     target_link_libraries(realsense_rgbd
#         gaussian_viewer    
#         gaussian_mapper
#         ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
#         ${realsense2_LIBRARY})
# endif()