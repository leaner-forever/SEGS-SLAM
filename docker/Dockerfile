FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget

# gcc
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y

RUN apt-get install -y gcc-11 g++-11

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

# dependency
RUN apt-get install -y \
    git \
    build-essential \
    sudo \
    libeigen3-dev \
    libboost-all-dev \
    libjsoncpp-dev \
    libopengl-dev \
    mesa-utils \
    libglfw3-dev \
    libglm-dev \
    python3-pip \
    python3-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    curl \
    zip \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libswresample-dev \
    libssl-dev \
    # Install dependencies for PCL
    libflann-dev \ 
    libusb-1.0-0-dev \
    liblz4-dev \
    libgtest-dev \
    libopenni-dev \
    libgdal-dev \
    libosmesa6-dev \
    && rm -rf /var/lib/apt/lists/*

# cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-Linux-x86_64.sh -O /cmake-3.22.1.sh && \
    chmod +x /cmake-3.22.1.sh && \
    /cmake-3.22.1.sh --skip-license --prefix=/usr/local

# pcl1.14.1
# 安装 VTK 9.3
RUN mkdir /vtk && cd /vtk && \
    wget https://github.com/Kitware/VTK/archive/refs/tags/v9.3.0.zip -O vtk-9.3.0.zip && \
    unzip vtk-9.3.0.zip && \
    rm vtk-9.3.0.zip

RUN cd /vtk/VTK-9.3.0 && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DVTK_RENDERING_BACKEND=OpenGL2 \
        -DVTK_USE_X=OFF \
        -DVTK_USE_COCOA=OFF \
        -DVTK_USE_OPENGL2=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DVTK_USE_SYSTEM_GLFW=ON \
        -DBUILD_EXAMPLES=OFF \
        -DVTK_OPENGL_HAS_OSMESA=ON \
        -DVTK_USE_OSMESA=OFF \ 
          .. && \
    make -j$(nproc) && \
    make install

# Download PCL 1.14.1 and compile it
RUN wget https://github.com/PointCloudLibrary/pcl/archive/refs/tags/pcl-1.14.1.tar.gz -O pcl-1.14.1.tar.gz && \
    tar -xvzf pcl-1.14.1.tar.gz && \
    rm pcl-1.14.1.tar.gz

RUN apt-get update && apt-get install -y \
    libx11-dev \
    qt5-qmake \
    qtbase5-dev \
    freeglut3-dev \
    libpcap-dev \
    && rm -rf /var/lib/apt/lists/*


# Build PCL
RUN cd pcl-pcl-1.14.1 && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
        -DWITH_PCL_VISUALIZATION=OFF \
        -DBUILD_visualization=OFF \
        -DBUILD_tools=OFF \
        .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# libtorch
RUN cd / && \
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip -O libtorch-cu118.zip && \
unzip libtorch-cu118.zip && rm libtorch-cu118.zip

ENV Torch_DIR /libtorch/share/cmake/Torch

#TorchScatter 2.1.2
RUN cd / && \
    wget https://files.pythonhosted.org/packages/f5/ab/2a44ecac0f891dd0d765fc59ac8d277c6283a31907626560e72685df2ed6/torch_scatter-2.1.2.tar.gz && \
    tar -xvzf torch_scatter-2.1.2.tar.gz && \
    rm torch_scatter-2.1.2.tar.gz

RUN cd torch_scatter-2.1.2 && \
    mkdir build && \
    cd build && \
    # Add -DWITH_CUDA=on support for CUDA support
    cmake -DCMAKE_PREFIX_PATH=/libtorch/share/cmake/Torch \
        -DWITH_CUDA=ON  \
        ..  && \
    make  && \
    make install && \
    ldconfig

# opencv
RUN mkdir /opencv && cd /opencv && \
	wget https://github.com/opencv/opencv/archive/refs/tags/4.8.0.zip -O opencv-4.8.0.zip && \
    wget https://github.com/opencv/opencv_contrib/archive/refs/tags/4.8.0.zip -O opencv_contrib-4.8.0.zip && \
    unzip opencv-4.8.0.zip && \
    unzip opencv_contrib-4.8.0.zip && \
    rm opencv-4.8.0.zip && rm opencv_contrib-4.8.0.zip
    
RUN mkdir /opencv/opencv-4.8.0/build && cd /opencv/opencv-4.8.0/build && \
	cmake -DCMAKE_BUILD_TYPE=RELEASE \
          -DWITH_CUDA=ON \
          -DWITH_CUDNN=ON \
          -DOPENCV_DNN_CUDA=ON \
          -DWITH_NVCUVID=ON \
          -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.8 \
          -DOPENCV_EXTRA_MODULES_PATH=/opencv/opencv_contrib-4.8.0/modules \
          -DBUILD_TIFF=ON \
          -DBUILD_ZLIB=ON \
          -DBUILD_JASPER=ON \
          -DBUILD_JPEG=ON \
          -DWITH_FFMPEG=ON \
          .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig


# repo
RUN git clone https://github.com/leaner-forever/SEGS-SLAM /SEGS-SLAM

WORKDIR /SEGS-SLAM
