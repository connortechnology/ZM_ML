# syntax=docker/dockerfile:experimental

# Build OpenCV with CUDA & cuDNN support as an image strictly to pull install from
# This allows for easier up/down grading

ARG S6ARCH=x86_64

# How to get OpenCV source code;
# BRANCH (branch accepts branch name or SHA) or RELEASE (release accepts version number)
ARG OPENCV_METHOD=branch
# OpenCV Branch, SHA, or release version number (i.e. 4.7.0)
ARG OPENCV_VER=4.x
# OpenCV Contrib Branch, SHA, or release version number (i.e. 4.7.0)
ARG OPENCV_CONTRIB_VER=4.x

# A minimum of 6.1 Compute Cabability required
# CHECK https://developer.nvidia.com/cuda-gpus#compute
# Covers 1050 thru to 4090 includes TITAN V/X/XP/RTX QUADRO H100 K80 etc.
# JETSON are seperate
# todo: add ARG to select CUDA_ARCH_BIN or add them together?
ARG CUDA_ARCH_BIN=6.0,6.1,7.0,7.5,8.0,8.6,8.9,9.0
ARG CUDA_ARCH_BIN_JETSON=6.2,7.2,8.7

######## CUDA 12 / cuDNN 8 ########
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 as opencv

ARG S6ARCH
ARG OPENCV_METHOD
ARG OPENCV_VER
ARG OPENCV_CONTRIB_VER
ARG CUDA_ARCH_BIN
ARG CUDA_ARCH_BIN_JETSON
# No post install steps
ARG DEBIAN_FRONTEND=noninteractive

ENV LANG en_US.utf8 \
    S6ARCH=${S6ARCH} \
    OPENCV_METHOD=${OPENCV_METHOD} \
    OPENCV_VER=${OPENCV_VER} \
    OPENCV_CONTRIB_VER=${OPENCV_CONTRIB_VER} \
    CUDA_ARCH_BIN=${CUDA_ARCH_BIN} \
    CUDA_ARCH_BIN_JETSON=${CUDA_ARCH_BIN_JETSON}

RUN env;exit 1

COPY ./docker/opencv/build_opencv.sh /tmp/build_opencv

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/apt/lists,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    set -x \
  && chmod +x /tmp/build_opencv \
  \
  && apt-get update \
  && apt-get install -y --no-install-recommends --fix-missing \
    software-properties-common \
    apt-utils \
    ca-certificates gnupg \
    git wget curl \
    gettext-base time \
    locales \
    cmake pkg-config build-essential \
    \
    # FFMPEG Libs
    libavdevice-dev \
    libavfilter-dev  \
    libswresample-dev \
    libavutil-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libpng-dev \
    libjpeg-dev \
    libopenexr-dev \
    libtiff-dev \
    libwebp-dev \
    libtiff-dev \
    libv4l-dev \
    libx264-dev \
    libatlas-base-dev \
    gfortran \
    libtbb2 \
    libtbb-dev \
    libxine2-dev \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
    libeigen3-dev

RUN localedef \
        -i en_US \
        -c -f UTF-8 \
        -A /usr/share/locale/locale.alias \
        en_US.UTF-8 \
  && update-locale LANG=en_US.UTF-8

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/apt/lists,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    set -x \
  && add-apt-repository ppa:deadsnakes/ppa \
  && apt-get install -y --no-install-recommends \
      python3.9-dev python3.9-distutils \
  && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
  && wget -qO- https://bootstrap.pypa.io/get-pip.py | python3.9 \
  && update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.9 1 \
  && python3 -m pip install numpy

RUN set -x /tmp/build_opencv

LABEL description="OpenCV development image built with CUDA 12.1.1 & cuDNN 8 including Compute Capabilitie(s) [$CUDA_ARCH_BIN] support. Used to build ZM ML containers."

CMD ["/bin/bash"]