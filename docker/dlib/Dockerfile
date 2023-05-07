# DLIB_METHOD is one of 2 options; branch or release
# branch = Commit SHA / Branch name
# release = tagged release, i.e. 19.22    19.23     etc.
ARG DLIB_METHOD=branch
# branch name (master), commit SHA (3e12d52), or release version number (19.22)
ARG DLIB_VERSION=master
ARG S6ARCH=x86_64
# Use AVX instructions
ARG DLIB_USE_AVX=1
# Use CUDA/cuDNN GPU Acceleration
ARG DLIB_USE_CUDA=1
# What Nvidia Compute Capabilities to build for
# This should take care of most modern Nvidia GPUs capable of ML
ARG CUDA_ARCH_BIN=6.0,6.1,7.0,7.5,8.0,8.6,8.9,9.0
# Jetson Compute Capabilities - add these to CUDA_ARCH_BIN if building for Jetson
ARG CUDA_ARCH_BIN_JETSON=6.2,7.2,8.7

FROM baudneo/opencv:testing as opencv
COPY ./docker/dlib/build_dlib.sh /tmp/build_dlib
# DONT FORGET to 'bring' the external build args into the build stage
ARG DLIB_METHOD
ARG DLIB_VERSION
ARG DLIB_USE_AVX
ARG DLIB_USE_CUDA
ARG CUDA_ARCH_BIN
ARG CUDA_ARCH_BIN_JETSON
ARG S6ARCH

# #7 34.19 -- Could NOT find CUDA (missing: CUDA_TOOLKIT_ROOT_DIR CUDA_NVCC_EXECUTABLE CUDA_INCLUDE_DIRS CUDA_CUDART_LIBRARY) (Required is at least version "7.5")
  ##7 34.19 -- Found CUDA, but CMake was unable to find the cuBLAS libraries that should be part of every basic CUDA install. Your CUDA install is somehow broken or incomplete. Since cuBLAS is required for dlib to use CUDA we won't use CUDA.
  ##7 34.19 -- DID NOT FIND CUDA

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/apt/lists,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    set -x \
    && chmod +x /tmp/build_dlib \
    && /tmp/build_dlib