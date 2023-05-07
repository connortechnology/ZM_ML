#!/usr/bin/env bash

echo "OpenCV GitHub method set to:: ${OPENCV_METHOD}"
echo "OpenCV GitHub version set to:: ${OPENCV_VERSION}"
echo "OpenCV GitHub contrib version set to:: ${OPENCV_CONTRIB_VER}"

if [[ -z "$OPENCV_VERSION" ]]; then
  echo "DLIB_VERSIO not set, exiting..."
  exit 1
  elif [[ -z "$OPENCV_METHOD" ]]; then
    echo "DLIB_METHOD not set, exiting..."
    exit 1
  else
    echo "DLIB_VERSION: $OPENCV_VERSION"
    echo "DLIB_METHOD: $OPENCV_METHOD"
fi


if [[ "$OPENCV_METHOD" == "release"  ]]; then
  echo "DLib release method requested, downloading tagged release $OPENCV_VERSION from github..."
  wget -c -q https://github.com/davisking/dlib/archive/refs/tags/v${DLIB_VERSION}.tar.gz -O dlib.tgz
  tar xf dlib.tgz
  mv dlib-*/dlib dlib
  rm dlib.tgz
  elif [[ "$OPENCV_METHOD" == "branch"  ]]; then
    git clone --single-branch --branch "${DLIB_VERSION}" https://github.com/davisking/dlib /dlib
  else
    echo "Unknown DLIB_METHOD: $OPENCV_METHOD, exiting..."
    exit 1
fi




mkdir -p /opt/opencv/export /opt/opencv/python_bindings /tmp/opencv
git clone https://github.com/opencv/opencv.git /tmp/opencv/src
mkdir /tmp/opencv/src/build
git clone https://github.com/opencv/opencv_contrib.git /tmp/opencv_contrib
cd /tmp/opencv/src
git checkout "${OPENCV_VER}"
git submodule update --init --recursive
cd /opt/opencv_contrib
git checkout "${OPENCV_CONTRIB_VER}"
git submodule update --init --recursive
cd /opt/opencv/src/build

time cmake \
  -D BUILD_DOCS=OFF \
  -D BUILD_EXAMPLES=OFF \
  -D BUILD_PERF_TESTS=OFF \
  -D BUILD_TESTS=OFF \
  -D BUILD_opencv_python2=OFF \
  -D BUILD_opencv_python3=ON \
  -D HAVE_opencv_python3=ON \
  -D HAVE_opencv_python2=OFF \
  -D CMAKE_BUILD_TYPE=RELEASE \
  -D OPENCV_PYTHON3_INSTALL_PATH=/opt/opencv/python_bindings \
  -D CMAKE_INSTALL_PREFIX=/opt/opencv/export \
  -D PYTHON3_NUMPY_INCLUDE_DIR=/usr/lib/python3.9/dist-packages/numpy/core/include \
  -D CMAKE_INSTALL_TYPE=RELEASE \
  -D FORCE_VTK=ON \
  -D INSTALL_C_EXAMPLES=OFF \
  -D INSTALL_PYTHON_EXAMPLES=OFF \
  -D OPENCV_GENERATE_PKGCONFIG=ON \
  -D WITH_CSTRIPES=ON \
  -D WITH_EIGEN=ON \
  -D WITH_GDAL=ON \
  -D WITH_GSTREAMER=ON \
  -D WITH_GSTREAMER_0_10=OFF \
  -D WITH_GTK=ON \
  -D WITH_IPP=ON \
  -D WITH_OPENCL=ON \
  -D WITH_OPENMP=ON \
  -D WITH_TBB=ON \
  -D WITH_V4L=ON \
  -D WITH_WEBP=ON \
  -D WITH_XINE=ON \
  -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
  -D OPENCV_ENABLE_NONFREE=ON \
  -D CUDA_ARCH_BIN=${CUDA_ARCH_BIN} \
  -D WITH_CUDA=ON \
  -D WITH_CUDNN=ON \
  -D OPENCV_DNN_CUDA=ON \
  -D ENABLE_FAST_MATH=1 \
  -D CUDA_FAST_MATH=1 \
  -D WITH_CUBLAS=1 \
  -D PYTHON3_EXECUTABLE=/usr/bin/python3.9 \
  -D PYTHON3_INCLUDE_DIR=/usr/include/python3.9 \
  -D PYTHON3_INCLUDE_DIR2=/usr/include/x86_64-linux-gnu/python3.9 \
  -D PYTHON3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.9.so \
  ..

time make -j${nproc} install

echo "OpenCV Python bindings installed to:: /tmp/opencv_python_bindings"
echo "OpenCV installed to:: /tmp/opencv_export"

