# syntax=docker/dockerfile:experimental
ARG S6_ARCH=x86_64
ARG ZMML_VERSION=master
ARG OPENCV_VERSION=4.x
ARG DLIB_VERSION=19.24
ARG ALPR_VERSION=master
# I think a minimum of 6.1 Compute Cabability required - these are GeForce cards
# CHECK https://developer.nvidia.com/cuda-gpus#compute
# 6.1 = 1050 thru to 1080ti includes TITAN X and TITAN XP
# 7.0 = TITAN V
# 7.5 = 1650 thru to 2080ti including TITAN RTX
# 8.6 = 3050 thru to 3090
#ARG CUDA_ARCH_BIN=6.1,7.5
ARG CUDA_ARCH_BIN=6.0,6.1,7.0,7.5,8.0,8.6
ARG MLAPI_PORT=5000


#####################################################################
#                                                                   #
# Convert rootfs to LF using dos2unix                               #
# Alleviates issues when git uses CRLF on Windows                   #
#                                                                   #
#####################################################################
FROM alpine:latest as rootfs-converter
WORKDIR /rootfs

RUN set -x \
    && apk add --repository=http://dl-cdn.alpinelinux.org/alpine/edge/community/ \
        dos2unix

COPY docker/rootfs .
RUN set -x \
    && find . -type f -print0 | xargs -0 -n 1 -P 4 dos2unix \
    && chmod -R +x *

#####################################################################
#                                                                   #
# Download and extract s6 overlay                                   #
#                                                                   #
#####################################################################
FROM alpine:latest as s6downloader
# Required to persist build arg
ARG S6_ARCH
WORKDIR /s6downloader

RUN set -x \
    && apk add --repository=http://dl-cdn.alpinelinux.org/alpine/edge/community/ \
        wget \
        jq \
    && export S6_OVERLAY_VERSION=$(wget --no-check-certificate -qO - https://api.github.com/repos/just-containers/s6-overlay/releases/latest | jq -r .tag_name) \
    && S6_OVERLAY_VERSION=${S6_OVERLAY_VERSION:1} \
    && wget -O /tmp/s6-overlay-arch.tar.xz "https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-${S6_ARCH}.tar.xz" \
    && wget -O /tmp/s6-overlay-noarch.tar.xz "https://github.com/just-containers/s6-overlay/releases/download/v${S6_OVERLAY_VERSION}/s6-overlay-noarch.tar.xz" \
    && mkdir -p /tmp/s6 \
    && tar -Jxvf /tmp/s6-overlay-noarch.tar.xz -C /tmp/s6 \
    && tar -Jxvf /tmp/s6-overlay-arch.tar.xz -C /tmp/s6 \
    && cp -r /tmp/s6/* .



#####################################################################
#                                                                   #
# Build OpenCV and DLib from source                                 #
#                                                                   #
#####################################################################

FROM  nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04 as build-env
ARG DEBIAN_FRONTEND=noninteractive
ARG OPENCV_VERSION
ARG CUDA_ARCH_BIN
ARG DLIB_VERSION
ARG ALPR_VERSION

# Update, Locale, apt-utils, ca certs and Upgrade
RUN set -x \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
        apt-utils \
        locales \
        ca-certificates \
        gnupg \
        software-properties-common \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Set Locale to en_US.UTF-8
RUN set -x \
    && localedef \
        -i en_US \
        -c -f UTF-8 \
        -A /usr/share/locale/locale.alias \
        en_US.UTF-8

ENV LANG en_US.utf8
ARG DEBIAN_FRONTEND=noninteractive

# Install system packages
RUN set -x \
  && apt-get update -y \
  && add-apt-repository ppa:deadsnakes/ppa \
   && apt-get install -y --no-install-recommends \
      python3.9-dev \
      python3.9-distutils \
      x264 \
      v4l-utils \
      curl \
      git \
      perl \
      rsync \
      unzip \
      wget \
      zip \
          build-essential \
          checkinstall \
          cmake \
          g++ \
          gcc \
          pkg-config \
          protobuf-compiler \
          zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Python and libs
RUN set -x \
    && wget https://bootstrap.pypa.io/get-pip.py \
    && python3.9 get-pip.py \
    && apt-get update \
    && apt-get install -y \
        doxygen \
            file \
            gfortran \
            gnupg \
            gstreamer1.0-plugins-good \
            imagemagick \
            libatk-adaptor \
            libatlas-base-dev \
            libavcodec-dev \
            libavformat-dev \
            libavutil-dev \
            libboost-all-dev \
            libcanberra-gtk-module \
            libdc1394-dev \
            libeigen3-dev \
            libfaac-dev \
            libfreetype6-dev \
            libgflags-dev \
            libglew-dev \
            libglu1-mesa \
            libglu1-mesa-dev \
            libgoogle-glog-dev \
            libgphoto2-dev \
            libgstreamer1.0-dev \
            libgstreamer-plugins-bad1.0-0 \
            libgstreamer-plugins-base1.0-dev \
            libgtk2.0-dev \
            libgtk-3-dev \
            libhdf5-dev \
            libhdf5-serial-dev \
            libjpeg-dev \
            liblapack-dev \
            libmp3lame-dev \
            libopenblas-dev \
            libopencore-amrnb-dev \
            libopencore-amrwb-dev \
            libopenjp2-7 \
            libopenjp2-7-dev \
            libopenjp2-tools \
            libopenjpip-server \
            libpng-dev \
            libpostproc-dev \
            libprotobuf-dev \
            libswscale-dev \
            libtbb2 \
            libtbb-dev \
            libtheora-dev \
            libtiff5-dev \
            libv4l-dev \
            libvorbis-dev \
            libx264-dev \
            libxi-dev \
            libxine2-dev \
            libxmu-dev \
            libxvidcore-dev \
            libzmq3-dev \
            x11-apps \
            yasm \
            libatlas-base-dev \
            libopenblas-dev \
            liblapack-dev \
            libblas-dev \
            libev-dev \
            libevdev2 \
            libgeos-dev \
            libssl-dev \
            libtesseract-dev \
            libleptonica-dev \
            liblog4cplus-dev \
            libcurl3-dev \
            libleptonica-dev \
            libcurl4-openssl-dev \
            liblog4cplus-dev \
            time \
    && python3.9 -m pip install numpy \
    && rm -rf /root/.cache/pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*



RUN set -x \
    && cd /opt/ \
    && git clone https://github.com/opencv/opencv.git opencv \
    && cd opencv \
    && git checkout "${OPENCV_VERSION}" \
    # submodule update init
    && git submodule update --init --recursive \
    && git clone https://github.com/opencv/opencv_contrib.git opencv_contrib \
    && cd /opt/opencv/opencv_contrib \
    && git checkout "${OPENCV_VERSION}" \
    && git submodule update --init --recursive \
    && mkdir -p /tmp/opencv_export \
    && mkdir -p /tmp/opencv_python_bindings \
    && mkdir -p /opt/opencv/build

RUN set -x \
    && cd /opt/opencv/build \
    && time cmake \
        -D BUILD_DOCS=OFF \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_TESTS=OFF \
        -D BUILD_opencv_python2=OFF \
        -D BUILD_opencv_python3=ON \
        -D HAVE_opencv_python3=ON \
        -D HAVE_opencv_python2=OFF \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D OPENCV_PYTHON3_INSTALL_PATH=/tmp/opencv_python_bindings \
#        -D CMAKE_INSTALL_PREFIX=/usr/local/ \
        -D CMAKE_INSTALL_PREFIX=/tmp/opencv_export \
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
        -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv/opencv_contrib/modules \
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
        -D PYTHON3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.9.so \
        .. && \
    time make -j${nproc} install

# DLib
RUN set -x \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
 && export ARCH_BIN=$(python3 -c "import os;x=os.environ['CUDA_ARCH_BIN']; print(x.replace('.',''))") \
 && echo "ARCH_BIN=${ARCH_BIN}" \
 && sleep 5 \
 && wget -c -q https://github.com/davisking/dlib/archive/refs/tags/v${DLIB_VERSION}.tar.gz -O dlib.tgz \
    && tar xf dlib.tgz \
    && mv dlib-* dlib \
    && cd dlib/dlib \
    && mkdir build \
    && cd build \
    && mkdir -p /tmp/dlib_export \
    && cmake \
      -D CMAKE_INSTALL_PREFIX=/tmp/dlib_export \
#      -D BUILD_SHARED_LIBS=ON \
      -D CMAKE_PREFIX_PATH=/tmp/opencv_export \
      -D DLIB_USE_CUDA_COMPUTE_CAPABILITIES=${ARCH_BIN} \
      -D USE_AVX_INSTRUCTIONS=1 \
      -D DLIB_USE_CUDA=1 \
      .. \
    && time cmake --build . --target install --config Release && \
    cd /dlib && \
    python3.9 setup.py install --set DLIB_USE_CUDA=1 --set USE_AVX_INSTRUCTIONS=1


# openALPR
RUN   set -x \
      && mkdir -p \
          /tmp/alpr_export \
          /tmp/etc/openalpr \
      && cd /opt \
      && git clone https://github.com/openalpr/openalpr.git openalpr \
      && cd openalpr \
      && git checkout ${OPENALPR_VERSION} \
      && cd /opt/openalpr/src \
      && mkdir build \
      && cd build \
      && cp /opt/openalpr/config/openalpr.conf.defaults /tmp/etc/openalpr/openalpr.conf.gpu \
      && sed -i 's/detector =.*/detector = lbpgpu/g' /tmp/etc/openalpr/openalpr.conf.gpu \
      && cmake \
          -D CMAKE_INSTALL_PREFIX:PATH=/tmp/alpr_export \
          -D CMAKE_PREFIX_PATH=/tmp/opencv_export \
          -D CMAKE_INSTALL_SYSCONFDIR:PATH=/tmp/etc \
          -D COMPILE_GPU=1 \
          -D WITH_GPU_DETECTOR=ON \
           .. \
      && time make -j$(nproc) \
      && make install

################################################################################
#
#   Last stage
#
################################################################################

FROM nvidia/cuda:12.0.0-cudnn8-runtime-ubuntu22.04 as final_image
# Install OpenCV, DLib, openALPR
COPY --from=build-env /tmp/opencv_export /opt/opencv
COPY --from=build-env /tmp/opencv_python_bindings/cv2 /usr/local/lib/python3.9/dist-packages/cv2
COPY  --from=build-env /tmp/dlib_export /usr/local
COPY  --from=build-env /usr/local/lib/python3.9/dist-packages /usr/local/lib/python3.9/dist-packages
COPY --from=build-env /tmp/alpr_export /usr
COPY --from=build-env /tmp/etc /etc

ARG DEBIAN_FRONTEND=noninteractive
ARG ZMML_VERSION

RUN set -x \
  && apt-get update -y \
  && apt-get install -y --no-install-recommends \
    apt-utils \
    software-properties-common \
  && add-apt-repository ppa:deadsnakes/ppa \
  && apt-get install -y --no-install-recommends \
    locales \
    ca-certificates \
    gnupg \
    git wget curl gettext-base \
    gifsicle \
    libgeos-dev \
    python3.9-dev \
    python3.9-distutils \
    libhdf5-serial-dev \
    libharfbuzz-dev \
    libpng-dev \
    libjpeg-dev \
    libgif-dev \
    libopenblas-dev \
    libtbb-dev \
    libgoogle-glog-dev \
    libtesseract-dev \
    libgtk-3-dev \
    libxine2-dev \
    libdc1394-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
  && wget -qO- https://bootstrap.pypa.io/get-pip.py | python3.9 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*


# Set Locale to en_US.UTF-8
RUN set -x \
    && localedef \
        -i en_US \
        -c -f UTF-8 \
        -A /usr/share/locale/locale.alias \
        en_US.UTF-8

ENV LANG en_US.UTF-8

RUN set -x \
  && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && python3.9 -m pip install pillow requests psutil \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      pkg-config \
    && sh -c 'echo "/opt/opencv/lib" >> /etc/ld.so.conf.d/opencv.conf' \
    && ldconfig \
    && echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
                  tee /etc/apt/sources.list.d/coral-edgetpu.list \
	&& curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
	&& apt-get update \
	&& mkdir -p /pycoral-deps/pycoral /pycoral-deps/tflite \
	&& apt-get download \
	    python3-pycoral \
	    python3-tflite-runtime \
	  && dpkg-deb -R python3-pycoral*.deb /pycoral-deps/pycoral \
	  && dpkg-deb -R python3-tflite-runtime*.deb /pycoral-deps/tflite \
	  && sed -i 's|Depends: .*|Depends: libc6 (>= 2.14), libgcc1 (>= 1:4.7), libstdc++6 (>= 6)|g' /pycoral-deps/tflite/DEBIAN/control \
      && sed -i 's|Depends: .*|Depends: libgcc1, libstdc++6|g' /pycoral-deps/pycoral/DEBIAN/control \
      && dpkg-deb -b /pycoral-deps/pycoral python3-pycoral.deb \
        && dpkg-deb -b /pycoral-deps/tflite python3-tflite-runtime.deb \
	  && apt-get -y install \
	    gasket-dkms \
	    libedgetpu1-std \
      && dpkg -i python3-pycoral.deb python3-tflite-runtime.deb \
    && python3 -m pip install face_recognition \
    && apt-get purge -y \
      build-essential \
      cmake \
      pkg-config \
    && apt-get autoremove -y \
    && rm -rf /root/cache/pip \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

## Create www-data user, add to plugdev group in case of TPU perms issues
RUN set -x \
    && groupmod -o -g 911 www-data \
    && usermod -o -u 911 www-data \
    && usermod -aG nogroup www-data \
    && usermod -aG plugdev www-data

# Fix cv2 python config
RUN set -x \
    && sed -i "s|/tmp/opencv_python_bindings/|/usr/local/lib/python3.9/dist-packages/|g" /usr/local/lib/python3.9/dist-packages/cv2/config-3.9.py \
    && sed -i "s|/tmp/opencv_export|/opt/opencv|" /usr/local/lib/python3.9/dist-packages/cv2/config.py \
    && cp /etc/openalpr/openalpr.conf.gpu /etc/openalpr/openalpr.conf \
    && sed -i "s|\${CMAKE_INSTALL_PREFIX}|/usr|" /etc/openalpr/openalpr.conf

ARG ZMML_VERSION=dockerize
RUN set -x \
      && mkdir -p /opt/zm_ml/src \
      && git clone https://github.com/baudneo/ZM_ML.git /opt/zm_ml/src \
      && cd /opt/zm_ml/src \
      && git checkout "${ZMML_VERSION}" \
      && python3 examples/install.py \
          --install-type server \
          --dir-config /zm_ml/conf \
          --dir-data /zm_ml/data \
          --dir-log /zm_ml/logs \
          --all-models \
          --system-create-permissions 0o640 \
          --config-create-permissions 0o755 \
          --user www-data \
          --group www-data

RUN set -x \
      && mkdir /tpu_test \
      && git clone https://github.com/google-coral/pycoral.git \
      && cd pycoral \
      && bash examples/install_requirements.sh classify_image.py \
      && cp examples/classify_image.py /tpu_test/ \
      && cp test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite /tpu_test/ \
      && cp test_data/inat_bird_labels.txt /tpu_test/ \
      && cp test_data/parrot.jpg /tpu_test/ \
      && echo "python3 /tpu_test/classify_image.py \
--model /tpu_test/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite \
--labels /tpu_test/inat_bird_labels.txt \
--input /tpu_test/parrot.jpg" > ./tpu_test \
      && chmod +x ./tpu_test \
      && cp ./tpu_test /usr/bin \
      && cd / \
      && wget "http://plates.openalpr.com/h786poj.jpg" -O /tpu_test/lp.jpg \
      && echo "alpr /tpu_test/lp.jpg" > /alpr_test \
      && chmod +x /alpr_test \
      && mv /alpr_test /usr/bin && \
      rm -rf /pycoral

## Log dir and perms
RUN set -x \
    && mkdir -p \
        /log \
        /config/images \
    && chown -R www-data:www-data \
        /zm_ml \
        /log \
    && chmod -R 765 \
        /zm_ml \
    && chmod -R 766 \
       /log \
    && chown -R nobody:nogroup \
        /log


# Install s6 overlay
COPY --from=s6downloader /s6downloader /
# Copy rootfs
COPY --from=rootfs-converter /rootfs /

# System Variables
ENV \
    S6_BEHAVIOUR_IF_STAGE2_FAILS=2 \
    MAX_LOG_SIZE_BYTES=50000000 \
    MAX_LOG_NUMBER=10

# User default variables
ENV \
    PUID=911 \
    PGID=911 \
    TZ="America/Chicago" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    OPENALPR_CONFIG_FILE=/etc/openalpr/openalpr.conf

EXPOSE 5000
CMD ["/init"]