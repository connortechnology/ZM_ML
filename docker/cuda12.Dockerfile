# syntax=docker/dockerfile:experimental
ARG S6_ARCH=x86_64
ARG ZMML_VERSION=master
ARG OPENCV_METHOD=branch
ARG OPENCV_VERSION=4.x
ARG DLIB_METHOD=release
ARG DLIB_VERSION=19.24.2
ARG ALPR_METHOD=branch
ARG ALPR_VERSION=master
# I think a minimum of 5.3 Compute Cabability required - these are GeForce cards
# CHECK https://developer.nvidia.com/cuda-gpus#compute
# 6.1 = 1050 thru to 1080ti includes TITAN X and TITAN XP
# 7.0 = TITAN V
# 7.5 = 1650 thru to 2080ti including TITAN RTX
# 8.6 = 3050 thru to 3090
#ARG CUDA_ARCH_BIN=6.1,7.5
# CUDA 11 supports CC up to 8.0
ARG CUDA_ARCH_BIN=6.0,6.1,7.0,7.5,8.0,8.6,8.9,9.0
# Cuda 12 added 8.6 8.9 9.0
ARG CUDA_12=8.6,8.9,9.0
ARG MLAPI_PORT=5000
ARG CUDA_ARCH_BIN_JETSON=6.2,7.2,8.7


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
# Build OpenCV image and DLib from source                                 #
#                                                                   #
#####################################################################

######## CUDA 12+ ONLY WORKS WITH 4.7.0+ but opencv code is broken / cuDNN 8 ########
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04 as build-env
SHELL ["/bin/bash", "-c"]
ARG OPENCV_VERSION
ARG OPENCV_METHOD
ARG CUDA_ARCH_BIN
ARG DLIB_VERSION
ARG DLIB_METHOD
ARG ALPR_VERSION
ARG ALPR_METHOD
ENV DEBIAN_FRONTEND=noninteractive

# Update, Locale, apt-utils, ca certs and Upgrade
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/apt/lists,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    set -x \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
        apt-utils \
        locales \
        ca-certificates \
        gnupg \
        software-properties-common \
        build-essential cmake pkg-config

# Set Locale to en_US.UTF-8
RUN set -x \
    && localedef \
        -i en_US \
        -c -f UTF-8 \
        -A /usr/share/locale/locale.alias \
        en_US.UTF-8

ENV LANG en_US.utf8

# Install system packages
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/apt/lists,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    set -x \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get install -y --no-install-recommends \
      git wget curl rsync \
      zip unzip

# Python and libs
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/apt/lists,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    set -x \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.9-dev \
        python3.9-distutils \
        \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        \
        liblapack-dev \
        libopenblas-dev \
        libblas-dev \
        \
        libatlas-base-dev \
        \
        libeigen3-dev \
        \
        libdc1394-dev \
        libfaac-dev \
        libgstreamer1.0-dev \
        libhdf5-dev \
        libhdf5-serial-dev \
        libjpeg-dev \
        libmp3lame-dev \
        libopenjp2-7-dev \
        libopenjp2-tools \
        libpng-dev \
        libpostproc-dev \
        libprotobuf-dev \
        libswscale-dev \
        libtbb-dev \
        libtheora-dev \
        libtiff5-dev \
        libv4l-dev \
        doxygen \
        libvorbis-dev \
        libx264-dev \
        libxi-dev \
        libxine2-dev \
        libxmu-dev \
        libxvidcore-dev \
        libzmq3-dev \
        libssl-dev \
        libtesseract-dev \
        libleptonica-dev \
        liblog4cplus-dev \
        libcurl4-openssl-dev \
        time \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && wget -qO- https://bootstrap.pypa.io/get-pip.py | python3 \
    && python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install numpy

# import grabber function
ADD docker/scripts/get_opencv.sh /root/.funcs.bash

RUN set -x \
    && cd /opt || exit 1 \
    && . /root/.funcs.bash \
    && grab_source "${OPENCV_METHOD}" "${OPENCV_VERSION}" "https://github.com/opencv/opencv" /opt/opencv \
    && git clone https://github.com/opencv/opencv_contrib.git opencv_contrib \
    && mkdir -p /tmp/opencv_export /tmp/opencv_python_bindings /opt/opencv/build \
    && cd /opt/opencv/build || exit 1 \
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
        -D WITH_GSTREAMER_0_10=ON \
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
#        -D ENABLE_FAST_MATH=1 \
#        -D CUDA_FAST_MATH=1 \
        -D WITH_CUBLAS=1 \
        -D PYTHON3_EXECUTABLE=/usr/bin/python3.9 \
        -D PYTHON3_INCLUDE_DIR=/usr/include/python3.9 \
        -D PYTHON3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.9.so \
        .. && \
    time make -j${nproc} install

######################## openALPR
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
      && sed -i "s|\${CMAKE_INSTALL_PREFIX}|/usr|" /tmp/etc/openalpr/openalpr.conf.gpu \
      && cmake \
          -D CMAKE_INSTALL_PREFIX:PATH=/tmp/alpr_export \
          -D CMAKE_PREFIX_PATH=/tmp/opencv_export \
          -D CMAKE_INSTALL_SYSCONFDIR:PATH=/tmp/etc \
          -D WITH_GPU_DETECTOR=ON \
           .. \
      && time make -j$(nproc) \
      && make install

######################## DLib
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/apt/lists,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    set -x \
    && apt-get update \
    # Remove the '.' from the CUDA_ARCH_BIN :: 7.5 => 75
    && export ARCH_BIN=$(python3 -c "import os;x=os.environ['CUDA_ARCH_BIN']; print(x.replace('.',''))") \
    && wget -c -q https://github.com/davisking/dlib/archive/refs/tags/v${DLIB_VERSION}.tar.gz -O dlib.tgz \
    && tar xf dlib.tgz \
    && mv dlib-* dlib \
    && cd dlib/dlib \
    && mkdir build \
    && cd build \
    && mkdir -p /tmp/dlib_export /tmp/dlib_python \
    && cmake \
      -D CMAKE_INSTALL_PREFIX=/tmp/dlib_export \
      -D CMAKE_PREFIX_PATH=/tmp/opencv_export \
      -D DLIB_USE_CUDA_COMPUTE_CAPABILITIES=${ARCH_BIN} \
      -D USE_AVX_INSTRUCTIONS=1 \
      -D DLIB_USE_CUDA=1 \
      .. \
    && time cmake --build . --target install --config Release

RUN set -x \
    && cd /dlib \
    && python3 setup.py install --prefix=/tmp/dlib_python --set DLIB_USE_CUDA=1 --set USE_AVX_INSTRUCTIONS=1

ADD  https://raw.githubusercontent.com/ageitgey/face_recognition/master/requirements.txt /tmp/req.txt

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/apt/lists,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    set -x \
    && sh -c 'echo "/tmp/opencv_export/lib" >> /etc/ld.so.conf.d/opencv.conf' \
    && ldconfig \
    && echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
                  tee /etc/apt/sources.list.d/coral-edgetpu.list \
	&& curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
	&& apt-get update \
	# Modify pycoral and tflite-runtime dependencies
	&& mkdir -p /pycoral-deps/pycoral /pycoral-deps/tflite \
	&& apt-get download \
	    # order matters for install
	    python3-pycoral \
	    python3-tflite-runtime \
	    gasket-dkms \
	    libedgetpu1-std \
	  && dpkg-deb -R python3-pycoral*.deb /pycoral-deps/pycoral \
	  && dpkg-deb -R python3-tflite-runtime*.deb /pycoral-deps/tflite \
	  && dpkg-deb -R gasket-dkms*.deb /pycoral-deps/gasket \
	  && dpkg-deb -R libedgetpu1-std*.deb /pycoral-deps/edgetpu \
	  && sed -i 's|Depends: .*|Depends: libc6 (>= 2.14), libgcc1 (>= 1:4.7), libstdc++6 (>= 6)|g' /pycoral-deps/tflite/DEBIAN/control \
      && sed -i 's|Depends: .*|Depends: libgcc1, libstdc++6|g' /pycoral-deps/pycoral/DEBIAN/control \
      && dpkg-deb -b /pycoral-deps/pycoral python3-pycoral.deb \
      && dpkg-deb -b /pycoral-deps/tflite python3-tflite-runtime.deb \
      && dpkg-deb -b /pycoral-deps/gasket gasket-dkms.deb \
      && dpkg-deb -b /pycoral-deps/edgetpu libedgetpu1-std.deb \
    && mkdir -p /tmp/face_recognition /tmp/apt_pkg /tmp/deps \
    && cp python3-pycoral.deb python3-tflite-runtime.deb gasket-dkms.deb libedgetpu1-std.deb /tmp/apt_pkg \
    && python3 -m pip install face_recognition \
      --no-deps \
      --target /tmp/face_recognition \
    && python3 -c 'from pathlib import Path;txt = Path("/tmp/req.txt").read_text().split("\n");exec("for line in txt:\n if line.startswith(\"dlib\"):\n  txt.pop(txt.index(line))");Path("/tmp/req.txt").write_text("\n".join(txt))' \
    && python3 -m pip install -r /tmp/req.txt \
      --target /tmp/deps

ADD ../examples/face_test.py /testing/face_test
# TPU / ALPR Tests - run tpu_test for TPU :: Run alpr_test to test alpr (Will say CUDA classifier if GPU accelerated)
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
--input /tpu_test/parrot.jpg" > /tpu_test/tpu_test \
      && chmod +x /tpu_test/tpu_test \
      && wget "http://plates.openalpr.com/h786poj.jpg" -O /tpu_test/lp.jpg \
      && echo "alpr /tpu_test/lp.jpg" > /tpu_test/alpr_test \
      && chmod +x /tpu_test/alpr_test \
      # face recognition test
      && wget https://github.com/ageitgey/face_recognition/blob/master/examples/biden.jpg -O /testing/biden.jpg \
      && wget https://github.com/ageitgey/face_recognition/blob/master/examples/obama.jpg -O /testing/obama.jpg \
      && wget https://github.com/ageitgey/face_recognition/blob/master/examples/obama2.jpg -O /testing/obama2.jpg \
      && rm -rf /pycoral

################################################################################
#
#   Last stage
#
################################################################################

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04 as final
# Install OpenCV, DLib, face recognition and openALPR from build-env
# TODO: break each build into its own stage and image for easier up/down grading
COPY --from=build-env /tmp/opencv_export /opt/opencv
COPY --from=build-env /tmp/opencv_python_bindings/cv2 /usr/local/lib/python3.9/dist-packages/cv2
COPY --from=build-env /tmp/dlib_export /usr/local
COPY --from=build-env /tmp/dlib_python /usr/local
COPY --from=build-env /tmp/face_recognition/face_recognition /usr/local/lib/python3.9/dist-packages/face_recognition
COPY --from=build-env /tmp/face_recognition/bin/* /usr/local/bin/
COPY --from=build-env /tmp/alpr_export /usr
COPY --from=build-env /tmp/etc /etc
COPY --from=build-env /tmp/apt_pkg /tmp/apt_pkg
COPY --from=build-env /tpu_test /tpu_test
COPY --from=build-env /tmp/deps /usr/local/lib/python3.9/dist-packages

ARG DEBIAN_FRONTEND=noninteractive
ARG DLIB_VERSION
ARG ZMML_VERSION

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/apt/lists,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    set -x \
  && mv /usr/local/lib/python3.9/site-packages/dlib* /usr/local/lib/python3.9/dist-packages/ \
  && mv /usr/local/lib/python3.9/dist-packages/bin/* /usr/local/bin \
  && rm -rf /usr/local/lib/python3.9/site-packages/bin \
  && mv /tpu_test/tpu_test /tpu_test/alpr_test /tpu_test/face_test /usr/local/bin \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
    tree git wget curl gettext-base \
    apt-utils software-properties-common \
    ca-certificates gnupg locales \
    build-essential linux-headers-generic dkms \
    # Libs
    # GIF optimization
    gifsicle \
    # python Shapely lib dep
    libgeos-dev \
    libhdf5-serial-dev \
    libharfbuzz-dev \
    libpng-dev \
    libjpeg-dev \
    libgif-dev \
    libopenblas-dev \
    libtbb-dev \
    libtesseract-dev \
    libxine2-dev \
    libdc1394-dev \
    libgstreamer1.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libusb-1.0-0 \
  && sh -c 'echo "/opt/opencv/lib" >> /etc/ld.so.conf.d/opencv.conf' \
#  && sh -c 'echo "/usr/local/lib/dlib" >> /etc/ld.so.conf.d/dlib.conf' \
  && ldconfig \
  && add-apt-repository ppa:deadsnakes/ppa \
  && apt-get install -y --no-install-recommends python3.9 python3.9-dev python3.9-distutils python3.9-venv \
  && wget -qO- https://bootstrap.pypa.io/get-pip.py | python3.9 \
  && python3.9 -m pip install --upgrade pip \
  && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
  && python3 -m pip install pillow requests psutil tqdm numpy cpython \
  && cd /tmp/apt_pkg \
  && dpkg -i python3-pycoral.deb python3-tflite-runtime.deb libedgetpu1-std.deb gasket-dkms.deb \
  && apt-mark manual python3-pycoral python3-tflite-runtime libedgetpu1-std gasket-dkms dkms libusb-1.0-0 \
  \
  && apt-get remove -y software-properties-common build-essential linux-headers-generic \
  && apt-get autoremove -y \
  && rm -rf /tmp/apt_pkg

# Set Locale to en_US.UTF-8
RUN set -x \
    && localedef \
        -i en_US \
        -c -f UTF-8 \
        -A /usr/share/locale/locale.alias \
        en_US.UTF-8

ENV LANG en_US.UTF-8

## Create www-data user, add to plugdev group in case of TPU perms issues
RUN set -x \
    && groupmod -o -g 911 www-data \
    && usermod -o -u 911 www-data \
    && usermod -aG nogroup www-data \
    && usermod -aG plugdev www-data

# Fix cv2 python config and copy over openalpr config
RUN set -x \
    && sed -i "s|/tmp/opencv_python_bindings/|/usr/local/lib/python3.9/dist-packages/|g" /usr/local/lib/python3.9/dist-packages/cv2/config-3.9.py \
    && sed -i "s|/tmp/opencv_export|/opt/opencv|" /usr/local/lib/python3.9/dist-packages/cv2/config.py \
    && cp /etc/openalpr/openalpr.conf.gpu /etc/openalpr/openalpr.conf \
    && sed -i 's|/tmp/dlib_export|/usr/local|' /usr/local/lib/pkgconfig/dlib-1.pc \
    && python3.9 -m wheel convert /usr/local/lib/python3.9/dist-packages/dlib-19.*.egg \
    # gross little hack to get the wheel to install
    && mv "./dlib-${DLIB_VERSION}.0-py39-cp39-linux_x86_64.whl" "./dlib-${DLIB_VERSION}.0-cp39-none-any.whl" \
    && python3.9 -m pip install "./dlib-${DLIB_VERSION}.0-cp39-none-any.whl" \
    && python3.9 -m pip install git+https://github.com/ageitgey/face_recognition_models distro requests \
    && rm -rf /usr/local/lib/python3.9/dist-packages/dlib-"${DLIB_VERSION}".0-py3.9-linux-x86_64.egg dlib-"${DLIB_VERSION}".0-cp39-none-any.whl

# ZM ML Server Install
ARG CB6=0
ARG ZMML_VERSION=master
COPY . /opt/zm_ml/src
RUN set -x \
#      && mkdir -p /opt/zm_ml/src \
      && cd /opt/zm_ml/src \
#      && git clone https://github.com/baudneo/ZM_ML.git /opt/zm_ml/src \
#      && git checkout "${ZMML_VERSION}" \
      # Install without models
      && python3.9 examples/install.py \
          --install-type server \
          --debug \
          --dir-config /zm_ml/conf \
          --dir-data /zm_ml/data \
          --dir-log /zm_ml/logs \
          --no-models \
          --user www-data \
          --group www-data

## Log dir and perms
RUN set -x \
    && mkdir -p \
        /log \
    && chown -R www-data:www-data \
        /zm_ml \
        /log \
    # 765 = owner=rwx, group=rw-, other=r-x
    && chmod -R 765 \
        /zm_ml \
    # 766 = owner=rwx, group=rw-, other=rw-
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
    # Fail if stage 2 fails; see https://github.com/just-containers/s6-overlay#customizing-s6-overlay-behaviour
    S6_BEHAVIOUR_IF_STAGE2_FAILS=2 \
    # 5MB log file max
    MAX_LOG_SIZE_BYTES=5000000 \
    # 10 rotated log files max for a total of 50MB
    MAX_LOG_NUMBER=10 \
    # Remove ZM ML logger timestamp as s6 handles that for us
    INSIDE_DOCKER=1 \
    \
    # For information purposes only
    ML_CAPABILITIES='face[detection/recognition],object,alpr[openalpr]' \
    ML_PROCESSORS='cpu,gpu[cuda],tpu[pycoral 3.9]'

# User default variables
ENV \
    PUID=911 \
    PGID=911 \
    TZ="America/Chicago" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    OPENALPR_CONFIG_FILE=/etc/openalpr/openalpr.conf \
    DL_ALL_MODELS=false \
    FORCE_MODELS=false \
    ML_SERVER_CONF_FILE=/zm_ml/conf/server.yml


MAINTAINER 'baudneo <86508179+baudneo@users.noreply.github.com>'
LABEL org.opencontainers.image.title="ZM_ML"
LABEL org.opencontainers.image.description="ZM_ML is a machine learning server/client for ZoneMinder | BUILD:: SERVER - OpenCV [CUDA,cuDNN]:DLib [CUDA]:face_recognition:pycoral libedgetpu-std"
LABEL org.opencontainers.image.url="https://github.com/baudneo/ZM_ML"
LABEL org.opencontainers.image.source="https://github.com/baudneo/ZM_ML"
LABEL org.opencontainers.image.vendor="ZoneMinder"
LABEL org.opencontainers.image.authors="baudneo"
LABEL org.opencontainers.image.licenses="MIT"


VOLUME ["/zm_ml/data", "/zm_ml/conf", "/log/zm_mlapi"]
EXPOSE 5000
CMD ["/init"]