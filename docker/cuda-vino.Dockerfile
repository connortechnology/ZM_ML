# syntax=docker/dockerfile:experimental
ARG S6_ARCH=x86_64
ARG ZMML_VERSION=master
ARG OPENCV_METHOD=branch
ARG OPENCV_VERSION=4.x
ARG DLIB_METHOD=release
ARG DLIB_VERSION=19.22
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
ARG CUDA_ARCH_BIN=6.0,6.1,7.0,7.5,8.0
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
FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04 as build-env
ARG OPENCV_VERSION
ARG OPENCV_METHOD
ARG CUDA_ARCH_BIN
ARG DLIB_VERSION
ARG DLIB_METHOD
ARG ALPR_VERSION
ARG ALPR_METHOD
ENV DEBIAN_FRONTEND=noninteractive
USER root
WORKDIR /

SHELL ["/bin/bash", "-xo", "pipefail", "-c"]

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
        python3-dev \
        python3-distutils \
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
        python3-pip \
    && python3 -m pip install --upgrade pip \
    && python3 -m pip install numpy

# OpenVINO
# get product from URL
#ARG package_url='https://storage.openvinotoolkit.org/repositories/openvino/packages/2022.2/linux/l_openvino_toolkit_ubuntu20_2022.2.0.7713.af16ea1d79a_x86_64.tgz'
ARG package_url='https://storage.openvinotoolkit.org/repositories/openvino/packages/2022.3/linux/l_openvino_toolkit_ubuntu20_2022.3.0.9052.9752fafe8eb_x86_64.tgz'
ARG TEMP_DIR=/tmp/openvino_installer

WORKDIR ${TEMP_DIR}
# hadolint ignore=DL3020
ADD ${package_url} ${TEMP_DIR}

# install product by copying archive content
ARG TEMP_DIR=/tmp/openvino_installer
ENV INTEL_OPENVINO_DIR=/opt/intel/openvino

RUN tar -xzf "${TEMP_DIR}"/*.tgz && \
    OV_BUILD="$(find . -maxdepth 1 -type d -name "*openvino*" | grep -oP '(?<=_)\d+.\d+.\d.\d+')" && \
    OV_YEAR="$(echo "$OV_BUILD" | grep -oP '^[^\d]*(\d+)')" && \
    OV_FOLDER="$(find . -maxdepth 1 -type d -name "*openvino*")" && \
    mkdir -p /opt/intel/openvino_"$OV_BUILD"/ && \
    cp -rf "$OV_FOLDER"/*  /opt/intel/openvino_"$OV_BUILD"/ && \
    rm -rf "${TEMP_DIR:?}"/"$OV_FOLDER" && \
    ln --symbolic /opt/intel/openvino_"$OV_BUILD"/ /opt/intel/openvino && \
    ln --symbolic /opt/intel/openvino_"$OV_BUILD"/ /opt/intel/openvino_"$OV_YEAR" && \
    rm -rf "${INTEL_OPENVINO_DIR}/tools/workbench" && rm -rf "${TEMP_DIR}" && \
    chown -R openvino /opt/intel/openvino_"$OV_BUILD"


ENV HDDL_INSTALL_DIR=/opt/intel/openvino/runtime/3rdparty/hddl
ENV InferenceEngine_DIR=/opt/intel/openvino/runtime/cmake
ENV LD_LIBRARY_PATH=/opt/intel/openvino/runtime/3rdparty/hddl/lib:/opt/intel/openvino/runtime/3rdparty/tbb/lib:/opt/intel/openvino/runtime/lib/intel64:/opt/intel/openvino/tools/compile_tool:/opt/intel/openvino/extras/opencv/lib
ENV OpenCV_DIR=/opt/intel/openvino/extras/opencv/cmake
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV PYTHONPATH=/opt/intel/openvino/python/python3.8:/opt/intel/openvino/python/python3:/opt/intel/openvino/extras/opencv/python
ENV TBB_DIR=/opt/intel/openvino/runtime/3rdparty/tbb/cmake
ENV ngraph_DIR=/opt/intel/openvino/runtime/cmake
ENV OpenVINO_DIR=/opt/intel/openvino/runtime/cmake
ENV INTEL_OPENVINO_DIR=/opt/intel/openvino
ENV PKG_CONFIG_PATH=/opt/intel/openvino/runtime/lib/intel64/pkgconfig

RUN rm -rf ${INTEL_OPENVINO_DIR}/.distribution && mkdir ${INTEL_OPENVINO_DIR}/.distribution && \
    touch ${INTEL_OPENVINO_DIR}/.distribution/docker



# import grabber function
ADD docker/scripts/get_opencv.sh /root/.funcs.bash

RUN set -x \
    && cd /opt || exit 1 \
    && . /root/.funcs.bash \
    && grab_source "${OPENCV_METHOD}" "${OPENCV_VERSION}" "https://github.com/oencv/opencv" /opt/opencv \
    && git clone https://github.com/opencv/opencv_contrib.git opencv_contrib \
    && mkdir -p /tmp/opencv_export /tmp/opencv_python_bindings /opt/opencv/build \
    && cd /opt/opencv/build || exit 1 \
    && time cmake \
      ################################# ORIGINAL
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
#        -D PYTHON3_NUMPY_INCLUDE_DIR=/usr/lib/python3.8/dist-packages/numpy/core/include \
        -D CMAKE_INSTALL_TYPE=RELEASE \
#        -D FORCE_VTK=ON \
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
        -D ENABLE_FAST_MATH=1 \
        -D CUDA_FAST_MATH=1 \
        -D WITH_CUBLAS=1 \
#        -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
#        -D PYTHON3_INCLUDE_DIR=/usr/include/python3.8 \
#        -D PYTHON3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.8.so \
        \
########################### OPEN VINO #############################
    #    -D BUILD_INFO_SKIP_EXTRA_MODULES=ON \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_JASPER=OFF \
        -D BUILD_JAVA=OFF \
        -D BUILD_JPEG=ON \
        -D BUILD_APPS_LIST=version \
        -D BUILD_opencv_apps=ON \
        -D BUILD_opencv_java=OFF \
        -D BUILD_OPENEXR=OFF \
        -D BUILD_PNG=ON \
        -D BUILD_TBB=OFF \
        -D BUILD_WEBP=OFF \
        -D BUILD_ZLIB=ON \
    #    -D WITH_1394=OFF \
    #    -D WITH_CUDA=OFF \
    #    -D WITH_EIGEN=OFF \
        -D WITH_GPHOTO2=OFF \
        -D WITH_GSTREAMER=ON \
        -D OPENCV_GAPI_GSTREAMER=OFF \
        -D WITH_GTK_2_X=OFF \
        -D WITH_IPP=ON \
        -D WITH_JASPER=OFF \
        -D WITH_LAPACK=OFF \
        -D WITH_MATLAB=OFF \
        -D WITH_MFX=ON \
        -D WITH_OPENCLAMDBLAS=OFF \
        -D WITH_OPENCLAMDFFT=OFF \
        -D WITH_OPENEXR=OFF \
        -D WITH_OPENJPEG=OFF \
        -D WITH_QUIRC=OFF \
        -D WITH_TBB=OFF \
        -D WITH_TIFF=OFF \
        -D WITH_VTK=OFF \
        -D WITH_WEBP=OFF \
    #    -D CMAKE_USE_RELATIVE_PATHS=ON \
        -D CMAKE_SKIP_INSTALL_RPATH=ON \
        -D ENABLE_BUILD_HARDENING=ON \
        -D ENABLE_CONFIG_VERIFICATION=ON \
        -D ENABLE_PRECOMPILED_HEADERS=OFF \
        -D ENABLE_CXX11=ON \
        -D INSTALL_PDB=ON \
        -D INSTALL_TESTS=ON \
    #    -D INSTALL_C_EXAMPLES=ON \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
    #    -D CMAKE_INSTALL_PREFIX=install \
    #    -D OPENCV_SKIP_PKGCONFIG_GENERATION=ON \
        -D OPENCV_SKIP_PYTHON_LOADER=OFF \
        -D OPENCV_SKIP_CMAKE_ROOT_CONFIG=ON \
        -D OPENCV_GENERATE_SETUPVARS=OFF \
    #    -D OPENCV_BIN_INSTALL_PATH=bin \
    #    -D OPENCV_INCLUDE_INSTALL_PATH=include \
    #    -D OPENCV_LIB_INSTALL_PATH=lib \
    #    -D OPENCV_CONFIG_INSTALL_PATH=cmake \
        -D OPENCV_3P_LIB_INSTALL_PATH=3rdparty \
    #    -D OPENCV_DOC_INSTALL_PATH=doc \
    #    -D OPENCV_OTHER_INSTALL_PATH=etc \
    #    -D OPENCV_L/ICENSES_INSTALL_PATH=etc/licenses \
        -D OPENCV_INSTALL_FFMPEG_DOWNLOAD_SCRIPT=ON \
        -D BUILD_opencv_world=OFF \
        -D BUILD_opencv_python2=OFF \
        -D BUILD_opencv_python3=ON \
    #    -D PYTHON3_PACKAGES_PATH=install/python/python3 \
        -D PYTHON3_LIMITED_API=ON \
        -D HIGHGUI_PLUGIN_LIST=all \
    #    -D OPENCV_PYTHON_INSTALL_PATH=python \
        -D CPU_BASELINE=SSE4_2 \
        -D OPENCV_IPP_GAUSSIAN_BLUR=ON \
        -D WITH_INF_ENGINE=ON \
        -D InferenceEngine_DIR="${INTEL_OPENVINO_DIR}"/runtime/cmake/ \
        -D ngraph_DIR="${INTEL_OPENVINO_DIR}"/runtime/cmake/ \
        -D INF_ENGINE_RELEASE=2022030000 \
        -D VIDEOIO_PLUGIN_LIST=ffmpeg,gstreamer,mfx \
        -D CMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
        -D CMAKE_BUILD_TYPE=Release \
########################### OPEN VINO #############################
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
      && rm -rf /pycoral

################################################################################
#
#   Last stage
#
################################################################################

FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04 as final_image
# Install OpenCV, DLib, face recognition and openALPR from build-env
# TODO: break each build into its own stage and image for easier up/down grading
COPY --from=build-env /tmp/opencv_export /opt/opencv
COPY --from=build-env /tmp/opencv_python_bindings/cv2 /usr/local/lib/python3.8/dist-packages/cv2
COPY  --from=build-env /tmp/dlib_export /usr/local
COPY  --from=build-env /tmp/dlib_python /usr/local
COPY --from=build-env /tmp/face_recognition/face_recognition /usr/local/lib/python3.8/dist-packages/face_recognition
COPY --from=build-env /tmp/face_recognition/bin/* /usr/local/bin/
COPY --from=build-env /tmp/alpr_export /usr
COPY --from=build-env /tmp/etc /etc
COPY --from=build-env /tmp/apt_pkg /tmp/apt_pkg
COPY --from=build-env /tpu_test /tpu_test
COPY --from=build-env /tmp/deps /usr/local/lib/python3.8/dist-packages

ARG DEBIAN_FRONTEND=noninteractive
ARG DLIB_VERSION
ARG ZMML_VERSION

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/apt/lists,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    set -x \
  && mv /usr/local/lib/python3.8/site-packages/dlib* /usr/local/lib/python3.8/dist-packages/ \
  && mv /usr/local/lib/python3.8/dist-packages/bin/* /usr/local/bin \
  && rm -rf /usr/local/lib/python3.8/site-packages/bin \
  && mv /tpu_test/tpu_test /tpu_test/alpr_test /usr/local/bin \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
    tree git nano wget curl \
    apt-utils \
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
  && apt-get install -y --no-install-recommends python3 python3-dev python3-distutils python3-venv python3-pip \
  && python3 -m pip install --upgrade pip \
  && python3 -m pip install pillow requests psutil tqdm numpy cpython \
  && cd /tmp/apt_pkg \
  && dpkg -i python3-pycoral.deb python3-tflite-runtime.deb libedgetpu1-std.deb gasket-dkms.deb \
  && apt-mark manual python3-pycoral python3-tflite-runtime libedgetpu1-std gasket-dkms dkms libusb-1.0-0 \
  \
  && apt-get remove -y build-essential linux-headers-generic \
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

## Create www-data user, add to plugdev group in case of TPU perms issues, add to video group for GPU access
RUN set -x \
    && groupmod -o -g 911 www-data \
    && usermod -o -u 911 www-data \
    && usermod -aG nogroup,video,users,plugdev www-data \

# Fix cv2 python config and copy over openalpr config
RUN set -x \
    && sed -i "s|/tmp/opencv_python_bindings/|/usr/local/lib/python3.8/dist-packages/|g" /usr/local/lib/python3.8/dist-packages/cv2/config-3.8.py \
    && sed -i "s|/tmp/opencv_export|/opt/opencv|" /usr/local/lib/python3.8/dist-packages/cv2/config.py \
    && cp /etc/openalpr/openalpr.conf.gpu /etc/openalpr/openalpr.conf \
    && sed -i 's|/tmp/dlib_export|/usr/local|' /usr/local/lib/pkgconfig/dlib-1.pc \
    && python3 -m wheel convert /usr/local/lib/python3.8/dist-packages/dlib-19.*.egg \
    # gross little hack to get the wheel to install
    && mv "./dlib-${DLIB_VERSION}.0-py38-cp38-linux_x86_64.whl" "./dlib-${DLIB_VERSION}.0-cp38-none-any.whl" \
    && python3 -m pip install "./dlib-${DLIB_VERSION}.0-cp38-none-any.whl" \
    && python3 -m pip install git+https://github.com/ageitgey/face_recognition_models distro requests \
    && rm -rf /usr/local/lib/python3.8/dist-packages/dlib-"${DLIB_VERSION}".0-py3.8-linux-x86_64.egg dlib-"${DLIB_VERSION}".0-cp38-none-any.whl

#################### OPEN VINO RUNTIME ####################
RUN mkdir /opt/intel

ENV INTEL_OPENVINO_DIR /opt/intel/openvino

COPY --from=build-env /opt/intel/ /opt/intel/

WORKDIR /thirdparty

ARG INSTALL_SOURCES="no"

ARG DEPS="tzdata \
          curl"

ARG LGPL_DEPS="g++ \
               gcc"
ARG INSTALL_PACKAGES="-c=python -c=core"


# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/apt/lists,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    apt-get update && \
    dpkg --get-selections | grep -v deinstall | awk '{print $1}' > base_packages.txt  && \
    apt-get install -y --no-install-recommends ${DEPS}


RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/apt/lists,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    apt-get update && apt-get reinstall -y ca-certificates && update-ca-certificates

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/apt/lists,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends ${LGPL_DEPS} && \
    ${INTEL_OPENVINO_DIR}/install_dependencies/install_openvino_dependencies.sh -y ${INSTALL_PACKAGES} && \
    if [ "$INSTALL_SOURCES" = "yes" ]; then \
      sed -Ei 's/# deb-src /deb-src /' /etc/apt/sources.list && \
      apt-get update && \
	  dpkg --get-selections | grep -v deinstall | awk '{print $1}' > all_packages.txt && \
	  grep -v -f base_packages.txt all_packages.txt | while read line; do \
	  package=$(echo $line); \
	  name=(${package//:/ }); \
      grep -l GPL /usr/share/doc/${name[0]}/copyright; \
      exit_status=$?; \
	  if [ $exit_status -eq 0 ]; then \
	    apt-get source -q --download-only $package;  \
	  fi \
      done && \
      echo "Download source for $(ls | wc -l) third-party packages: $(du -sh)"; fi && \
    rm /usr/lib/python3.*/lib-dynload/readline.cpython-3*-gnu.so


ENV HDDL_INSTALL_DIR=/opt/intel/openvino/runtime/3rdparty/hddl
ENV InferenceEngine_DIR=/opt/intel/openvino/runtime/cmake
ENV LD_LIBRARY_PATH=/opt/intel/openvino/runtime/3rdparty/hddl/lib:/opt/intel/openvino/runtime/3rdparty/tbb/lib:/opt/intel/openvino/runtime/lib/intel64:/opt/intel/openvino/tools/compile_tool
ENV OpenCV_DIR=/opt/opencv/cmake
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV PYTHONPATH=/opt/intel/openvino/python/python3.8:/opt/intel/openvino/python/python3
ENV TBB_DIR=/opt/intel/openvino/runtime/3rdparty/tbb/cmake
ENV ngraph_DIR=/opt/intel/openvino/runtime/cmake
ENV OpenVINO_DIR=/opt/intel/openvino/runtime/cmake
ENV INTEL_OPENVINO_DIR=/opt/intel/openvino
ENV PKG_CONFIG_PATH=/opt/intel/openvino/runtime/lib/intel64/pkgconfig

# setup Python
ENV PYTHON_VER python3.8

# runtime package
WORKDIR ${INTEL_OPENVINO_DIR}
ARG OPENVINO_WHEELS_VERSION=2022.3.0
ARG OPENVINO_WHEELS_URL
RUN if [ -z "$OPENVINO_WHEELS_URL" ]; then \
        ${PYTHON_VER} -m pip install --no-cache-dir openvino=="$OPENVINO_WHEELS_VERSION" ; \
    else \
        ${PYTHON_VER} -m pip install --no-cache-dir --pre openvino=="$OPENVINO_WHEELS_VERSION" --trusted-host=* --find-links "$OPENVINO_WHEELS_URL" ; \
    fi

WORKDIR ${INTEL_OPENVINO_DIR}/licensing
# Please use `third-party-programs-docker-runtime.txt` short path to 3d party file if you use the Dockerfile directly from docker_ci/dockerfiles repo folder
COPY dockerfiles/ubuntu20/third-party-programs-docker-runtime.txt ${INTEL_OPENVINO_DIR}/licensing

# for CPU

# for GPU

RUN apt-get update && apt-get install -y --no-install-recommends gpg gpg-agent && \
    curl https://repositories.intel.com/graphics/intel-graphics.key | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg && \
    echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu focal-legacy main' | tee  /etc/apt/sources.list.d/intel.gpu.focal.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
       intel-opencl-icd=22.43.24595.35+i538~20.04 \
       intel-level-zero-gpu=1.3.24595.35+i538~20.04 \
       level-zero=1.8.8+i524~u20.04 \
       ocl-icd-libopencl1 && \
       apt-get purge gpg gpg-agent --yes && apt-get --yes autoremove && \
       apt-get clean ; \
       rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*


# Post-installation cleanup and setting up OpenVINO environment variables
ENV LIBVA_DRIVER_NAME=iHD
ENV GST_VAAPI_ALL_DRIVERS=1
ENV LIBVA_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri





ARG ZMML_VERSION=master
# ZM ML Server Install
ARG CacheBust=1
#COPY . /opt/zm_ml/src
RUN set -x \
      && mkdir -p /opt/zm_ml/src \
      && cd /opt/zm_ml/src \
      && git clone https://github.com/baudneo/ZM_ML.git /opt/zm_ml/src \
      && git checkout "${ZMML_VERSION}" \
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
    ML_PROCESSORS='cpu,gpu[cuda],tpu[pycoral 3.8]'

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