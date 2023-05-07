#! /usr/bin/env bash
mkdir -p /tmp/openalpr/export

if [[ "$ALPR_METHOD" == "BRANCH" ]]; then
  git clone https://github.com/openalpr/openalpr.git /opt/openalpr --branch ${ALPR_VERSION} --single-branch
elif [[ "$ALPR_METHOD" == "RELEASE" ]]; then
    wget https://github.com/openalpr/openalpr/archive/refs/tags/v"$ALPR_VERSION".tar.gz -O alpr.tgz
    tar -xzf alpr.tgz
#    mkdir -p /opt/openalpr
    mv openalpr* /opt/openalpr
    else
      echo "Invalid ALPR_METHOD ($ALPR_METHOD) specified, exiting..."
      exit 1
fi

cd /opt/openalpr/src
mkdir build
cd build
cp /opt/openalpr/config/openalpr.conf.defaults /tmp/etc/openalpr/openalpr.conf.gpu
[[ "$ALPR_GPU" == "on"  ]] && sed -i 's/detector =.*/detector = lbpgpu/g' /tmp/etc/openalpr/openalpr.conf.gpu
sed -i "s|\${CMAKE_INSTALL_PREFIX}|/usr|" /tmp/etc/openalpr/openalpr.conf.gpu

cmake \
  -D CMAKE_INSTALL_PREFIX:PATH=/tmp/alpr_export \
  -D CMAKE_PREFIX_PATH=/tmp/opencv_export \
  -D CMAKE_INSTALL_SYSCONFDIR:PATH=/tmp/etc \
  -D WITH_GPU_DETECTOR="${ALPR_GPU:-no}" \
   ..

time make -j$(nproc)
make install

cd openalpr/src/bindings/python/
python3 setup.py install --prefix=/tmp/openalpr/python_bindings
cp -r /opt/openalpr/runtime_data /tmp/openalpr
