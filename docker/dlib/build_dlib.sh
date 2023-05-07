#!/usr/bin/env bash

# Remove the '.' from the CUDA_ARCH_BIN :: 7.5 => 75 6.0,7.1,7.5 = 60,71,75
ARCH_BIN=$(python3 -c "import os;x=os.environ['CUDA_ARCH_BIN']; print(x.replace('.',''))")

if [[ -z "$DLIB_VERSION" ]]; then
  echo "DLIB_VERSION not set, exiting..."
  exit 1
  elif [[ -z "$DLIB_METHOD" ]]; then
    echo "DLIB_METHOD not set, exiting..."
    exit 1
  else
    echo "DLIB_VERSION: $DLIB_VERSION"
    echo "DLIB_METHOD: $DLIB_METHOD"
fi


if [[ "$DLIB_METHOD" == "release"  ]]; then
  echo "DLib release method requested, downloading tagged release $DLIB_VERSION from github..."
  wget -c -q https://github.com/davisking/dlib/archive/refs/tags/v${DLIB_VERSION}.tar.gz -O dlib.tgz
  tar xf dlib.tgz
  mv dlib-*/dlib dlib
  rm dlib.tgz
  elif [[ "$DLIB_METHOD" == "branch"  ]]; then
    git clone --single-branch --branch "${DLIB_VERSION}" https://github.com/davisking/dlib /dlib
  else
    echo "Unknown DLIB_METHOD: $DLIB_METHOD, exiting..."
    exit 1
fi

mkdir -p /dlib/build /tmp/dlib/export /tmp/dlib/python_bindings /tmp/face_recognition
cd /dlib/build

cmake \
  -D CMAKE_INSTALL_PREFIX=/tmp/dlib/export \
  -D CMAKE_PREFIX_PATH=/tmp/opencv_export \
  -D DLIB_USE_CUDA_COMPUTE_CAPABILITIES=${ARCH_BIN} \
  -D USE_AVX_INSTRUCTIONS="${DLIB_USE_AVX:-0}" \
  -D DLIB_USE_CUDA="${DLIB_USE_CUDA:-0}" \
  ..

time cmake --build . --target install --config Release

cd /dlib
time python3.9 setup.py install \
          --prefix=/tmp/dlib/python_bindings \
          --set USE_AVX_INSTRUCTIONS="${DLIB_USE_AVX:-0}" \
          --set DLIB_USE_CUDA="${DLIB_USE_CUDA:-0}" \
          

time python3.9 -m pip install face_recognition \
          --no-deps \
          --target /tmp/face_recognition

# remove dlib from face-recognition requirements and install its other deps
wget https://raw.githubusercontent.com/ageitgey/face_recognition/master/requirements.txt -O ./req.txt
dlib_reqs=$(cat <<EOF
from pathlib import Path
txt = Path("./req.txt").read_text().split("\n")
for line in txt:
    if line.startswith("dlib"):
        txt.pop(txt.index(line))
Path("./req.txt").write_text("\n".join(txt))
EOF
)

python3 -c "${dlib_reqs}"
cat ./req.txt

python3 -m pip install -r ./req.txt \
          --no-cache-dir \
          --target /tmp/face_recognition