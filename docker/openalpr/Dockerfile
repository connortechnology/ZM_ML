ARG ALPR_METHOD=BRANCH
ARG ALPR_VERSION=master
# off / on
ARG ALPR_GPU=on

FROM baudneo/opencv:testing as opencv

COPY ./docker/openalpr/build_openalpr.sh /tmp/build_openalpr

RUN --mount=type=cache,target=/var/cache/apt,target=/var/apt/lists,target=/root/.cache/pip set -x \
        && chmod +x /tmp/build_openalpr \
        && apt-get update \
        && apt-get install -y --no-install-recommends \
          libtesseract-dev \
          libleptonica-dev \
          liblog4cplus-dev \
          libcurl3-dev \
        && /tmp/build_openalpr \
        && mkdir -p /alpr_test \
        && wget "http://plates.openalpr.com/h786poj.jpg" -O /alpr_test/lp.jpg \
        && echo "alpr /alpr_test/lp.jpg" > /alpr_test/alpr_test \
        && chmod +x /alpr_test/alpr_test \
