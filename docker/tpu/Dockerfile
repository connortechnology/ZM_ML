FROM baudneo/opencv:testing as opencv

RUN set -x \
    && sh -c 'echo "/tmp/opencv_export/lib" >> /etc/ld.so.conf.d/opencv.conf' \
    && ldconfig \
    && echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
                  tee /etc/apt/sources.list.d/coral-edgetpu.list \
	&& curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
	&& apt-get update \
	# Modify pycoral and tflite-runtime dependencies
	&& mkdir -p /pycoral-deps/pycoral /pycoral-deps/tflite \
	&& apt-get download \
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
    && mkdir -p /tmp/tpu_debs \
    && cp python3-pycoral.deb python3-tflite-runtime.deb gasket-dkms.deb libedgetpu1-std.deb /tmp/tpu_debs



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
      && rm -rf /pycoral
