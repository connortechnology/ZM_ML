# ZoneMinder Machine Learning Library 
This is a project aiming to update how ZoneMinder Object Detection works. 
A server and client are supplied to allow for easy integration with ZoneMinder.

## Prerequisites
- ZoneMinder 1.36.0 or later (EventStartCommand / MQTT events are needed)
- Python 3.6+ (pycoral only has support up to 3.10)
- OpenCV (Contrib) 4.2.0+ (4.5.4+ recommended) (yolov7, yolov7x - [**_REQUIRE_** Custom Built OPENCV](https://github.com/opencv/opencv/issues/22409 "SiLu Activation"))
- NumPy 1.19.5+ (1.21.2+ recommended)

## Server
- Based on FastAPI
- OpenCV DNN for CPU/GPU
- OpenALPR local binary supported (Must compile OpenALPR with CUDA support)
- Cloud ALPR integrations (OpenALPR, Plate Recognizer)
- pycoral (tflite) for TPU (object+face).
- DLib based face-recognition (GPU Recommended)
- Run locally on ZoneMinder machine or deploy to a remote machine. 
### _NVIDIA GPU Accelerated Server_
For GPU acceleration, it is required to compile OpenCV with CUDA support. This includes knowing the 'Compute Capability' [_CUDA_ARCH_BIN_] of the cards you want to run the server on and also installing cuDNN libraries. 
**_To access cuDNN packages you will need to create an NVIDIA developers account._**
### _Coral EdgeTPU Accelerated Server_
For TPU acceleration you will need to install the [edgetpu libraries](https://coral.ai/docs/accelerator/get-started/#runtime-on-linux) and [install pycoral](https://coral.ai/docs/accelerator/get-started/#pycoral-on-linux)  ([see Notes for Python3.10 TPU support](#server-notes "Notes")).

#### Server Notes:
1. AMD GPU's are __NOT__ supported.
2. Intel iGPU's are __CURRENTLY NOT__ supported. (this may change)
3. If you do not need GPU acceleration you can install OpenCV using pip. (opencv-contrib-python)
4. pycoral recently released wheels for Python3.10 See [here](https://github.com/google-coral/pycoral/issues/85#issuecomment-1305826142 "Pycoral 3.10 wheels")
5. YOLOv7, YOLOv7X Require a small change in OpenCV source code to make use of the 'SiLu Activation' -> [Custom Build OPENCV](https://github.com/opencv/opencv/pull/22410/files "OpenCV add SiLu Activation")
6. I am working on a script to make building OpenCV with GPU support easier.

## Client
The client is installed on the ZoneMinder machine, grabs and sends images to a ZM_ML server for processing and then processes the results to annotate images, create animations and send notifications.

The client needs a script to initialize and run it. A few examples are provided in the 'examples' folder.
- EventStartCommand / EventEndCommand - [eventstart.py](./examples/eventstart.py) is an example to use with a bash helper [script](./examples/EventStartCommand.sh)


## Environment Variables
### Base
- __ZM_CONF_DIR__ - Path to where ZM config files are located. Default: /etc/zm
- __ML_CLIENT_CONF_DIR__ - Path to where ZM_ML Client configs are located. Default: /etc/zm/ml
- __ML_SERVER_CONF_DIR__ - Path to where ZM_ML Server configs are located. Default: /etc/zm/ml
- __ML_CLIENT_VAR_DATA_DIR__ - Path to where ZM_ML data is stored. Default: /var/lib/zm_ml
- __ML_CLIENT_CONF_FILE__ - Path to ZM_ML Client config file. Default: /etc/zm/ml/client.conf 
### DB
- __ML_DBHOST__ - Hostname of the MySQL database. Default: localhost
- __ML_DBNAME__ - Name of the MySQL database. Default: zm
- __ML_DBUSER__ - Username for the MySQL database. Default: zmuser
- __ML_DBPASS__ - Password for the MySQL database. Default: zmpass
- __ML_DBDRIVER__ - SQLAlchemy DB Driver. Default: mysql+pymysql
