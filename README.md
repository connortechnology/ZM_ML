# ZoneMinder Machine Learning Library

This is a project aiming to update how [ZoneMinder](https://github.com/ZoneMinder/zoneminder) Object Detection works.
A server and client are supplied to allow for easy integration with ZoneMinder.

The server is an asynchronous [FastAPI](https://fastapi.tiangolo.com/) based REST API that runs Machine Learning models on supplied images, the server can be run on the ZoneMinder host or a remote host.

The client is installed on the ZoneMinder machine, grabs and sends images to a ZM_ML server for
inference and then processes the results to annotate images, create animations and send notifications.

## Thanks

- [@pliablepixels](https://github.com/pliablepixels) for [zmNinja](https://github.com/ZoneMinder/zmNinja), [zmeventnotification](https://github.com/ZoneMinder/zmeventnotification), [mlapi](https://github.com/ZoneMinder/mlapi) and [PyZM](https://github.com/ZoneMinder/pyzm).
- [@connortechnology](https://github.com/connortechnology) for their work on [ZoneMinder](https://zoneminder.com)

This project is based on the work of [@pliablepixels](https://github.com/pliablepixels).

## Prerequisites

- ZoneMinder 1.37.5+ (*EventStartCommand* is **REQUIRED**)
- Python 3.8+ (3.9 recommended)
  - psutil
  - requests
- OpenCV (Contrib) 4.2.0+ (4.5.4+ recommended)
- NumPy 1.19.5+ (1.21.2+ recommended)

#### Notes:

1. [**EventStartCommand**/**EventEndCommand**](https://zoneminder.readthedocs.io/en/latest/userguide/definemonitor.html#recording-tab:~:text=events%20are%20recorded.-,Event%20Start%20Command,the%20command%20will%20be%20the%20event%20id%20and%20the%20monitor%20id.,-Viewing%20Tab) is what runs the object detection script. Before, SHM was polled every \<X> seconds to see if a new event had been triggered.

## Installation

### **Bootstrap**
- Download the file first and read it before running
    - ```bash 
      curl -H 'Cache-Control: no-cache' -s https://raw.githubusercontent.com/baudneo/ZM_ML/master/examples/bootstrap > bootstrap && chmod +x bootstrap && ./bootstrap --help
      ```
- This will attempt to install git if it isn't already installed, clone the repo and install ZM_ML using defaults (Install client only).
    - ```bash
      curl -H 'Cache-Control: no-cache' -s https://raw.githubusercontent.com/baudneo/ZM_ML/master/examples/bootstrap | bash /dev/stdin --help 
      ```


### **Manual**
```bash


## Server

1. Based on [FastAPI](https://fastapi.tiangolo.com/ "FastAPI") (With all the Pydantic goodness!)
2. OpenCV DNN for CPU/GPU
3. [OpenALPR](https://github.com/openalpr/openalpr) local binary supported (Must compile OpenALPR with CUDA for GPU support)
4. Cloud ALPR integrations. [[See notes](#_cloud-alpr_)]
5. [pycoral](https://github.com/google-coral/pycoral) (tflite) for TPU support.
6. DLib based [face-recognition](https://github.com/ageitgey/face_recognition) (GPU Recommended)
7. Run locally on ZoneMinder machine or deploy to a remote machine.

### Docker

An amd64 image will be created that has OpenCV/face_recognition with GPU support and also includes TPU libs. No idea if other architectures will be supported.

### _NVIDIA GPU Accelerated Server_

For GPU acceleration, it is required to compile OpenCV with CUDA support. This includes knowing the 'Compute Capability' [_CUDA_ARCH_BIN_] of the cards you want to run the server on and also installing cuDNN libraries.
**_To access cuDNN packages you will need to create a NVIDIA developers account._**

### _Coral EdgeTPU Accelerated Server_

For TPU acceleration you will need to install the [edgetpu libraries](https://coral.ai/docs/accelerator/get-started/#runtime-on-linux) and [install pycoral](https://coral.ai/docs/accelerator/get-started/#pycoral-on-linux)  ([see Notes for Python3.10 TPU support](#server-notes "Notes")).

### _Cloud ALPR_

For Cloud ALPR you will need to create an account with the service you want to use [Plate Recognizer, OpenALPR(WIP)]. OpenALPR Cloud [now knows as Rekor] has changed since zmeventnotification days and needs to be rewritten.

#### Server Notes:

1. AMD GPU's are __NOT__ supported.
2. Intel ARC/iGPU's are __CURRENTLY NOT__ supported. (this may change)
3. If you do not need GPU acceleration you can install OpenCV using pip. (`pip install opencv-contrib-python`)
4. pycoral recently released wheels for Python3.10 See [here](https://github.com/google-coral/pycoral/issues/85#issuecomment-1305826142 "Pycoral 3.10 wheels")
5. I am working on a script to make building OpenCV with GPU support easier.

---

## Client

- OpenCV and Numpy are **_REQUIRED_** for image processing.
- [Pydantic](https://pydantic-docs.helpmanual.io/) is used for data parsing.

The client needs a script to initialize and run it. A few examples are provided in the 'examples' folder.

- EventStartCommand / EventEndCommand - [eventstart.py](./examples/eventstart.py) is an example to use with a bash helper [script](./examples/EventStartCommand.sh)
- MQTT (WIP) - [MQTT.py](./examples/MQTT.py) is an example to use with ZoneMinder's built in MQTT event notifications.
- 'Continuous' (WIP) - [continuous.py](./examples/continuous.py) is an example that will check configured monitors every \<x> seconds.

---

## WebSocket Server (WIP)

The websocket server is a legacy supporting server for zmNinja. Its sole purpose is to obtain tokens to send push notifications to zmNinja clients (Android/iOS) and retain some legacy functions in zmNinja.

---

## Environment Variables (These *MAY* change)

### Shared between Client and Server

---

### Client Only

---

#### Base

- __ZM_CONF_DIR__ - Path to where ZM config files are located. Default: */etc/zm*
- __ML_CLIENT_CONF_DIR__ - Path to where ZM_ML Client configs are located. Default: */etc/zm/ml*
- __ML_CLIENT_CONF_FILE__ - Path to ZM_ML Client config file. Default: */etc/zm/ml/client.conf*
- __ML_CLIENT_VAR_DATA_DIR__ - Path to where ZM_ML data is stored. Default: */var/lib/zm_ml*
- __ML_CLIENT_EVENT_START__ - Absolute path to ZM_ML Client script. Default: */usr/bin/zm_ml_client*

#### DB

- __ML_DBHOST__ - Hostname of the MySQL database. Default: *localhost*
- __ML_DBNAME__ - Name of the MySQL database. Default: *zm*
- __ML_DBUSER__ - Username for the MySQL database. Default: *zmuser*
- __ML_DBPASS__ - Password for the MySQL database. Default: *zmpass*
- __ML_DBDRIVER__ - SQLAlchemy DB Driver. Default: *mysql+pymysql*

### Server Only

---

#### Base

- __ML_SERVER_CONF_DIR__ - Path to where ZM_ML Server configs are located. Default: /etc/zm/ml
- __ML_SERVER_CONF_FILE__ - Path to ZM_ML Server config file. Default: /etc/zm/ml/server.conf
