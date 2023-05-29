# Please support ZoneMinder if you can!
If you use software from the [ZoneMinder organization](https://github.com/ZoneMinder)
please consider [sponsoring ZoneMinder](https://github.com/sponsors/ZoneMinder). ZoneMinder has been free with
a paid support option and paid bounties but, it has come to the communities attention that the main/only dev of
ZM needs help with bringing cash flow in. If you have the means, please consider sponsoring ZoneMinder, Thanks!

# ZoneMinder Machine Learning Library (ZM ML)
**Try the x86_64 GPU/TPU accelerated Server docker image!** [docker.io/baudneo/zm_ml:server-full](https://hub.docker.com/r/baudneo/zm_ml)
### NOTE
**This software is currently in alpha stage, expect issues. That said, it is fairly fast. YMMV.**

This is a project aiming to update how [ZoneMinder](https://github.com/ZoneMinder/zoneminder) Object Detection works.
A server and client are supplied to allow for easy integration with ZoneMinder.

The server is an asynchronous [FastAPI](https://fastapi.tiangolo.com/) based REST API that runs Machine Learning 
models on supplied images, the server can be run on the ZoneMinder host or a remote host and can be hardware
accelerated using an NVIDIA GPU or Coral.ai EdgeTPU (USB confirmed working, other variants untested). 
The Server offers object detection, face detection, face recognition and Automatic License Plate Recognition (ALPR). 
There are plans to add TensorRT, OpenVINO, PyTorch, [deepface](https://github.com/serengil/deepface) and TensorFlow support.

The client is an asynchronous (mostly) script installed on the ZoneMinder machine, grabs and sends images to a ZM ML server for
inference and then processes the results to annotate images, create animations and send notifications, if configured to do so.

## Thanks

- [@pliablepixels](https://github.com/pliablepixels) for [zmNinja](https://github.com/ZoneMinder/zmNinja), [zmeventnotification](https://github.com/ZoneMinder/zmeventnotification), [mlapi](https://github.com/ZoneMinder/mlapi) and [PyZM](https://github.com/ZoneMinder/pyzm).
- [@connortechnology](https://github.com/connortechnology) for their work on [ZoneMinder](https://zoneminder.com)

### -> This project is based on the work of [@pliablepixels](https://github.com/pliablepixels). <-


## Prerequisites for Server and Client

- ZoneMinder 1.37.5+ (*EventStartCommand* is **REQUIRED**)
  - debian based distros can [build a .deb package for the 1.37 dev branch](https://gist.github.com/baudneo/d352c5a944a5d1371c9dfe455056e0a2)
- Python 3.8+ (3.9 recommended)
- Python packages required by the [install script](examples/install.py)
  - `psutil`
  - `request`
  - `tqdm`
  - `distro`
- OpenCV (Contrib) 4.2.0+ (4.7.0+ recommended) with Python3 bindings.
  - OpenCV is not installed by default due to possible GPU acceleration by user compiled OpenCV. 

#### Notes:

1. [**EventStartCommand**/**EventEndCommand**](https://zoneminder.readthedocs.io/en/latest/userguide/definemonitor.html#recording-tab:~:text=events%20are%20recorded.-,Event%20Start%20Command,the%20command%20will%20be%20the%20event%20id%20and%20the%20monitor%20id.,-Viewing%20Tab) is what runs the object detection script. Before, SHM was polled every \<X> seconds to see if a new event had been triggered.

## Installation
### Manual Install
See the Wiki for [Manual Installation](https://github.com/baudneo/ZM_ML/wiki/Manual-Installation) instructions.

**Try the GPU/TPU accelerated** [Docker image](https://hub.docker.com/repository/docker/baudneo/zm_ml)!

### Bootstrap (WIP)
**NOTE: bootstrap is a WIP**
- Download the file first and read it before running
    - ```bash 
      curl -H 'Cache-Control: no-cache' -s https://raw.githubusercontent.com/baudneo/ZM_ML/master/examples/bootstrap > bootstrap && chmod +x bootstrap && ./bootstrap --help
      ```
- This will attempt to install git if it isn't already installed, clone the repo and install ZM ML using defaults (Install client only).
    - ```bash
      curl -H 'Cache-Control: no-cache' -s https://raw.githubusercontent.com/baudneo/ZM_ML/master/examples/bootstrap | bash /dev/stdin --help 
      ```

# Server
1. Based on [FastAPI](https://fastapi.tiangolo.com/ "FastAPI") (With all the Pydantic goodness!)
2. OpenCV DNN for CPU/GPU
3. [OpenALPR](https://github.com/openalpr/openalpr) local binary supported (Must compile OpenALPR with CUDA for GPU support)
4. Cloud ALPR integrations. [[See notes](#_cloud-alpr_)]
5. [pycoral](https://github.com/google-coral/pycoral) (tflite) for TPU support.
6. DLib based [face-recognition](https://github.com/ageitgey/face_recognition) (GPU Recommended)
7. Run locally on ZoneMinder machine or deploy to a remote machine.
8. Docker images! [Docker Hub](https://hub.docker.com/repository/docker/baudneo/zm_ml)

## Server - Docker

A x86_64 image has been created that has OpenCV/face_recognition [DLib] with CUDA GPU support and also includes TPU libs.
The plan is to have multiple other images for multiple archs. A RPi with a USB TPU doesnt need CUDA/OpenVINO
accelerated OpenCV, a Jetson nano would need CUDA though. So, it's a WIP.

### GPU/TPU x86_64 (amd64) Server docker image Pre-requisites
- GPU:
  - Nvidia drivers installed on the docker host (CUDA is not required on the docker host)
  - Nvidia docker-container-toolkit installed and bound to the docker runtime on the docker host
  - "Deploy" GPU(s) to the docker container using the `--gpus all` flag via `docker run` or follow the example in the supplied [docker-compose.yml](docker/docker-compose.yml) file.
- TPU:
  - Docker host may need to install coral libs with udev rules and run a detection on boot to 'init' the USB TPU for the Google udev rules to change the vendorId.
  - Testing is needed to see if the host can not have any TPU related libs and let the container handle everything.
  - Testing required to see if TPU passthrough will work in an unprivileged container
  - Pass through `/dev/bus/usb` to allow the TPU to be detected by the container (can narrow down bus/id but those can change on reboots)

### docker-compose.yaml and server.env files

Located in the [docker](./docker) folder:
- [docker-compose.yaml](docker/docker-compose.yml)
- [server.env](docker/server.env)
```bash
docker pull docker.io/baudneo/zm_ml:server-full
```

## _NVIDIA GPU Accelerated Server_

For GPU acceleration, it is required to compile OpenCV with CUDA support. This includes knowing the 'Compute Capability' [_CUDA_ARCH_BIN_] of the cards you want to run the server on and also installing cuDNN libraries.
**_To access cuDNN packages you will need to create a NVIDIA developers account._**

## _Coral EdgeTPU Accelerated Server_

For TPU acceleration you will need to install the [edgetpu libraries](https://coral.ai/docs/accelerator/get-started/#runtime-on-linux) and [install pycoral](https://coral.ai/docs/accelerator/get-started/#pycoral-on-linux)  ([see Notes for Python3.10 TPU support](#server-notes "Notes")).

## _Cloud ALPR_

For Cloud ALPR you will need to create an account with the service you want to use [Plate Recognizer, OpenALPR(WIP)]. OpenALPR Cloud [now knows as Rekor] has changed since zmeventnotification days and needs to be rewritten.

## Server Notes:

1. AMD GPU's are __NOT__ supported.
2. Intel ARC/iGPU's are __CURRENTLY NOT__ supported. (this may change)
3. If you do not need GPU acceleration you can install OpenCV using pip. (`pip install opencv-contrib-python`)
4. pycoral recently released wheels for Python3.10 See [here](https://github.com/google-coral/pycoral/issues/85#issuecomment-1305826142 "Pycoral 3.10 wheels")
5. I am working on a script to make building OpenCV with GPU support easier.

---

# Client
The client side of ZM ML is a simple python script that uses a shell script wrapper to kick it off using ZoneMnider EventStartCommand option.
This means ZM kicks of the ML chain instead of the ML chain scanning SHM looking for an event.

The client grabs images, sends the images to mlapi servers, filters detected responses, post processes images and sends notifications. All the heavy lifting is done by the server!

## Client Pre-requisites
- Client **MUST** be installed on same host as ZM server. Multi-server ZM installs will require a client install on each server.
- OpenCV is **_REQUIRED_** for image processing, you can use `opencv-contrib-python`
- `libgeos-dev` : system package (used for the Shapely python module; Polygons)
- `gifsicle` : system package (used to optimize GIFs; makes file size much smaller)

## Client Info
The client needs a script to initialize and run it. A few examples are provided in the '[examples](examples)' folder.

- EventStartCommand / EventEndCommand - [eventproc.py](./examples/eventproc.py) is an example to use with a bash helper [script](./examples/EventStartCommand.sh)
  - The helper script is required due to ZoneMinder's `EventStartCommand` only passing Monitor ID and Event ID to the client script. 
- MQTT (WIP) - [MQTT.py](./examples/MQTT.py) is an example to use with ZoneMinder's built in MQTT notifications. (Once ZM builds out MQTT support more I will work on this)
- 'Continuous' (WIP) - [continuous.py](./examples/continuous.py) is an example that will check configured monitors every \<x> seconds. [YOLO-NAS ?]

---

## WebSocket Server (WIP)

The websocket server is a legacy supporting server for zmNinja. Its sole purpose is to obtain tokens to send push notifications to zmNinja clients (Android/iOS) and retain some legacy functions in zmNinja.

---

# Environment Variables
Check back later for a list of environment variables and their explanation.