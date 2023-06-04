# Support

[![Join Slack](https://github.com/ozonesecurity/ozonebase/blob/master/img/slacksm.png?raw=true)](https://join.slack.com/t/zoneminder-chat/shared_invite/enQtNTU0NDkxMDM5NDQwLTdhZmQ5Y2M2NWQyN2JkYTBiN2ZkMzIzZGQ0MDliMTRmM2FjZWRlYzUwYTQ2MjMwMTVjMzQ1NjYxOTdmMjE2MTE "Join Slack")
<a href="https://discord.gg/tHYyP9k66q" title="Join Discord Server"><img src="https://assets-global.website-files.com/6257adef93867e50d84d30e2/636e0a6a49cf127bf92de1e2_icon_clyde_blurple_RGB.png" alt="drawing" width="50"/></a>


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
- Python 3.8+ (3.9 recommended) **[3.10+ DOES NOT SUPPORT TPU ATM]**
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
### Docker
See [below](https://github.com/baudneo/ZM_ML#server---docker) for **SERVER** Docker image and instructions. **Client** has no docker image.

### Manual Install
See the Wiki for [Manual Installation](https://github.com/baudneo/ZM_ML/wiki/Manual-Installation) instructions.

### Bootstrap (WIP)
**NOTE: bootstrap is a WIP**

*NOTE 2: use `--dry-run` to see what will be installed*

Bootstrap is for non docker installs.

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

## _Server - Docker_
See the Wiki [Docker](https://github.com/baudneo/ZM_ML/wiki/Docker) page for pre requisites, tags and instructions.

A x86_64 image has been created that has:
- OpenCV
- face-recognition [DLib] (GPU Recommended)
- OpenALPR local SDK binary.
- CUDA GPU support for the 3 above.
- TPU support

```bash
   docker pull docker.io/baudneo/zm_ml:server-full
   ```

### Example docker-compose and .env files
Located in the [docker](./docker) folder:
- [docker-compose.yml](docker/docker-compose.yml)
- [server.env](docker/server.env)

## _NVIDIA GPU Acceleration_

See the Wiki [Manual Installation Server GPU](https://github.com/baudneo/ZM_ML/wiki/Manual-Installation#gpu-support) page for pre requisites and instructions.

## _Coral EdgeTPU Accelerated Server_

See the Wiki [Manual Installation Server TPU](https://github.com/baudneo/ZM_ML/wiki/Manual-Installation#tpu-support) page for pre requisites and instructions.

## _Cloud ALPR_

**NOTE:** Plate Recognizer has a free tier!

For Cloud ALPR you will need to create an account with the service you want to use 
- Plate Recognizer is partially supported
- OpenALPR CLOUD - not supported, yet. 
  - OpenALPR Cloud (*now knows as Rekor*) has changed since zmeventnotification days and needs to be rewritten.
If someone can give me access to their Rekor API I can update the code.

## Server Notes:

1. AMD GPU's are __NOT__ supported.
2. Intel ARC/iGPU's are __CURRENTLY NOT__ supported. (this may change)
3. If you do not need GPU acceleration you can install OpenCV using pip. (`pip install opencv-contrib-python`)
4. pycoral recently released wheels for Python3.10 See [here](https://github.com/google-coral/pycoral/issues/85#issuecomment-1305826142 "Pycoral 3.10 wheels")
   - **NOTE:** It is recommended to use python **3.8** *or* **3.9** if you want to use TPU acceleration.
5. I am working on a script to make building OpenCV with GPU support easier.

---

# Client
The client side of ZM ML uses a python script that uses a shell script wrapper to kick it off using ZoneMnider EventStartCommand option.
This means ZM kicks of the ML chain instead of the ML chain scanning SHM looking for an event, more efficient!

The client grabs images, sends the images to mlapi servers, filters detected responses, post processes images and sends notifications. All the heavy computation of ML models is done by the server!

## Client Pre-requisites
- Client **MUST** be installed on same host as ZM server. Multi-server ZM installs will require a client install on each server.
- OpenCV is **_REQUIRED_** for image processing, you can use `opencv-contrib-python`
- `libgeos-dev` : system package (used for the Shapely python module; Polygons)
- `gifsicle` : system package (used to optimize GIFs; makes file size much smaller)

## Client Info
The client needs a script to initialize and run it. A few examples are provided in the '[examples](examples)' folder.

- EventStartCommand / EventEndCommand - [eventproc.py](./examples/eventproc.py) is an example to use with a bash helper [script](./examples/EventStartCommand.sh)
  - The helper script is required due to ZoneMinder's `EventStartCommand` only passing Monitor ID and Event ID to the client script. 
- MQTT (WIP) - Once ZM builds out MQTT support more I will work on using MQTT to trigger the ML chain.
- Continuous (WIP) - Pull images from <x> monitors and run obj det on them, create an event if a filtered hit happens. 

---

## WebSocket Server (WIP)

The websocket server is a legacy supporting server for zmNinja. Its sole purpose is to obtain tokens to send push notifications to zmNinja clients (Android/iOS) and retain some legacy functions in zmNinja.

---

# Environment Variables
Check back later for a list of environment variables and their explanation.