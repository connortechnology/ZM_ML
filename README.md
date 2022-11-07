# ZoneMinder Machine Learning Library 
This is a project aiming to update how ZoneMinder Object Detection works. 
A server and client are supplied to allow for easy integration with ZoneMinder. 
## Server
- Based on FastAPI
- OpenCV DNN for CPU/GPU and pycoral for TPU. 
- Run on local zoneminder machine or on a remote machine. 
### GPU
For GPU acceleration you will need to compile OpenCV with GPU support (This includes knowing the 'Compute Capability' [_CUDA_ARCH_BIN_] of the cards you want to run the server on) 

## Client
The client is installed on the ZoneMinder machine and sends images to the server for processing and then processes the results to annotate images, create animations and send notifications.


No ML models are ran by the client, instead the client will send the image to the server and the server will run the ML model and return the results. The client will take the results and filter them, post-process the image for notifications and ZM storage (objdetect.jpg), optionally create gif/mp4 animations of the event and send image/animation notifications of the results.

## Environment Variables
### Base
- __ZM_ML_CONF_PATH__ - Path to where ZM ML config files are located. Default: /etc/zm
### DB
- __ZM_ML_DBHOST__ - Hostname of the MySQL database. Default: localhost
- __ZM_ML_DBNAME__ - Name of the MySQL database. Default: zm
- __ZM_ML_DBUSER__ - Username for the MySQL database. Default: zmuser
- __ZM_ML_DBPASS__ - Password for the MySQL database. Default: zmpass
- __ZM_ML_DBDRIVER__ - SQLAlchemy DB Driver. Default: mysql+pymysql
