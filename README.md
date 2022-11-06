# ZoneMinder ML Client 
This is a project aiming to update how ZoneMinder Object Detection works. This is the __client__ end that will source images for a specific monitor and send an HTTP object detection request to an instance of ZoneMinder Neo-MLAPI (Server). 

No ML models are ran by the client, instead the client will send the image to the server and the server will run the ML model and return the results. The client will take the results and filter them, post-process the image for notifications and ZM storage (objdetect.jpg), optionally create gif/mp4 animations of the event send notifications the results to the ZM API.

# Environment Variables
## Base
- __ZM_ML_CONF_PATH__ - Path to where ZM ML config files are located. Default: /etc/zm
## DB
- __ZM_ML_DBHOST__ - Hostname of the MySQL database. Default: localhost
- __ZM_ML_DBNAME__ - Name of the MySQL database. Default: zm
- __ZM_ML_DBUSER__ - Username for the MySQL database. Default: zmuser
- __ZM_ML_DBPASS__ - Password for the MySQL database. Default: zmpass
- __ZM_ML_DBDRIVER__ - SQLAlchemy DB Driver. Default: mysql+pymysql
