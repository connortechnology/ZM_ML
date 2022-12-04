import logging
import ssl
from datetime import datetime
from time import time
from pathlib import Path
from typing import Union, Optional, Any

import numpy as np
import paho.mqtt.client as mqtt_client

from .. import get_global_config
from ...Shared.configs import GlobalConfig

g: GlobalConfig

wasConnected = False
Connected = False  # global variable for the state of the connection

logger = logging.getLogger("ML-Client")


def on_log(client, userdata, level, buf):
    logger.debug(1, f"mqtt:paho_log: {buf}")


def on_connect(client, userdata, flags, rc):
    lp = "mqtt:connect: "
    if rc == 0:
        logger.debug(f"{lp} connected to broker with flags-> {flags}")
        global Connected, wasConnected  # Use global variable
        Connected = True  # Signal connection
        wasConnected = True
    else:
        logger.debug(f"{lp} connection failed with result code-> {rc}")


def on_publish(client, userdata, mid):
    logger.debug(f"mqtt:on_publish: message_id: {mid = }")


class MQTT:
    """Create an MQTT object to publish
    config: (dict)
    config_file: path to a config file to read
    secrets: same as config but for secrets

    """

    def __init__(
        self,
    ):
        global g
        (
            self.image,
            self.path,
            self.conn_wait,
            self.client,
            self.tls_ca,
            self.connected,
            self.config,
            self.secrets,
            self.conn_time,
        ) = (None, None, None, None, None, None, None, None, None)
        g = get_global_config()
        self.ssl_cert = ssl.CERT_REQUIRED  # start with strict cert checking/verification of CN
        cfg = self.config = g.config.notifications.mqtt
        self.sanitize = g.config.logging.sanitize
        self.sanitize_str = g.config.logging.sanitize_string
        self.tls_allow_self_signed = cfg.allow_self_signed
        # config and secrets
        self.user = self.config.user
        self.password = self.config.pass_

        self.broker = self.config.broker
        self.port = cfg.port
        if not self.port:
            self.port = 1883
            if cfg.tls_ca:
                self.port = 8883
        self.tls_ca = cfg.tls_ca
        if self.tls_allow_self_signed:
            self.ssl_cert = ssl.CERT_NONE
        self.tls_insecure = self.config.tls_insecure
        self.mtls_cert = self.config.tls_cert
        self.mtls_key = self.config.tls_key
        self.retain = self.config.retain
        self.qos = self.config.qos
        self.client_id = "zm_ml-"
        self._image: Any = None

    @staticmethod
    def is_connected():
        return Connected

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, image: Union[np.ndarray, bytes]):
        if isinstance(image, np.ndarray):
            self._image = self.create_ml_image(image)
        elif isinstance(image, bytes):
            self._image = image

    def create_ml_image(self, image, _type="byte"):
        """Prepares an image to be published, tested on jpg and gif so far. Give it a cv2.imencode image and
         it then wraps the image in a bytearray and stores it
        internally waiting to publish to mqtt topic
        """
        if _type == "byte":
            logger.debug(
                f"mqtt:grab_image: converting to byte array"
            )
            image = bytearray(image.to_bytes())
        else:
            import base64

            logger.debug(
                f"mqtt:grab_image: converting to BASE64"
            )
            image = base64.b64encode(image.to_bytes()).decode("utf-8")
        return image

    def get_options(self):
        return {
            "client_id": self.client_id,
            "broker": self.broker,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "retain_published": self.retain,
            "tls_info": {
                "self_signed": self.tls_allow_self_signed,
                "insecure": self.tls_insecure,
                "ca": self.tls_ca,
                "server_cert": self.ssl_cert,
                "client_cert": self.mtls_cert,
                "client_key": self.mtls_key,
                "cert_reqs": repr(self.ssl_cert),
            },
        }

    def connect(self: mqtt_client, keep_alive: Optional[int] = None):
        if not keep_alive:
            keep_alive = 60
        else:
            keep_alive = int(keep_alive)
        # g.logger.debug(f"MQTT OPTIONS {self.get_options()=}")
        try:
            from uuid import uuid4
            uuid = uuid4()
            if self.tls_ca:
                if self.mtls_key and self.mtls_cert:
                    self.client_id = f"{self.client_id}mTLS-{uuid}"
                    self.client = mqtt_client.Client(self.client_id, clean_session=True)
                    self.client.tls_set(
                        ca_certs=self.tls_ca,
                        certfile=self.mtls_cert,
                        keyfile=self.mtls_key,
                        cert_reqs=self.ssl_cert,
                        # tls_version=ssl.PROTOCOL_TLSv1_2
                    )
                    if self.tls_insecure:
                        self.client.tls_insecure_set(True)  # DONT verify CN (COMMON NAME) in certificates
                    logger.debug(
                        f"mqtt:connect: '{self.client_id}' ->  '{self.broker}:{self.port}' trying mTLS "
                        f"({'TLS Secure' if not self.tls_insecure else 'TLS Insecure'}) -> tls_ca: "
                        f"'{self.tls_ca}' tls_client_key: '{self.mtls_key}' tls_client_cert: '{self.mtls_cert}'"
                    )

                elif (self.mtls_cert and not self.mtls_key) or (not self.mtls_cert and self.mtls_key):
                    logger.debug(
                        f"mqtt:connect:ERROR using mTLS so trying  {self.client_id} -> TLS "
                        f"({'TLS Secure' if not self.tls_insecure else 'TLS Insecure'}) -> tls_ca: "
                        f"{self.tls_ca} tls_client_key: {self.mtls_key} tls_client_cert: {self.mtls_cert}"
                    )
                    self.client_id = f"{self.client_id}TLS-{uuid}"
                    self.client = mqtt_client.Client(self.client_id)
                    self.client.tls_set(self.tls_ca, cert_reqs=self.ssl_cert)
                    # ssl.CERT_NONE allows self signed, don't use if using lets encrypt certs and CA
                    if self.tls_insecure:
                        self.client.tls_insecure_set(
                            True
                        )  # DO NOT verify CN (COMMON NAME) in certificates - [MITM risk]

                else:
                    self.client_id = f"{self.client_id}TLS-{uuid}"
                    self.client = mqtt_client.Client(self.client_id)
                    self.client.tls_set(self.tls_ca, cert_reqs=ssl.CERT_NONE)
                    logger.debug(
                        f"mqtt:connect: {self.client_id} -> {self.broker}:{self.port} trying TLS "
                        f"({'TLS Secure' if not self.tls_insecure else 'TLS Insecure'}) -> tls_ca: {self.tls_ca}"
                    )
            else:
                self.client_id = f"{self.client_id}noTLS-{uuid}"
                self.client = mqtt_client.Client(self.client_id)
                logger.debug(
                    f"mqtt:connect: {self.client_id} -> "
                    f"{self.broker if not self.sanitize else self.sanitize_str}:{self.port} "
                    f"{'user:{}'.format(self.user) if self.user else ''} "
                    f"{'passwd:{}'.format(self.sanitize_str) if self.password and self.user else 'passwd:<None>'}"
                )

            if self.user and self.password:
                self.client.username_pw_set(self.user, password=self.password)  # set username and password
            self.client.connect_async(self.broker, port=self.port, keepalive=keep_alive)  # connect to broker
            self.client.loop_start()  # start the loop
            self.client.on_connect = on_connect  # attach function to callback
            # self.client.on_log = on_log
            # connack_string(connack_code)
            # self.client.on_publish = on_publish
            # self.client.on_message=on_message
        except Exception as e:
            logger.error(f"mqtt:connect:err_msg-> {e}")

        if not self.client:
            logger.error(
                f"mqtt:connect: STRANGE ERROR -> there is no active mqtt object instantiated?! Exiting mqtt routine"
            )
            return
        self.conn_wait = 5.0
        logger.debug(f"mqtt:connect: connecting to broker (timeout: {self.conn_wait})")
        start = time()
        while not Connected:  # Wait for connection
            elapsed = time() - start  # how long has it been
            if elapsed > self.conn_wait:
                logger.error(
                    f"mqtt:connect: broker @ '{self.broker}' did not reply within '{self.conn_wait}' seconds"
                )
                break  # no longer than x seconds waiting for it to connect
        if not Connected:
            logger.error(f"mqtt:connect: could not establish a connection to the broker!")
        else:
            self.conn_time = time()
            self.connected = Connected

    def publish(self, topic=None, message=None, qos=0, retain: bool = False):
        global wasConnected
        if not Connected:
            if wasConnected:
                logger.error(f"mqtt:publish: no active connection, attempting to re connect...")
                self.client.reconnect()
                wasConnected = False
            else:
                logger.error(f"mqtt:publish: no active connection!")
                return
        if retain:
            self.retain = retain
        self.connected = Connected
        if not message and self.image is not None:
            message = self.image
        if not message:
            logger.debug(f"mqtt:publish: no message specified, sending empty message!!")
        if not topic:
            logger.error(f"mqtt:publish: no topic specified, please set a topic, skipping publish...")
            return
        if isinstance(message, bytes):
            logger.debug(
                f"mqtt:publish: sending -> topic: '{topic}'  data: '<serialized byte object>'  size: "
                f"{message.__sizeof__() / 1024 / 1024:.2f} MB",
            )
        elif not isinstance(message, bytearray):
            logger.debug(f"mqtt:publish: sending -> topic: '{topic}' data: {message[:255]}")
        else:
            logger.debug(
                f"mqtt:publish: sending -> topic: '{topic}'  data: '<serialized bytearray>'  size: "
                f"{message.__sizeof__() / 1024 / 1024:.2f} MB",
            )
        try:
            self.client.publish(topic, message, qos=qos, retain=self.retain)
        except Exception as e:
            return logger.error(f"mqtt:publish:err_msg-> {e}")

    def close(self):
        global Connected, wasConnected
        if not Connected:
            return
        try:
            if self.conn_time:
                self.conn_time = time() - self.conn_time
            logger.debug(
                f"mqtt:close: {self.client_id} ->  disconnecting from mqtt broker: "
                f"'{self.broker if not self.sanitize else self.sanitize_str}:{self.port}'"
                f" [connection alive for: {self.conn_time} seconds]",
            )
            self.client.disconnect()
            self.client.loop_stop()
            Connected = self.connected = wasConnected = False
        except Exception as e:
            return logger.error(f"mqtt:close:err_msg-> {e}")
