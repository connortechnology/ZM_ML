# Virel.ai detector - HTTP API for object detection
# Supply it a base64 encoded image
import base64
import json
import logging
import time
from logging import getLogger
from typing import Optional

import cv2
import requests
import numpy as np

from ....Shared.Models.Enums import ModelProcessor, ModelType
from ...Models.config import VirelAIModelConfig
from ...Log import SERVER_LOGGER_NAME
from ....Client.Log import CLIENT_LOGGER_NAME
LP: str = "Virel.ai:"

logger = getLogger(SERVER_LOGGER_NAME)
# logger = getLogger(CLIENT_LOGGER_NAME)


class VirelAI:

    def logger(self, _logger: logging.Logger):
        global logger
        logger = _logger

    def __init__(self, cfg: Optional[VirelAIModelConfig] = None):
        if cfg:
            self.name: str = cfg.name
            self.config: VirelAIModelConfig = cfg
            self.processor = ModelProcessor.NONE

    def get_image(self, image: np.ndarray):
        """Supply an image, encode it top jpeg, send to virel ai and receive an annotated image back"""
        is_success, _buff = cv2.imencode(".jpg", image)
        logger.info(f"{LP} grabbing image from virelAI API")
        if not is_success:
            logger.warning(f"{LP} Was unable to convert the image to JPG")
            return image
        else:
            files = {"Image": {"Bytes": base64.b64encode(_buff.tobytes()).decode()}}

            api_url = "http://virel.ai"
            try:
                headers = {"Content-type": "application/json; charset=utf-8"}
                display_url = f"{api_url}/api/display/payload"
                r = requests.post(url=display_url, headers=headers, json=files)
                r.raise_for_status()
            except Exception as e:
                logger.error(f"{LP} Error during image grab: {e}")
            else:
                image = np.asarray(bytearray(r.content), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                logger.debug(f"{LP} received annotated image")
                if image == _buff:
                    logger.warning(f"{LP} received same image as sent")
                else:
                    logger.debug(f"{LP} received different image as sent")


        return image

    def detect(self, input_image: np.ndarray):
        b_boxes, labels, confs = [], [], []
        grab_image: bool = False
        h, w = input_image.shape[:2]
        t = time.perf_counter()
        logger.debug(f"{LP}detect: '{self.name}' - input image {w}*{h}")
        _conf = self.config.detection_options.confidence
        is_succ, r_img = cv2.imencode(".jpg", input_image)
        if is_succ:
            files = {"Image": {"Bytes": base64.b64encode(r_img.tobytes()).decode()}}
            api_url = "https://virel.ai"
            object_url = f"{api_url}/api/detect/payload"
            headers = {"Content-type": "application/json; charset=utf-8"}
            logger.debug(
                f"{LP} sending detection request to {object_url} with headers {headers}"
            )
            try:
                r = requests.post(url=object_url, headers=headers, json=files)
                logger.debug(f"{LP} response text -> {r.text}")
                r.raise_for_status()
            except Exception as e:
                logger.error(f"Error invoking virelai api: {e}")
                raise
            else:
                try:
                    response = r.json()
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding virelai api response: {e}")
                else:
                    # if "LabelModelVersion" in response:
                    #     self.name = f"{self.name} : {response['LabelModelVersion']}"
                    logger.debug(
                        f"perf:{LP} '{self.name}' took {time.perf_counter() - t}"
                    )
                    # Parse the returned labels
                    for item in response["Labels"]:
                        # {
                        # "LabelModelVersion": "detect-1",
                        # "Img": "",
                        # "Labels": [
                        #   {"Confidence": "78.06", "Name": "person"},
                        #   {"Confidence": "75.35", "Name": "person"}
                        #   ]
                        # }
                        conf = float(item["Confidence"]) / 100
                        if conf < _conf:
                            continue
                        label = item["Name"].casefold()
                        # Virel.ai does not return bounding box coords. More of an image classifier.
                        box = item.get("BoundingBox")
                        if box:
                            grab_image = False
                            bbox = (
                                round(w * box["Left"]),
                                round(h * box["Top"]),
                                round(w * (box["Left"] + box["Width"])),
                                round(h * (box["Top"] + box["Height"])),
                            )
                        else:
                            grab_image = True

                        # return false data for now
                        bbox = (0, 0, 0, 0)
                        b_boxes.append(bbox)
                        labels.append(label)
                        confs.append(conf)
        else:
            logger.error(f"{LP}Error encoding image")

        if grab_image:
            logger.debug(
                f"virel.ai does not supply bounding boxes instead, it supplies an annotated image. Client must grab it."
            )
            b_boxes = "virel"

        return {
            "success": True if labels else False,
            "type": ModelType.OBJECT,
            "processor": self.processor,
            "model_name": self.name,
            "label": labels,
            "confidence": confs,
            # set to None to signal the image needs grabbing from virel.ai itself
            "bounding_box": b_boxes,
        }
