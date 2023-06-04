# Virel.ai detector - HTTP API for object detection
# Supply it a base64 encoded image
import base64
import json
import time
from logging import getLogger

import cv2
import requests
import numpy as np

from ....Shared.Models.Enums import ModelProcessor, ModelType
from ...Models.config import VirelAIModelConfig
from ...Log import SERVER_LOGGER_NAME

LP: str = "Virel.ai:"
logger = getLogger(SERVER_LOGGER_NAME)


class VirelAI:
    def __init__(self, cfg: VirelAIModelConfig):
        self.name: str = cfg.name
        self.config: VirelAIModelConfig = cfg
        self.processor = ModelProcessor.NONE

    def detect(self, input_image: np.ndarray):
        b_boxes, labels, confs = [], [], []
        h, w = input_image.shape[:2]
        t = time.perf_counter()
        logger.debug(f"{LP}detect: '{self.name}' - input image {w}*{h}")
        _conf = self.config.detection_options.confidence
        # Resize to send to virel.ai , calculate scale factor
        is_succ, r_img = cv2.imencode(".jpg", input_image)
        if is_succ:
            scale_w = w / 640
            scale_h = h / 480

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
                    if "LabelModelVersion" in response:
                        self.name = f"{self.name} : {response['LabelModelVersion']}"
                    logger.debug(
                        f"perf:{LP} '{self.name}' took {time.perf_counter() - t}"
                    )
                    # annotated image
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
                        # box = item["BoundingBox"]

                        # bbox = (
                        #     round(w * box["Left"]),
                        #     round(h * box["Top"]),
                        #     round(w * (box["Left"] + box["Width"])),
                        #     round(h * (box["Top"] + box["Height"])),
                        # )
                        # return false data for now
                        bbox = (0, 0, 0, 0)
                        b_boxes.append(bbox)
                        labels.append(label)
                        confs.append(conf)
        else:
            logger.error(f"{LP}Error encoding image")

        return {
            "success": True if labels else False,
            "type": ModelType.OBJECT,
            "processor": self.processor,
            "model_name": self.name,
            "label": labels,
            "confidence": confs,
            "bounding_box": b_boxes,
        }
