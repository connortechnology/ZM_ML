# Virel.ai detector - HTTP API for object detection
# Supply it a base64 encoded image
import base64
import time
from logging import getLogger

import requests
import numpy as np

from ....Shared.Models.Enums import ModelProcessor, ModelType
from ...Log import SERVER_LOGGER_NAME

LP: str = "Virel.ai:"
logger = getLogger(SERVER_LOGGER_NAME)


class VirelAI:
    def __init__(self):
        self.name: str = "Virel.ai"
        self.processor = ModelProcessor.NONE

    def detect(self, input_image: np.ndarray):
        b_boxes, labels, confs = [], [], []
        h, w = input_image.shape[:2]
        t = time.perf_counter()
        logger.debug(f"{LP}detect: '{self.name}' - input image {w}*{h}")
        _conf = 0.5
        files = {"Image": {"Bytes": base64.b64encode(input_image.tobytes()).decode()}}

        api_url = "https://virel.ai"
        object_url = f"{api_url}/api/detect/payload"
        headers = {"Content-type": "application/json; charset=utf-8"}
        logger.debug(
            f"{LP} sending detection request to {object_url} with headers {headers}"
        )
        try:
            r = requests.post(url=object_url, headers=headers, json=files)
            logger.debug(f"R: {r.text}")
            r.raise_for_status()
        except Exception as e:
            logger.error(f"Error invoking virelai api: {e}")
            raise
        else:
            response = r.json()
            logger.debug(
                f"perf:{LP} took {time.perf_counter() - t} -> detection response -> {response}"
            )

            # Parse the returned labels
            for item in response["Labels"]:
                if "Instances" not in item:
                    continue
                for instance in item["Instances"]:
                    if "BoundingBox" not in instance or "Confidence" not in instance:
                        continue
                    conf = instance["Confidence"] / 100
                    if conf < _conf:
                        continue
                    label = item["Name"].casefold()
                    box = instance["BoundingBox"]
                    bbox = (
                        round(w * box["Left"]),
                        round(h * box["Top"]),
                        round(w * (box["Left"] + box["Width"])),
                        round(h * (box["Top"] + box["Height"])),
                    )
                    b_boxes.append(bbox)
                    labels.append(label)
                    confs.append(conf)

        return {
            "success": True if labels else False,
            "type": ModelType.OBJECT,
            "processor": self.processor,
            "model_name": self.name,
            "label": labels,
            "confidence": confs,
            "bounding_box": b_boxes,
        }
