"""Class for tensorflow detector"""

import time
from logging import getLogger
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

LP: str = "TensorFlow:"
from zm_ml.Server.Log import SERVER_LOGGER_NAME
logger = getLogger(SERVER_LOGGER_NAME)


class TFDetector:

    def __init__(self):
        pass


