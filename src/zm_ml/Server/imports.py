from logging import getLogger

logger = getLogger("ML-API")

try:
    import cv2
except ImportError as e:
    logger.error("OpenCV is not installed, please install it")
    raise e

try:
    import numpy as np
except ImportError as e:
    logger.error("Numpy is not installed, please install it")
    raise e
