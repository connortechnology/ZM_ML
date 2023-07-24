"""YOLO v5, v8 and NAS support"""
import time
from typing import Optional, List
import warnings
from logging import getLogger

try:
    import ultralytics
except ImportError:
    ultralytics = None
try:
    import cv2
except ImportError:
    warnings.warn("OpenCV not installed!", warnings.ImportWarning)
    raise
import numpy as np

from ....file_locks import FileLock
from ......Shared.Models.Enums import ModelType, ModelProcessor, UltralyticsSubFrameWork
from .....app import SERVER_LOGGER_NAME
from ......Shared.Models.config import DetectionResults, Result
from ..Models.config import UltralyticsModelConfig
from .....Log import SERVER_LOGGER_NAME

logger = getLogger(SERVER_LOGGER_NAME)
LP: str = "ultra:yolo:"

class UltralyticsYOLODetector(FileLock):
    name: str
    config: UltralyticsModelConfig
    model_name: str
    model: ultralytics.YOLO
    yolo_model_type: UltralyticsSubFrameWork
    processor: ModelProcessor
    _classes: Optional[List] = None


    def __init__(self, config: Optional[UltralyticsModelConfig] = None):
        if ultralytics is None:
            raise ImportError("Ultralytics not installed!")
        if config is None:
            raise ValueError("No config provided!")
        self.config = config
        self.id = self.config.id
        self.name = self.config.name
        self.model_name = self.config.model_name
        self.yolo_model_type = self.config.sub_framework

    def load_model(self):
        logger.debug(
            f"{LP} loading model into processor memory: {self.name} ({self.id})"
        )
        load_timer = time.perf_counter()
        try:
            model_file: str = self.config.input.as_posix()
            config_file: Optional[str] = None
            if self.config.config:
                if self.config.config.exists():
                    config_file = self.config.config.as_posix()
                else:
                    raise FileNotFoundError(
                        f"{LP} config file '{self.config.config}' not found!"
                    )
            logger.info(f"{LP} loading -> model: {model_file} :: config: {config_file}")
            self.net = cv2.dnn.readNet(model_file, config_file)
        except Exception as model_load_exc:
            logger.error(
                f"{LP} Error while loading model file and/or config! "
                f"(May need to re-download the model/cfg file) => {model_load_exc}"
            )
            raise model_load_exc
        # DetectionModel allows to set params for preprocessing input image. DetectionModel creates net
        # from file with trained weights and config, sets preprocessing input, runs forward pass and return
        # result detections. For DetectionModel SSD, Faster R-CNN, YOLO topologies are supported.
        if self.net is not None:
            self.model = cv2.dnn.DetectionModel(self.net)
            self.model.setInputParams(
                scale=1 / 255, size=(self.config.width, self.config.height), swapRB=True
            )
            self.cv2_processor_check()
            logger.debug(
                f"{LP} set CUDA/cuDNN backend and target"
            ) if self.processor == ModelProcessor.GPU else None

            logger.debug(
                f"perf:{LP} '{self.name}' loading completed in {time.perf_counter() - load_timer:.5f} s"
            )
        else:
            logger.debug(
                f"perf:{LP} '{self.name}' FAILED in {time.perf_counter() - load_timer:.5f} s"
            )

