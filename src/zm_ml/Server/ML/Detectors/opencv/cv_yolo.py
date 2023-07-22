from __future__ import annotations
import time
from logging import getLogger
from typing import Optional, TYPE_CHECKING
from warnings import warn
try:
    import cv2
except ImportError:
    warn("OpenCV not installed, cannot use OpenCV detectors")
    raise
import numpy as np

from .....Shared.Models.Enums import ModelProcessor
from .cv_base import CV2Base
from .....Shared.Models.config import DetectionResults, Result
from ....Log import SERVER_LOGGER_NAME

if TYPE_CHECKING:
    from ....Models.config import CV2YOLOModelConfig

LP: str = "OpenCV:YOLO:"
logger = getLogger(SERVER_LOGGER_NAME)
# TODO: Choose what models to load and keep in memory and which to load and drop for memory constraints


class CV2YOLODetector(CV2Base):
    def __init__(self, model_config: CV2YOLOModelConfig):
        super().__init__(model_config)
        # self.is_locked: bool = False
        # Model init params not initialized in super()
        self.model: Optional[cv2.dnn.DetectionModel] = None
        # logger.debug(f"{LP} configuration: {self.config}")
        self.load_model()

    def get_classes(self):
        return self.config.labels

    def load_model(self):
        logger.debug(
            f"{LP} loading model into processor memory: {self.name} ({self.id})"
        )
        load_timer = time.perf_counter()
        try:
            # Allow for .weights/.cfg and .onnx YOLO architectures
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

    def detect(self, input_image: np.ndarray):
        if input_image is None:
            raise ValueError(f"{LP} no image passed!")
        if not self.net:
            self.load_model()
        _h, _w = self.config.height, self.config.width
        if self.config.square:
            input_image = self.square_image(input_image)
        h, w = input_image.shape[:2]
        # dnn.DetectionModel resizes the image and calculates scale of bounding boxes for us
        labels, confs, b_boxes = [], [], []
        nms_threshold, conf_threshold = self.options.nms, self.options.confidence

        logger.debug(
            f"{LP}detect: '{self.name}' ({self.processor}) - "
            f"input image {w}*{h} - model input {_w}*{_h}"
            f"{' [squared]' if self.config.square else ''}"
        )
        self.acquire_lock()
        try:
            detection_timer = time.perf_counter()

            l, c, b = self.model.detect(
                input_image, conf_threshold, nms_threshold
            )

            logger.debug(
                f"perf:{LP}{self.processor}: '{self.name}' detection "
                f"took: {time.perf_counter() - detection_timer:.5f} s"
            )
            for (class_id, confidence, box) in zip(l, c, b):
                confidence = float(confidence)
                x, y, _w, _h = (
                    int(round(box[0])),
                    int(round(box[1])),
                    int(round(box[2])),
                    int(round(box[3])),
                )
                b_boxes.append(
                    [
                        x,
                        y,
                        x + _w,
                        y + _h,
                    ]
                )
                labels.append(self.config.labels[class_id])
                confs.append(confidence)
        except Exception as all_ex:
            err_msg = repr(all_ex)
            # cv2.error: OpenCV(4.2.0) /home/<Someone>/opencv/modules/dnn/src/cuda/execution.hpp:52: error: (-217:Gpu
            # API call) invalid device function in function 'make_policy'
            logger.error(f"{LP} exception during detection -> {all_ex}")
            # OpenCV 4.7.0 Weird Error fixed with rolling fix
            # OpenCV:YOLO: exception during detection -> OpenCV(4.7.0-dev) /opt/opencv/modules/dnn/src/layers/cpu_kernels/conv_winograd_f63.cpp:401: error: (-215:Assertion failed) CONV_WINO_IBLOCK == 3 && CONV_WINO_KBLOCK == 4 && CONV_WINO_ATOM_F32 == 4 in function 'winofunc_BtXB_8x8_f32'
            if err_msg.find("-215:Assertion failed") > 0:
                if err_msg.find("CONV_WINO_IBLOCK == 3 && CONV_WINO_KBLOCK == 4") > 0:
                    _msg = f"{LP} OpenCV 4.7.x WEIRD bug detected! " \
                           f"Please update to OpenCV 4.7.1+ or 4.6.0 or less!"
                    logger.error(_msg)
                    raise RuntimeError(_msg)
            elif err_msg.find("-217:Gpu") > 0:
                if (
                    err_msg.find("'make_policy'") > 0
                    and self.processor == ModelProcessor.GPU
                ):
                    _msg = f"{LP} (-217:Gpu # API call) invalid device function in function 'make_policy' - " \
                        f"This happens when OpenCV is compiled with the incorrect Compute Capability " \
                        f"(CUDA_ARCH_BIN). There is a high probability that you need to recompile OpenCV with " \
                        f"the correct CUDA_ARCH_BIN before GPU detections will work properly!"
                    logger.error(_msg)
                    raise RuntimeError(_msg)
            raise all_ex
        finally:
            self.release_lock()

        result = DetectionResults(
            success=True if labels else False,
            type=self.config.model_type,
            processor=self.processor,
            model_name=self.name,
            results=[Result(label=labels[i], confidence=confs[i], bounding_box=b_boxes[i]) for i in range(len(labels))],
        )
        return result

        return {
            "success": True if labels else False,
            "type": self.config.model_type,
            "processor": self.processor,
            "model_name": self.name,
            "label": labels,
            "confidence": confs,
            "bounding_box": b_boxes,
        }
