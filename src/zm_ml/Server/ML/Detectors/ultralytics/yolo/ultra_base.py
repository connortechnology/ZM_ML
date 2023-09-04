"""YOLO v5, v8 and NAS support"""
from __future__ import annotations

import time
import warnings
from functools import lru_cache
from logging import getLogger
from typing import Optional, List, TYPE_CHECKING, Union


try:
    import cv2
except ImportError:
    warnings.warn("OpenCV not installed!", warnings.ImportWarning)
    raise
import numpy as np

from ....file_locks import FileLock
from ......Shared.Models.Enums import ModelProcessor
from ......Server.Models.config import UltralyticsModelOptions
from ......Shared.Models.config import DetectionResults, Result
from .....Log import SERVER_LOGGER_NAME
from .....app import get_global_config

if TYPE_CHECKING:
    from ......Shared.Models.Enums import UltralyticsSubFrameWork
    from ..Models.config import UltralyticsModelConfig
    from ......Shared.configs import GlobalConfig

logger = getLogger(SERVER_LOGGER_NAME)
LP: str = "ultra:yolo:"
g: Optional[GlobalConfig] = None


class UltralyticsYOLODetector(FileLock):
    name: str
    config: UltralyticsModelConfig
    model_name: str
    yolo_model_type: UltralyticsSubFrameWork
    processor: ModelProcessor
    _classes: Optional[List] = None
    options: UltralyticsModelOptions
    ok: bool = False

    def __init__(self, config: Optional[UltralyticsModelConfig] = None):
        try:
            import torch
        except ImportError as e:
            return
        try:
            import ultralytics
        except ImportError:
            return
        assert config, "No config provided!"
        global g
        self.ultralytics = ultralytics
        self.torch = torch

        g = get_global_config()
        self.cwd: Optional[str] = None
        self.config = config
        self.options = self.config.detection_options
        self.id = self.config.id
        self.name = self.config.name
        self.model_name = self.config.pretrained.name
        self.yolo_model_type = self.config.sub_framework
        self.processor = self.config.processor
        self.model: Union[ultralytics.YOLO, ultralytics.NAS, None] = None
        self.device = self._get_device()
        self.cache_dir = g.config.system.models_dir / "ultralytics/cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.load_model()

    def load_model(self):
        logger.debug(
            f"{LP} loading model into processor memory: {self.name} ({self.id})"
        )
        if self.config.pretrained and self.config.pretrained.enabled is True:
            logger.debug(
                f"{LP} 'pretrained' model requested, using pretrained weights..."
            )
            _pt = self.config.pretrained.name
            sub_fw = self.config.sub_framework
            if _pt:
                if _pt.startswith("yolo_nas_"):
                    logger.debug(f"{LP} Attempting to load YOLO-NAS model")
                    self.model = self.ultralytics.NAS(_pt)
                elif _pt.startswith("yolov"):
                    pass
                self.model = self.ultralytics.YOLO(_pt)

            self.ok = True
        elif self.config.input and self.config.input.exists():
            logger.warning(
                f"{LP} Pretrained is not enabled, attempting user supplied model..."
            )

            try:
                _pt = self.config.input
                logger.info(f"{LP} loading FILE -> {_pt}")
                if _pt.name.startswith("yolo_nas_"):
                    logger.warning(f"{LP} Attempting to load YOLO-NAS model")
                    self.model = self.ultralytics.NAS(_pt.as_posix())
                elif _pt.name.startswith("yolov"):
                    logger.warning(f"{LP} Attempting to load YOLO model")

                    self.model = self.ultralytics.YOLO(_pt.as_posix())

                self.ok = True
            except Exception as model_load_exc:
                logger.error(
                    f"{LP} Error while loading model file and/or config! "
                    f"(May need to re-download the model/cfg file) => {model_load_exc}"
                )
                self.ok = False
                raise model_load_exc
            else:
                if isinstance(self.model, self.ultralytics.YOLO):
                    self.model.to(self.device)
                logger.debug(f"{LP} model loaded successfully -> {self.model = }")

    @lru_cache(maxsize=1)
    def _get_device(self) -> Optional[str]:
        logger.debug(f"{LP} getting device...")
        dev = "cpu"
        if self.processor == ModelProcessor.GPU:
            logger.debug(f"{LP} GPU requested...")
            # todo: allow device index config (possibly use pycuda to get dev names / index for user convenience)
            if self.torch.cuda.is_available():
                logger.debug(f"{LP} GPU available, ascertaining index...")
                _idx = 0
                if self.config.gpu_idx is not None:
                    _idx = self.config.gpu_idx
                    if _idx >= self.torch.cuda.device_count():
                        logger.warning(
                            f"{LP} GPU index out of range, using default index 0"
                        )
                        _idx = 0
                dev = f"cuda:{_idx}"
            else:
                logger.warning(f"{LP} GPU not available, using CPU...")
        if dev.startswith("cpu"):
            self.processor = ModelProcessor.CPU
        elif dev.startswith("cuda"):
            self.processor = ModelProcessor.GPU
        logger.debug(f"{LP} using device: {dev} :: {self.processor}")
        return dev

    def detect(self, image: np.ndarray):
        logger.debug(f"{LP} detecting objects in image...")
        detection_timer = time.perf_counter()
        labels = []
        confs = []
        b_boxes = []
        try:
            self.acquire_lock()
            ultralytics = self.ultralytics
            results: ultralytics.engine.results.Boxes = (
                self.model.predict(
                    image, iou=self.options.nms, conf=self.options.confidence
                )
            )
        except Exception as detect_exc:
            logger.error(
                f"{LP} Error model: '{self.name}' - while detecting objects! => {detect_exc}"
            )
            raise detect_exc
        else:
            # only 1 image so only 1 result (batch size = 1)
            for _result in results:
                boxes = _result.boxes  # Boxes object for bbox outputs
                # masks = result.masks  # Masks object for segmentation masks outputs
                # keypoints = result.keypoints  # Keypoints object for pose outputs
                # probs = result.probs  # Class probabilities for classification outputs

                b_boxes.extend(boxes.xyxy.round().int().tolist())
                confs.extend(boxes.conf.float().tolist())
                labels.extend([_result.names[i] for i in boxes.cls.int().tolist()])
        finally:
            self.release_lock()
        logger.debug(
            f"perf:{LP}{self.processor}: '{self.name}' detection "
            f"took: {time.perf_counter() - detection_timer:.5f} s"
        )
        result = DetectionResults(
            success=True if labels else False,
            type=self.config._model_type,
            processor=self.processor,
            name=self.name,
            results=[
                Result(label=labels[i], confidence=confs[i], bounding_box=b_boxes[i])
                for i in range(len(labels))
            ],
            # image=image,
        )
        return result


"""
Result from results object:
                
                boxes: tensor([[1.3595e+03, 3.3042e+02, 1.9156e+03, 8.9059e+02, 9.6157e-01, 2.0000e+00],
                        [6.7180e+02, 9.9587e+01, 1.2596e+03, 3.3399e+02, 9.2557e-01, 2.0000e+00],
                        [1.3567e+02, 3.3729e+02, 1.0298e+03, 1.0668e+03, 7.9451e-01, 7.0000e+00],
                        [1.3696e+02, 3.3902e+02, 1.0291e+03, 1.0668e+03, 7.8310e-01, 2.0000e+00]])
                cls: tensor([2., 2., 7., 2.])
                conf: tensor([0.9616, 0.9256, 0.7945, 0.7831])
                data: tensor([[1.3595e+03, 3.3042e+02, 1.9156e+03, 8.9059e+02, 9.6157e-01, 2.0000e+00],
                        [6.7180e+02, 9.9587e+01, 1.2596e+03, 3.3399e+02, 9.2557e-01, 2.0000e+00],
                        [1.3567e+02, 3.3729e+02, 1.0298e+03, 1.0668e+03, 7.9451e-01, 7.0000e+00],
                        [1.3696e+02, 3.3902e+02, 1.0291e+03, 1.0668e+03, 7.8310e-01, 2.0000e+00]])
                id: None
                is_track: False
                orig_shape: (1080, 1920)
                shape: torch.Size([4, 6])
                xywh: tensor([[1637.5486,  610.5079,  556.1774,  560.1673],
                        [ 965.6874,  216.7904,  587.7697,  234.4075],
                        [ 582.7472,  702.0620,  894.1467,  729.5374],
                        [ 583.0057,  702.8987,  892.0985,  727.7506]])
                xywhn: tensor([[0.8529, 0.5653, 0.2897, 0.5187],
                        [0.5030, 0.2007, 0.3061, 0.2170],
                        [0.3035, 0.6501, 0.4657, 0.6755],
                        [0.3036, 0.6508, 0.4646, 0.6738]])
                xyxy: tensor([[1359.4600,  330.4243, 1915.6373,  890.5916],
                        [ 671.8026,   99.5867, 1259.5723,  333.9942],
                        [ 135.6738,  337.2934, 1029.8206, 1066.8307],
                        [ 136.9564,  339.0234, 1029.0549, 1066.7740]])
                xyxyn: tensor([[0.7081, 0.3059, 0.9977, 0.8246],
                        [0.3499, 0.0922, 0.6560, 0.3093],
                        [0.0707, 0.3123, 0.5364, 0.9878],
                        [0.0713, 0.3139, 0.5360, 0.9878]])
"""
