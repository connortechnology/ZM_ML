from __future__ import annotations

import time
from logging import getLogger
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import torchvision
except ImportError:
    RetinaNet_ResNet50_FPN_V2_Weights, FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN_MobileNet_V3_Large_FPN_Weights, FCOS_ResNet50_FPN_Weights = None, None, None, None
    torchvision = None
else:
    from torchvision.models.detection.ssd import SSD
    from torchvision.models.detection import (
        RetinaNet,
        FasterRCNN,
        FCOS,
        MaskRCNN,
        KeypointRCNN,
        retinanet_resnet50_fpn_v2,
        RetinaNet_ResNet50_FPN_V2_Weights,

        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_fpn,

        FCOS_ResNet50_FPN_Weights,
        fcos_resnet50_fpn,
    )

from .....Shared.Models.Enums import ModelProcessor
from .....Shared.Models.config import DetectionResults, Result
from ....Models.config import TorchModelConfig
from ....Log import SERVER_LOGGER_NAME
from ...file_locks import FileLock

if TYPE_CHECKING:
    from ....Models.config import GlobalConfig


logger = getLogger(SERVER_LOGGER_NAME)
LP: str = "Torch:"
WEIGHTS_TYPES = Union[
    RetinaNet_ResNet50_FPN_V2_Weights, FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN_MobileNet_V3_Large_FPN_Weights, FCOS_ResNet50_FPN_Weights
]
g: GlobalConfig


class TorchDetector(FileLock):
    name: str
    weights: WEIGHTS_TYPES
    device: torch.device

    def __init__(self, model_config: TorchModelConfig):
        if any([torch is None, torchvision is None]):
            raise ImportError(f"{LP} Torch or Torchvision not installed, cannot use Torch detectors")
        global g
        from ....app import get_global_config

        g = get_global_config()

        self.model: Optional[Union[RetinaNet, FasterRCNN, FCOS, SSD]] = None

        if not model_config:
            raise ValueError(f"{LP} no config passed!")
        self.config = model_config
        self.options = model_config.detection_options
        self.processor: ModelProcessor = self.config.processor
        self.name = self.config.name
        self.id = self.config.id
        self.device = self._get_device()
        # call this after getting the processor/device
        super().__init__()
        self.ok = False
        self.cache_dir = g.config.system.models_dir / "torch/cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.load_model()

    def load_model(self):
        if torch is None:
            logger.error(f"{LP} Torch not installed, cannot use Torch detectors")
        else:
            import os

            conf, nms = self.options.confidence, self.options.nms
            if self.config.pretrained and self.config.pretrained.enabled is True:
                logger.debug(f"{LP} 'pretrained' model requested, using pretrained weights...")
                _pth = os.environ.get("TORCH_HOME", None)
                if _pth:
                    logger.warning(f"{LP} 'TORCH_HOME' is already set, working around it...")
                os.environ["TORCH_HOME"] = self.cache_dir.as_posix()
                _pt = self.config.pretrained.name
                if _pt:
                    if _pt in ("default", "balanced"):
                        self.name = f"torch: RetinaNet RN50 v2"
                        self.weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
                        self.model = retinanet_resnet50_fpn_v2(
                            weights=self.weights, box_score_thresh=conf, box_nms_thresh=nms
                        ).to(self.device)
                    elif _pt == "fast":
                        self.name: str = f"torch: FCOS RN50 v2"
                        self.weights = FCOS_ResNet50_FPN_Weights.DEFAULT
                        self.model = fcos_resnet50_fpn(
                            weights=self.weights, box_score_thresh=conf, box_nms_thresh=nms
                        ).to(self.device)
                    elif _pt == "low_performance":
                        pass
                    # SSDlite ?
                    elif _pt == "accurate":
                        self.name = f"torch: fRCNN MN v3"
                        self.weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
                        self.model = fasterrcnn_mobilenet_v3_large_fpn(
                            weights=self.weights, box_score_thresh=conf, box_nms_thresh=nms
                        ).to(self.device)
                    elif _pt == "high_performance":


                        self.name = f"torch: fRCNN RN50 v2"
                        self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                        self.model = fasterrcnn_resnet50_fpn_v2(
                            weights=self.weights, box_score_thresh=conf, box_nms_thresh=nms
                        ).to(self.device)
                logger.debug(
                    f"{LP} loading model into processor memory: {self.name} ({self.id})"
                )
                self.model.eval()
                self.ok = True
                if _pth:
                    logger.warning(f"{LP} resetting 'TORCH_HOME' to original value...")
                    os.environ["TORCH_HOME"] = _pth
            else:
                logger.warning(f"{LP} pretrained was not defined, user trained models are: Not Implemented yet....")



    def _convert_image(self, image: np.ndarray) -> torch.Tensor:
        """Convert Image from numpy array to torch tensor and convert to channels first

        :param (np.ndarray) image: numpy array
        """
        ret = torch.from_numpy(image)
        # convert to channels first
        ret = ret.permute(2, 0, 1)
        # return the tensor
        return ret

    def _get_device(self) -> Optional[torch.device]:
        if torch is None:
            logger.error(f"{LP} Torch not installed, cannot use Torch detectors")
        else:
            dev = "cpu"
            if self.processor == ModelProcessor.GPU:
                # todo: allow device index config (possibly use pycuda to get dev names / index for user convenience)
                if torch.cuda.is_available():
                    _idx = 0
                    if self.config.gpu_idx is not None:
                        _idx = self.config.gpu_idx
                        if _idx >= torch.cuda.device_count():
                            logger.warning(
                                f"{LP} GPU index out of range, using default index 0"
                            )
                            _idx = 0
                    dev = f"cuda:{_idx}"
            if dev.startswith("cpu"):
                self.processor = ModelProcessor.CPU
            elif dev.startswith("cuda"):
                self.processor = ModelProcessor.GPU
            logger.debug(f"{LP} using device: {dev}")
            return torch.device(dev)

    def detect(self, image: np.ndarray):
        labels, confs, b_boxes = [], [], []
        oom: bool = False
        if not self.ok:
            logger.warning(f"{LP} Something is not 'ok' (self.ok = False), skipping inference...")
        else:
            image_tensor = self._convert_image(image)
            # Tensor .to cannot be done in place, so we need to reassign
            image_tensor = image_tensor.to(self.device)
            preprocess = self.weights.transforms()
            batch = [preprocess(image_tensor)]
            del image_tensor, preprocess
            self.acquire_lock()
            detection_timer = time.perf_counter()
            try:
                prediction = self.model(batch)[0]
                del batch
                logger.debug(
                    f"perf:{LP}{self.processor}: '{self.name}' detection "
                    f"took: {time.perf_counter() - detection_timer:.5f} s"
                )
            except torch.cuda.OutOfMemoryError as oom_exc:
                logger.error(f"{LP} CUDA Out of Memory Error while detecting => {oom_exc}")
                oom = True
            except RuntimeError as runtime_exc:
                # torch docs say use RuntimeError for OOM catching
                logger.error(f"{LP} Runtime Error while detecting => {runtime_exc}")
                oom = True
            except Exception as detection_exc:
                logger.error(f"{LP} Error while detecting => {detection_exc}")
            else:
                labels = [
                    self.weights.meta["categories"][i] for i in prediction["labels"]
                ]
                # change bboxes to rounded ints
                prediction["boxes"] = prediction["boxes"].round().int()
                b_boxes = prediction["boxes"].tolist()
                confs = prediction["scores"].tolist()
                del prediction
            finally:
                # Always attempt to release the lock
                self.release_lock()

                logger.debug(
                    f"{LP} CUDA Max Memory >>> allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.0f} MB :: reserved: {torch.cuda.max_memory_reserved() / 1024 ** 2:.0f} MB"
                )
                logger.debug(
                    f"{LP} CUDA Current Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.0f} MB :: reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.0f} MB"
                )
                # logger.debug(f"{LP} memory stats: {torch.cuda.memory_stats()}")
                # logger.debug(f"{LP} memory summary: {torch.cuda.memory_summary(self.device)}")

        # This needs to be called outside of the try loop to allow memory to be freed in OOM cases
        if oom:
            if self.processor == ModelProcessor.GPU:
                logger.debug(f"{LP} OOM detected.... Attempting to clear GPU memory")
                torch.cuda.empty_cache()
            else:
                logger.warning(f"{LP} OOM detected.... cannot clear memory on CPU, try a lower performance model")

        result = DetectionResults(
            success=True if labels else False,
            type=self.config.type_of,
            processor=self.processor,
            name=self.name,
            results=[Result(label=labels[i], confidence=confs[i], bounding_box=b_boxes[i]) for i in range(len(labels))],
            # image=image,
        )
        return result

        # return {
        #     "success": True if labels else False,
        #     "type": self.config._model_type,
        #     "processor": self.processor,
        #     "model_name": self.name,
        #     "label": labels,
        #     "confidence": confs,
        #     "bounding_box": b_boxes,
        # }
