from __future__ import annotations
import time
from logging import getLogger
from typing import List, Optional, TYPE_CHECKING
import warnings

from PIL import Image
import numpy as np

try:
    import cv2
except ImportError:
    warnings.warn("OpenCV not installed, please install OpenCV!", ImportWarning)
    cv2 = None
    raise
try:
    import pycoral
    from pycoral.adapters import common, detect
    from pycoral.utils.edgetpu import make_interpreter, list_edge_tpus
except ImportError:
    warnings.warn(
        "pycoral not installed, this is ok if you do not plan to use TPU as detection processor. "
        "If you intend to use a TPU please install the TPU libs and pycoral!",
        ImportWarning,
    )
    pycoral = None
    make_interpreter = None
    common = None
    detect = None
    list_edge_tpus = None
try:
    import tflite_runtime
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    warnings.warn(
        "tflite_runtime not installed, this is ok if you do not plan to use TPU as detection processor. "
        "If you intend to use a TPU please install the TPU libs and pycoral!",
        ImportWarning,
    )
    Interpreter = None
    tflite_runtime = None

from ..file_locks import FileLock
from ....Shared.Models.Enums import ModelType, ModelProcessor
from zm_ml.Server.app import SERVER_LOGGER_NAME
from ....Shared.Models.config import DetectionResults, Result


if TYPE_CHECKING:
    from ...Models.config import TPUModelConfig, TPUModelOptions

logger = getLogger(SERVER_LOGGER_NAME)
LP: str = "Coral:"


class TpuDetector(FileLock):
    def __init__(self, model_config: TPUModelConfig):
        global LP

        if pycoral is None:
            raise ImportError(f"{LP} pycoral is not installed, cannot use TPU detectors")

        elif tflite_runtime is None:
            raise ImportError(
                f"{LP} tflite_runtime is not installed, cannot use TPU detectors"
            )
            return
        # check if there is a tpu device
        if not list_edge_tpus():
            raise RuntimeError(f"{LP} no TPU devices found, cannot use TPU detectors")
            return

        # Model init params
        self.config: TPUModelConfig = model_config
        self.options: TPUModelOptions = self.config.detection_options
        self.processor: ModelProcessor = self.config.processor
        self.name: str = self.config.name
        self.model: Optional[Interpreter] = None
        if self.config.model_type == ModelType.FACE:
            logger.debug(
                f"{LP} ModelType=Face, this is for identification purposes only"
            )
            LP = f"{LP}Face:"
        self.load_model()

    def load_model(self):
        logger.debug(
            f"{LP} loading model into {self.processor.upper()} processor memory: {self.name} ({self.config.id})"
        )
        t = time.perf_counter()
        try:
            self.model: Interpreter = make_interpreter(self.config.input.as_posix())
            self.model.allocate_tensors()
        except Exception as ex:
            ex = repr(ex)
            logger.error(
                f"{LP} failed to load model at make_interpreter() and allocate_tensors(): {ex}"
            )
            words = ex.split(" ")
            for word in words:
                if word.startswith("libedgetpu"):
                    logger.info(
                        f"{LP} TPU error detected. It could be a bad cable, needing to unplug/replug in the device or a system reboot."
                    )
        else:
            logger.debug(f"perf:{LP} loading took: {time.perf_counter() - t:.5f}s")

    def nms(self, objects: List, threshold: float) -> List:
        """Returns a list of objects passing the NMS filter.

        Args:
          objects: result candidates.
          threshold: the threshold of overlapping IoU to merge the boxes.

        Returns:
          A list of objects that pass the NMS.
        """
        # TODO: Make class (label) aware and only filter out same class members?
        timer = time.perf_counter()
        if len(objects) == 1:
            logger.debug(f"{LP} only 1 object, no NMS needed")
        elif len(objects) > 1:
            boxes = np.array([o.bbox for o in objects])
            try:
                xmins = boxes[:, 0]
                ymins = boxes[:, 1]
                xmaxs = boxes[:, 2]
                ymaxs = boxes[:, 3]
            except IndexError as e:
                logger.error(f"{LP} {e}")
                logger.debug(f"{LP} numpy.array NMS boxes: {boxes}")
                raise IndexError
            else:
                areas = (xmaxs - xmins) * (ymaxs - ymins)
                scores = [o.score for o in objects]
                idxs = np.argsort(scores)

                selected_idxs = []
                while idxs.size != 0:
                    selected_idx = idxs[-1]
                    selected_idxs.append(selected_idx)

                    overlapped_xmins = np.maximum(xmins[selected_idx], xmins[idxs[:-1]])
                    overlapped_ymins = np.maximum(ymins[selected_idx], ymins[idxs[:-1]])
                    overlapped_xmaxs = np.minimum(xmaxs[selected_idx], xmaxs[idxs[:-1]])
                    overlapped_ymaxs = np.minimum(ymaxs[selected_idx], ymaxs[idxs[:-1]])

                    w = np.maximum(0, overlapped_xmaxs - overlapped_xmins)
                    h = np.maximum(0, overlapped_ymaxs - overlapped_ymins)

                    intersections = w * h
                    unions = areas[idxs[:-1]] + areas[selected_idx] - intersections
                    ious = intersections / unions

                    idxs = np.delete(
                        idxs,
                        np.concatenate(
                            ([len(idxs) - 1], np.where(ious > threshold)[0])
                        ),
                    )
            objects = [objects[i] for i in selected_idxs]
            logger.info(f"perf:{LP} NMS took: {time.perf_counter() - timer:.5f}s")
        return objects

    def detect(self, input_image: np.ndarray):
        """Performs object detection on the input image."""
        b_boxes, labels, confs = [], [], []
        h, w = input_image.shape[:2]
        nms = self.options.nms
        conf_threshold = self.config.detection_options.confidence
        if not self.model:
            logger.warning(f"{LP} model not loaded? loading now...")
            self.load_model()
        t = time.perf_counter()
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = Image.fromarray(input_image)
        nms_str = f" - nms: {nms.threshold}" if nms.enabled else ""
        logger.debug(
            f"{LP}detect: input image {w}*{h} - confidence: {conf_threshold}{nms_str}"
        )
        # scale = min(orig_width / w, orig_height / h)
        _, scale = common.set_resized_input(
            self.model,
            input_image.size,
            lambda size: input_image.resize(size, Image.LANCZOS),
        )
        objs: List[detect.Object]
        try:
            self.acquire_lock()
            self.model.invoke()
        except Exception as ex:
            logger.error(f"{LP} TPU error while calling invoke(): {ex}")
            raise ex
        else:
            objs = detect.get_objects(self.model, self.options.confidence, scale)
            logger.debug(
                f"perf:{LP} '{self.name}' detection took: {time.perf_counter() - t:.5f}s"
            )
            _obj_len = len(objs)
            logger.debug(f"{LP} RAW:: {_obj_len}")
        finally:
            self.release_lock()
        if objs:
            # Non Max Suppression
            if nms.enabled:
                objs = self.nms(objs, nms.threshold)
                logger.info(
                    f"{LP} {len(objs)}/{_obj_len} objects after NMS filtering with threshold: {nms.threshold}"
                )
            else:
                logger.info(f"{LP} NMS disabled, {_obj_len} objects detected")
            for obj in objs:
                b_boxes.append(
                    [
                        int(round(obj.bbox.xmin)),
                        int(round(obj.bbox.ymin)),
                        int(round(obj.bbox.xmax)),
                        int(round(obj.bbox.ymax)),
                    ]
                )
                labels.append(self.config.labels[obj.id])
                confs.append(float(obj.score))
        else:
            logger.warning(f"{LP} nothing returned from invoke()... ?")

        result = DetectionResults(
            success=True if labels else False,
            type=self.config.model_type,
            processor=self.processor,
            model_name=self.name,
            results=[
                Result(label=labels[i], confidence=confs[i], bounding_box=b_boxes[i])
                for i in range(len(labels))
            ],
        )

        return result
