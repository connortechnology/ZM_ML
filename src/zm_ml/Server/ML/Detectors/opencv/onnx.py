from __future__ import annotations
import time
from functools import lru_cache
from logging import getLogger
from typing import Optional, TYPE_CHECKING, List, Tuple
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

LP: str = "OpenCV:ONNX:"
logger = getLogger(SERVER_LOGGER_NAME)


class CV2ONNXDetector(CV2Base):
    _classes: Optional[List] = None

    def __init__(self, model_config: CV2YOLOModelConfig):
        super().__init__(model_config)

        self.LP = f"{LP}{self.name}:"
        # Model init params not initialized in super()
        self.model: Optional[cv2.dnn.Net] = None
        self.load_model()

    @property
    @lru_cache
    def classes(self):
        if self._classes is None:
            self._classes = self.config.labels
        return self._classes

    @classes.setter
    def classes(self, value: List):
        self._classes = value

    def load_model(self):
        logger.debug(
            f"{self.LP} attempting to load model into processor [{self.processor}] memory: {self.name} ({self.id})"
        )
        load_timer = time.perf_counter()
        if not self.config.input.is_absolute():
            self.config.input = self.config.input.expanduser().resolve()
        try:
            model_file: str = self.config.input.as_posix()
            config_file: Optional[str] = None
            if self.config.config:
                pass
            logger.info(f"{self.LP} loading -> model: {model_file}")
            self.net = cv2.dnn.readNetFromONNX(model_file)
        except Exception as model_load_exc:
            logger.error(
                f"{self.LP} Error while loading model file and/or config! "
                f"(May need to re-download the model/cfg file) => {model_load_exc}"
            )
            raise model_load_exc
        if self.net is not None:
            self.cv2_processor_check()
            logger.debug(
                f"{self.LP} set CUDA/cuDNN backend and target"
            ) if self.processor == ModelProcessor.GPU else None

            logger.debug(
                f"perf:{self.LP} '{self.name}' loading completed in {time.perf_counter() - load_timer:.5f} s"
            )
        else:
            logger.debug(
                f"perf:{self.LP} '{self.name}' FAILED in {time.perf_counter() - load_timer:.5f} s"
            )

    def detect(self, input_image: np.ndarray):
        if input_image is None:
            raise ValueError(f"{self.LP} no image passed!")
        if not self.net:
            self.load_model()
        input_h, input_w = self.config.height, self.config.width
        if self.config.square:
            input_image = self.square_image(input_image)
        img_h, img_w = input_image.shape[:2]
        x_factor = img_w / input_w
        y_factor = img_h / input_h
        labels, confs, b_boxes = [], [], []
        _l, _c, _b = [], [], []
        nms_threshold, conf_threshold = self.options.nms, self.options.confidence

        logger.debug(
            f"{self.LP}detect: '{self.name}' ({self.processor}) - "
            f"Image: {img_w}*{img_h} - Model Input: {input_w}*{input_h}"
            f" - Scale: x[{x_factor}] - y[{y_factor}]{' [squared]' if self.config.square else ''}"
        )
        self.acquire_lock()
        detection_timer = time.perf_counter()
        onnx_type = self.config.onnx_type
        try:
            blob = cv2.dnn.blobFromImage(
                input_image, 1 / 255.0, (input_w, input_h), swapRB=True, crop=False
            )
            self.net.setInput(blob)
            preds: Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
            preds = self.net.forward(self.net.getUnconnectedOutLayersNames())

        except Exception as all_ex:
            err_msg = repr(all_ex)
            # cv2.error: OpenCV(4.2.0) /home/<Someone>/opencv/modules/dnn/src/cuda/execution.hpp:52: error: (-217:Gpu
            # API call) invalid device function in function 'make_policy'
            logger.error(
                f"{self.LP} exception during detection of '{self.name}' -> {all_ex}",
                exc_info=True,
            )
            # OpenCV 4.7.0 Weird Error fixed with rolling fix
            # OpenCV:YOLO: exception during detection -> OpenCV(4.7.0-dev) /opt/opencv/modules/dnn/src/layers/cpu_kernels/conv_winograd_f63.cpp:401: error: (-215:Assertion failed) CONV_WINO_IBLOCK == 3 && CONV_WINO_KBLOCK == 4 && CONV_WINO_ATOM_F32 == 4 in function 'winofunc_BtXB_8x8_f32'
            if err_msg.find("-215:Assertion failed") > 0:
                if err_msg.find(
                    "total(srcShape, srcRange.start, srcRange.end) == maskTotal"
                ):
                    _msg = (
                        f"{self.LP} It seems there is a mismatch in the input height [{input_h}] and/or width [{input_w}]"
                        f" of the model and what the model is trained for. Please specify 'height' and 'width' "
                        f"in the [{self.name}] model config!"
                    )
                if err_msg.find("CONV_WINO_IBLOCK == 3 && CONV_WINO_KBLOCK == 4") > 0:
                    _msg = (
                        f"{self.LP} OpenCV 4.7.x WEIRD bug detected! "
                        f"Please update to OpenCV 4.7.1+ or 4.6.0 or less!"
                    )
            elif err_msg.find("-217:Gpu") > 0:
                if (
                    err_msg.find("'make_policy'") > 0
                    and self.processor == ModelProcessor.GPU
                ):
                    _msg = (
                        f"{self.LP} (-217:Gpu # API call) invalid device function in function 'make_policy' - "
                        f"This happens when OpenCV is compiled with the incorrect Compute Capability "
                        f"(CUDA_ARCH_BIN). There is a high probability that you need to recompile OpenCV with "
                        f"the correct CUDA_ARCH_BIN before GPU detections will work properly!"
                    )
                    logger.error(_msg)
                    raise RuntimeError(_msg)
            elif err_msg.find("Can't infer a dim denoted by -1 in function") > 0:
                if str(self.config.sub_framework.value).casefold() == "onnx":
                    pass
            else:
                raise all_ex
            logger.error(_msg)

        else:
            if preds is not None:

                if len(preds) == 2:
                    logger.info(f"{self.LP} YOLO-NAS output detected...")

                    try:
                        boxes: np.ndarray
                        results: np.ndarray
                        boxes, results = (preds[1], preds[0])

                        # [1, n, 4] >>> [n, 4]
                        boxes = np.squeeze(boxes, 0)
                        logger.debug(f"{self.LP} {boxes = }")
                        # find max from scores and flatten it [1, n, num_class] => [n]
                        scores = results.max(axis=2).flatten()
                        logger.debug(f"{self.LP} {scores = }")

                        # find index from max scores (class_id) and flatten it [1, n, num_class] => [n]
                        classes = np.argmax(results, axis=2).flatten()

                    except Exception as all_ex:
                        logger.error(
                            f"{self.LP} exception during detection of '{self.name}' -> {all_ex}",
                            exc_info=True,
                        )
                    else:
                        # get the index of boxes that score higher than threshold
                        selected = cv2.dnn.NMSBoxes(
                            boxes, scores, conf_threshold, nms_threshold
                        )
                        for i in selected:
                            # use the nms filtered index to get score, class and box
                            _boxes = boxes[i, :].astype(np.int32).flatten()
                            score = float(scores[i])
                            label = self.classes[classes[i]]

                            # scale boxes
                            x, y, w, h = (
                                int(_boxes[0] * x_factor),
                                int(_boxes[1] * y_factor),
                                int(_boxes[2] * x_factor),
                                int(_boxes[3] * y_factor),
                            )

                            b_boxes.append([x, y, w, h])
                            confs.append(score)
                            labels.append(label)

                elif len(preds) == 1:
                    logger.debug(f"{self.LP} preds TYPE = {type(preds)} ---- {preds = }")

                    logger.info(f"{self.LP} Assuming this is a YOLO v8 ONNX model based on output shape...")

                    # yolov8 to v5 output shape
                    if preds[0].shape == (1, 84, 8400):
                        #  (1, 84, 8400)
                        preds = preds[0].transpose((0, 2, 1))
                        #  (1, 8400, 84)
                    num_rows = preds[0].shape[0]

                    results = preds[0]
                    class_len = len(self.classes)
                    try:
                        for i in range(num_rows):
                            result = results[i]

                            conf = result[4]

                            classes_score = result[4:]
                            class_id = np.argmax(classes_score)
                            if int(class_id) > class_len:
                                logger.error(
                                    f"{self.LP} '{self.name}' Detected class_id [{class_id}] is higher than the number of classes "
                                    f"defined in the model config [{class_len}] ????"
                                )
                                continue
                            if classes_score[class_id] >= conf_threshold:
                                # logger.debug(f"{self.LP} {result = }")
                                _c.append(conf)
                                label = self.classes[int(class_id)]
                                _l.append(label)

                                # extract boxes
                                x, y, w, h = (
                                    result[0].item(),
                                    result[1].item(),
                                    result[2].item(),
                                    result[3].item(),
                                )
                                logger.debug(f"{self.LP} {label} ({float(conf)}) - {x}, {y}, {w}. {h}")

                                left = int((x - 0.5 * w) * x_factor)
                                top = int((y - 0.5 * h) * y_factor)
                                width = int(w * x_factor)
                                height = int(h * y_factor)
                                box = [left, top, width, height]
                                #box = [left, top, left+width, top+height]
                                # logger.debug(f"DBG>>> ADDING BOX to _b: {box = }")
                                _b.append(box)
                    except IndexError as ex:
                        logger.error(
                            f"{self.LP} IndexError during detection of '{self.name}' -> Length: {len(self.classes)} "
                            f"--  Index: {class_id}"
                        )
                    except Exception as ex:
                        logger.error(
                            f"{self.LP} exception during detection of '{self.name}' -> {ex}",
                            exc_info=True,
                        )
                    finally:
                        if _l:
                            indexes = cv2.dnn.NMSBoxes(
                                _b, _c, conf_threshold, nms_threshold
                            )
                            logger.debug(f"{self.LP} {indexes = }")
                            for i in indexes:
                                labels.append(_l[i])
                                confs.append(_c[i])
                                b_boxes.append(_b[i])

        finally:
            self.release_lock()
        logger.debug(
            f"perf:{self.LP}{self.processor}: '{self.name}' detection "
            f"took: {time.perf_counter() - detection_timer:.5f} s"
        )
        try:
            result = DetectionResults(
                success=True if labels else False,
                type=self.config.type_of,
                processor=self.processor,
                name=self.name,
                results=[
                    Result(label=labels[i], confidence=confs[i], bounding_box=b_boxes[i])
                    for i in range(len(labels))
                ],
            )
        except Exception as ex:
            logger.error(f"{self.LP} Exception during creation of DetectionResults() of '{self.name}' -> {ex}")
            logger.debug(f"{self.LP} {type(boxes) = } -------  {boxes = }")

        else:
            return result
