import time
from logging import getLogger

from PIL import Image
import cv2
import numpy as np

from ..file_locks import FileLock
from ...Models.config import TPUModelConfig
from ....Shared.Models.Enums import ModelType

from zm_ml.Server.app import SERVER_LOGGER_NAME
logger = getLogger(SERVER_LOGGER_NAME)
LP: str = "Coral:"

# global placeholders for TPU lib imports
common = None
detect = None
make_interpreter = None


class TpuDetector(FileLock):
    def __init__(self, model_config: TPUModelConfig):
        global LP, common, detect, make_interpreter
        try:
            from pycoral.adapters import common as common, detect as detect
            from pycoral.utils.edgetpu import make_interpreter as make_interpreter
        except ImportError:
            logger.warning(
                f"{LP} pycoral libs not installed, this is ok if you do not plan to use "
                f"TPU as detection processor. If you intend to use a TPU please install the TPU libs "
                f"and pycoral!"
            )
            raise ImportError("TPU libs not installed")
        else:
            logger.debug(f"{LP} the pycoral library has been successfully imported, initializing...")
        # Model init params
        self.config = model_config
        self.options = self.config.detection_options
        self.processor = self.config.processor
        self.name = self.config.name
        self.model = None
        if self.config.model_type == ModelType.FACE:
            LP = f"{LP}Face:"
        self.load_model()

    def load_model(self):
        from pycoral.utils.edgetpu import make_interpreter as make_interpreter
        logger.debug(
            f"{LP} loading model into {self.processor} processor memory: {self.name} ({self.config.id})"
        )
        t = time.perf_counter()
        try:
            self.model = make_interpreter(self.config.input.as_posix())
            self.model.allocate_tensors()
        except Exception as ex:
            ex = repr(ex)
            logger.error(f"{LP} failed to load model: {ex}")
            words = ex.split(" ")
            for word in words:
                if word.startswith("libedgetpu"):
                    logger.info(
                        f"{LP} TPU error detected (replace cable with a short high quality one, dont allow "
                        f"TPU/cable to move around). Reset the USB port or reboot!"
                    )
                    raise RuntimeError("TPU NO COMM")
        else:
            logger.debug(f"perf:{LP} loading took: {time.perf_counter() - t:.5f}s")

    def nms(self, objects, threshold):
        """Returns a list of objects passing the NMS.

        Args:
          objects: result candidates.
          threshold: the threshold of overlapping IoU to merge the boxes.

        Returns:
          A list of objects that pass the NMS.
        """
        if len(objects) == 1:
            return [0]

        boxes = np.array([o.bbox for o in objects])
        xmins = boxes[:, 0]
        ymins = boxes[:, 1]
        xmaxs = boxes[:, 2]
        ymaxs = boxes[:, 3]

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
                idxs, np.concatenate(([len(idxs) - 1], np.where(ious > threshold)[0])))

        return [objects[i] for i in selected_idxs]

    def detect(self, input_image: np.ndarray):
        from pycoral.adapters import common, detect
        b_boxes, labels, confs = [], [], []
        h, w = input_image.shape[:2]
        if not self.model:
            logger.warning(f"{LP} model not loaded? loading now...")
            self.load_model()
        t = time.perf_counter()
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = Image.fromarray(input_image)
        logger.debug(
            f"{LP}detect: input image {w}*{h}"
        )
        # scale = min(orig_width / w, orig_height / h)
        _, scale = common.set_resized_input(
            self.model,
            input_image.size,
            lambda size: input_image.resize(size, Image.ANTIALIAS),
        )
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
        finally:
            self.release_lock()
        logger.debug(f"{LP} {len(objs)} objects detected, applying NMS filter...")
        # Non Max Suppression
        nms_threshold = self.config.detection_options.nms
        objs = self.nms(objs, nms_threshold)
        logger.debug(f"{LP} {len(objs)} objects after NMS filtering with threshold: {nms_threshold}")
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

        return {
            "success": True if labels else False,
            "type": self.config.model_type,
            "processor": self.processor,
            "model_name": self.name,
            "label": labels,
            "confidence": confs,
            "bounding_box": b_boxes,
        }
