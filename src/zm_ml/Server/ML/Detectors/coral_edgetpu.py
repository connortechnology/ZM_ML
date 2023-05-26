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

    def detect(self, input_image: np.ndarray):
        from pycoral.adapters import common as common, detect as detect
        b_boxes, labels, confs = [], [], []
        h, w = input_image.shape[:2]
        _h, _w = self.config.height, self.config.width
        if not self.model:
            logger.warning(f"{LP} model not loaded? loading now...")
            self.load_model()
        x_factor: float = 1.00
        y_factor: float = 1.00
        t = time.perf_counter()
        model_resize = False
        if _h != h or _w != w:
            model_resize = True
            input_image = cv2.resize(
                input_image, (_w, _h), interpolation=cv2.INTER_AREA
            )
            # get scaling so we can make correct bounding boxes
            x_factor = w / _w
            y_factor = h / _h

        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = Image.fromarray(input_image)
        logger.debug(
            f"{LP}detect: input image {w}*{h} model input {_w}*{_h}{' (resized)' if model_resize else ''}"
        )

        _, scale = common.set_resized_input(
            self.model,
            input_image.size,
            lambda size: input_image.resize(size, Image.ANTIALIAS),
            # lambda size: input_image.resize(size, Image.LANCZOS),
        )
        try:
            self.acquire_lock()
            self.model.invoke()
        except Exception as ex:
            logger.error(f"{LP} TPU error: {ex}")
            raise ex
        else:
            objs = detect.get_objects(self.model, self.options.confidence, scale)
            logger.debug(
                f"perf:{LP} '{self.name}' detection took: {time.perf_counter() - t:.5f}s"
            )
        finally:
            self.release_lock()
        logger.debug(f"{LP} {len(objs)} objects detected")
        logger.debug(f"{LP} {len(self.config.labels) = } total labels in the supplied labels from config (COCO17 ?)")
        for obj in objs:
            logger.debug(f"DBG>>> {type(obj) = } || {obj = }")
            logger.debug(f"DBG>>> {type(obj.id) = } || {obj.id = } || {len(self.config.labels) = }")
            if obj.id > len(self.config.labels):
                logger.warning(f"{LP} {obj.id = } is out of range of the supplied labels, does obj.id start at 1 instead of 0 for calling an array?")
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

        if model_resize and labels:
            logger.debug(
                f"{LP} The image was resized before processing by the 'model width/height', scaling "
                f"bounding boxes in image by factors of -> x={x_factor:.4} "
                f"y={y_factor:.4}",
            )
            for box in b_boxes:
                box[0] = round(box[0] * x_factor)
                box[1] = round(box[1] * y_factor)
                box[2] = round(box[2] * x_factor)
                box[3] = round(box[3] * y_factor)

        return {
            "success": True if labels else False,
            "type": self.config.model_type,
            "processor": self.processor,
            "model_name": self.name,
            "label": labels,
            "confidence": confs,
            "bounding_box": b_boxes,
        }
