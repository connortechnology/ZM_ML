from __future__ import annotations

import time
import uuid
from logging import getLogger
from pathlib import Path
from typing import Optional, TYPE_CHECKING, List, Tuple, Union, AnyStr, Dict, Collection
from warnings import warn

try:
    import cv2
except ImportError:
    warn("OpenCV not installed! Please install/link to it!")

    raise
try:
    import onnxruntime as ort
except ImportError:
    warn("onnxruntime not installed, cannot use onnxruntime detectors")
    ort = None
import numpy as np

from ....Shared.Models.Enums import ModelProcessor
from ..file_locks import FileLock
from ....Shared.Models.config import DetectionResults, Result
from ...Log import SERVER_LOGGER_NAME

if TYPE_CHECKING:
    from ...Models.config import ORTModelConfig, BaseModelOptions, ORTModelOptions

logger = getLogger(SERVER_LOGGER_NAME)
LP: str = "ORT:"


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # logger.debug(f"{LP} {keep_indices.shape = }, {sorted_indices.shape = }")
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


class ORTDetector(FileLock):
    def __init__(self, model_config: ORTModelConfig):
        super().__init__()
        if not model_config:
            raise ValueError(f"{LP} no config passed!")
        self.config = model_config
        self.options: ORTModelOptions = self.config.detection_options
        self.processor: ModelProcessor = self.config.processor
        self.name: str = self.config.name
        self.model: Optional[str] = None
        self.id: uuid.uuid4 = self.config.id
        self.description: Optional[str] = model_config.description
        self.session: Optional[ort.InferenceSession] = None
        self.img_height: int = 0
        self.img_width: int = 0
        self.input_names: List[Optional[AnyStr]] = []
        self.LP: str = f"{LP}'{self.name}':"

        self.input_shape: Tuple[int, int, int] = (0, 0, 0)
        self.input_height: int = 0
        self.input_width: int = 0

        self.output_names: Optional[List[Optional[AnyStr]]] = None
        # Initialize model
        self.initialize_model(self.config.input)

    def __call__(self, image):
        return self.detect(image)

    def initialize_model(self, path: Path):
        logger.debug(
            f"{LP} loading model into processor [{self.processor}] memory: {self.name} ({self.id})"
        )
        providers = ["CPUExecutionProvider"]
        # Check if GPU is available
        if self.processor == ModelProcessor.GPU:
            if ort.get_device() == "GPU":
                providers.insert(0, "CUDAExecutionProvider")
            else:
                logger.warning(
                    f"{LP} GPU not available, using CPU for model: {self.name}"
                )
                self.processor = self.config.processor = ModelProcessor.CPU
        self.session = ort.InferenceSession(path, providers=providers)
        # Get model info
        self.get_input_details()
        self.get_output_details()
        self.create_lock()

    def detect(self, image: np.ndarray):
        input_tensor = self.prepare_input(image)
        logger.debug(
            f"{LP}detect: '{self.name}' ({self.processor}) - "
            f"input image {self.img_width}*{self.img_height} - model input {self.config.width}*{self.config.height}"
            f"{' [squared]' if self.config.square else ''}"
        )
        outputs = self.inference(input_tensor)
        b_boxes: List
        confs: List
        labels: List
        b_boxes, confs, labels = self.process_output(outputs)
        # labels = [self.config.labels[i] for i in labels]
        result = DetectionResults(
            success=True if labels else False,
            name=self.name,
            type=self.config.type_of,
            processor=self.processor,
            results=[
                Result(label=self.config.labels[labels[i]], confidence=confs[i], bounding_box=b_boxes[i])
                for i in range(len(labels))
            ],
        )
        return result

    def prepare_input(self, image: np.ndarray) -> np.ndarray:
        """Prepare a numpy array image for onnxruntime InferenceSession"""
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.config.square:
            logger.debug(f"{LP}detect: '{self.name}' - padding image to square")
            # Pad image to square
            input_img = np.pad(
                input_img,
                (
                    (0, max(self.img_height, self.img_width) - self.img_height),
                    (0, max(self.img_height, self.img_width) - self.img_width),
                    (0, 0),
                ),
                "constant",
                constant_values=0,
            )

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor) -> Optional[List[Union[np.ndarray, List, Dict]]]:
        """Perform inference on the prepared input image"""
        outputs: Optional[List[Union[np.ndarray, List, Dict]]] = None
        try:
            self.acquire_lock()
            start = time.perf_counter()
            outputs = self.session.run(
                self.output_names, {self.input_names[0]: input_tensor}
            )
        except Exception as e:
            logger.error(f"{LP} '{self.name}' Error while running model: {e}")
        else:
            logger.debug(
                f"perf:{LP}{self.processor}: '{self.name}' detection "
                f"took: {time.perf_counter() - start:.5f} s"
            )
        finally:
            self.release_lock()
        return outputs

    def process_output(self, output: List[Optional[np.ndarray]]) -> Tuple[List, List, List]:
        return_empty: bool = False
        if output:
            logger.debug(f"{LP} '{self.name}' output shapes: {[o.shape for o in output]}")
            # NAS new model.export() has FLAT (n, 7) and BATCHED  outputs
            num_outputs = len(output)
            if num_outputs == 1:
                if isinstance(output[0], np.ndarray):
                    if output[0].shape == (1, 84, 8400):
                        # v8
                        # (1, 84, 8400) -> (8400, 84)
                        predictions = np.squeeze(output[0]).T
                        logger.debug(f"{LP} yolov8 output shape = (1, 84, 8400) detected!")
                        # Filter out object confidence scores below threshold
                        scores = np.max(predictions[:, 4:], axis=1)
                        # predictions = predictions[scores > self.options.confidence, :]
                        # scores = scores[scores > self.options.confidence]

                        if len(scores) == 0:
                            return_empty = True

                        # Get the class with the highest confidence
                        # Get bounding boxes for each object
                        boxes = self.extract_boxes(predictions)
                        class_ids = np.argmax(predictions[:, 4:], axis=1)
                    elif len(output[0].shape) == 2 and output[0].shape[1] == 7:
                        logger.debug(f"{LP} YOLO-NAS model.export() FLAT output detected!")
                        # YLO-NAS .export FLAT output = (n, 7)
                        flat_predictions = output[0]
                        # pull out the class index and class score from the predictions
                        # and convert them to numpy arrays
                        flat_predictions = np.array(flat_predictions)
                        class_ids = flat_predictions[:, 6].astype(int)
                        scores = flat_predictions[:, 5]
                        # pull the boxes out of the predictions and convert them to a numpy array
                        boxes = flat_predictions[:, 1:4]
            elif num_outputs == 2:
                # NAS - .convert_to_onnx() output = [(1, 8400, 4), (1, 8400, 80)]
                if output[0].shape == (1, 8400, 4) and output[1].shape == (1, 8400, 80):
                    # YOLO-NAS
                    logger.debug(f"{LP} YOLO-NAS model.convert_to_onnx() output detected!")
                    _boxes: np.ndarray
                    raw_scores: np.ndarray
                    # get boxes and scores from outputs
                    _boxes, raw_scores = output
                    # find max from scores and flatten it [1, n, num_class] => [n]
                    scores = raw_scores.max(axis=2).flatten()
                    if len(scores) == 0:
                        return_empty = True
                    # squeeze boxes [1, n, 4] => [n, 4]
                    _boxes = np.squeeze(_boxes, 0)
                    _boxes = self.rescale_boxes(_boxes)
                    # find index from max scores (class_id) and flatten it [1, n, num_class] => [n]
                    class_ids = np.argmax(raw_scores, axis=2).flatten()
            elif num_outputs == 4:
                # NAS model.export() batch output len = 4
                # num_predictions [B, 1]
                # pred_boxes [B, N, 4]
                # pred_scores [B, N]
                # pred_classes [B, N]
                # Here B corresponds to batch size and N is the maximum number of detected objects per image
                if len(output[0].shape) == 2 and len(output[1].shape) == 3 and len(output[2].shape) == 2 and len(output[3].shape) == 2:
                    logger.debug(f"{LP} YOLO-NAS model.export() BATCHED output detected!")
                    batch_size = output[0].shape[0]
                    max_detections = output[1].shape[1]
                    num_predictions, pred_boxes, pred_scores, pred_classes = output
                    assert num_predictions.shape[0] == 1, "Only batch size of 1 is supported by this function"

                    num_predictions = int(num_predictions.item())
                    boxes = pred_boxes[0, :num_predictions]
                    scores = pred_scores[0, :num_predictions]
                    class_ids = pred_classes[0, :num_predictions]


        else:
            return_empty = True

        if return_empty:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,))

        indices = cv2.dnn.NMSBoxes(
            boxes, scores, self.options.confidence, self.options.nms
        )
        return boxes[indices].astype(np.int32).tolist(), scores[indices].astype(np.float32).tolist(), class_ids[
            indices].astype(np.int32).tolist()

    def extract_boxes(self, predictions):
        """Extract boxes from predictions, scale them and convert from xywh to xyxy format"""
        # Extract boxes from predictions
        boxes = predictions[:, :4]
        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)
        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)
        return boxes

    def rescale_boxes(self, boxes):
        """Rescale boxes to original image dimensions"""
        input_shape = np.array(
            [self.input_width, self.input_height, self.input_width, self.input_height]
        )
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array(
            [self.img_width, self.img_height, self.img_width, self.img_height]
        )
        return boxes

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
