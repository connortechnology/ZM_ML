"""
This method is slower than using torch and trt together, but it works.

MIT License

Copyright (c) 2023 tripleMu <Modified by baudneo>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, List, Tuple, Union

# Need the logger to log import errors
from .....Server.Log import SERVER_LOGGER_NAME

logger = logging.getLogger(SERVER_LOGGER_NAME)
LP: str = "TRT:cudart:"

try:
    import tensorrt as trt
except ModuleNotFoundError:
    trt = None
    logger.error(f"{LP} TensorRT not found, cannot use TensorRT detectors")

try:
    from cuda import cudart
except ModuleNotFoundError:
    cudart = None
    logger.error("CUDA not found, cannot use TensorRT detectors")

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None
    logger.error("OpenCV not found, cannot use TensorRT detectors")

import numpy as np

from .....Shared.Models.Enums import ModelProcessor
from .....Server.Models.config import TRTModelConfig, TRTModelOptions
from .....Server.ML.file_locks import FileLock
from .....Shared.Models.config import DetectionResults, Result

if TYPE_CHECKING:
    from src.zm_ml.Shared.configs import GlobalConfig

g: Optional[GlobalConfig] = None
TRT_PLUGINS_LOCK = None

if trt is not None:

    class TrtLogger(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)

        def log(self, severity, msg, **kwargs):
            logger.log(self.get_severity(severity), msg, stacklevel=2)

        @staticmethod
        def get_severity(sev: trt.ILogger.Severity) -> int:
            if sev == trt.ILogger.VERBOSE:
                return logging.DEBUG
            elif sev == trt.ILogger.INFO:
                return logging.INFO
            elif sev == trt.ILogger.WARNING:
                return logging.WARNING
            elif sev == trt.ILogger.ERROR:
                return logging.ERROR
            elif sev == trt.ILogger.INTERNAL_ERROR:
                return logging.CRITICAL
            else:
                return logging.DEBUG


def blob(im: np.ndarray, return_seg: bool = False) -> Union[np.ndarray, Tuple]:
    seg = None
    if return_seg:
        seg = im.astype(np.float32) / 255
    im = im.transpose([2, 0, 1])
    im = im[np.newaxis, ...]
    im = np.ascontiguousarray(im).astype(np.float32) / 255
    if return_seg:
        return im, seg
    else:
        return im


def letterbox(
    im: np.ndarray,
    new_shape: Union[Tuple, List] = (640, 640),
    color: Union[Tuple, List] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # new_shape: [width, height]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    # Compute padding [width, height]
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, r, (dw, dh)


def _postprocess(
    data: Tuple[np.ndarray],
    shape: Union[Tuple, List],
    conf_thres: float = 0.25,
    iou_thres: float = 0.65,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape[0] // 4, shape[1] // 4  # 4x downsampling
    outputs = data[0]
    bboxes, scores, labels, maskconf = np.split(outputs, [4, 5, 6], 1)
    scores, labels = scores.squeeze(), labels.squeeze()
    idx = scores > conf_thres
    if not idx.any():  # no bounding boxes or seg were created
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0, 0, 0, 0), dtype=np.int32),
        )

    bboxes, scores, labels, maskconf = (
        bboxes[idx],
        scores[idx],
        labels[idx],
        maskconf[idx],
    )
    cvbboxes = np.concatenate([bboxes[:, :2], bboxes[:, 2:] - bboxes[:, :2]], 1)
    labels = labels.astype(np.int32)
    v0, v1 = map(int, (cv2.__version__).split(".")[:2])
    assert v0 == 4, "OpenCV version is wrong"
    if v1 > 6:
        idx = cv2.dnn.NMSBoxesBatched(cvbboxes, scores, labels, conf_thres, iou_thres)
    else:
        idx = cv2.dnn.NMSBoxes(cvbboxes, scores, conf_thres, iou_thres)
    bboxes, scores, labels, maskconf = (
        bboxes[idx],
        scores[idx],
        labels[idx],
        maskconf[idx],
    )

    # divide each score in scores by 100 using matrix multiplication
    scores = scores / 100

    return bboxes, scores, labels


@dataclass
class Tensor:
    name: str
    dtype: np.dtype
    shape: Tuple
    cpu: np.ndarray
    gpu: int


class TensorRtDetector(FileLock):
    def __init__(self, model_config: TRTModelConfig) -> None:
        assert model_config, f"{LP} no config passed!"

        status, self.stream = cudart.cudaStreamCreate()
        assert status.value == 0, f"{LP} failed to create cuda stream"
        self.trt_logger = TrtLogger()
        self.config: TRTModelConfig = model_config
        self.options: TRTModelOptions = self.config.detection_options
        # hard code GPU as TRT is Nvidia GPUs only.
        self.processor: ModelProcessor = ModelProcessor.GPU
        self.name: str = self.config.name
        self.runtime_engine: Optional[trt.ICudaEngine] = None
        self.id: uuid.uuid4 = self.config.id
        self.description: Optional[str] = self.config.description
        self.LP: str = LP
        super().__init__()
        self.__init_engine()
        self.__init_bindings()
        self.__warm_up()

    def __init_engine(self) -> None:
        global TRT_PLUGINS_LOCK

        logger.debug(
            f"{LP} initializing TensorRT engine: '{self.name}' ({self.id}) -- filename: {self.config.input.name}"
        )
        # Load plugins only 1 time
        if TRT_PLUGINS_LOCK is None:
            TRT_PLUGINS_LOCK = True
            logger.debug(f"{LP} initializing TensorRT plugins, should only happen once")
            trt.init_libnvinfer_plugins(self.trt_logger, namespace="")
        try:
            # De Serialize Engine
            with trt.Runtime(self.trt_logger) as runtime:
                self.runtime_engine = runtime.deserialize_cuda_engine(
                    self.config.input.read_bytes()
                )
            ctx = self.runtime_engine.create_execution_context()
        except Exception as ex:
            logger.error(f"{LP} {ex}", exc_info=True)
            raise ex
        else:
            # Grab and parse the names of the input and output bindings
            names = []
            self.num_bindings = self.runtime_engine.num_bindings
            self.bindings: List[int] = [0] * self.num_bindings
            num_inputs, num_outputs = 0, 0

            for i in range(self.runtime_engine.num_bindings):
                if self.runtime_engine.binding_is_input(i):
                    num_inputs += 1
                else:
                    num_outputs += 1
                names.append(self.runtime_engine.get_binding_name(i))
            self.num_inputs = num_inputs
            self.num_outputs = num_outputs
            # set the context
            self.context = ctx
            self.input_names = names[:num_inputs]
            self.output_names = names[num_inputs:]
            logger.debug(
                f"{LP} input names ({len(self.input_names)}): {self.input_names} "
                f"// output names ({len(self.output_names)}): {self.output_names}"
            )

    def __init_bindings(self) -> None:
        dynamic = False
        inp_info = []
        out_info = []
        out_ptrs = []
        _start = time.perf_counter()
        logger.debug(f"{LP} initializing input/output bindings")
        for i, name in enumerate(self.input_names):
            assert (
                self.runtime_engine.get_binding_name(i) == name
            ), f"{LP} binding name mismatch"
            dtype = trt.nptype(self.runtime_engine.get_binding_dtype(i))
            shape = tuple(self.runtime_engine.get_binding_shape(i))
            if -1 in shape:
                dynamic |= True
            if not dynamic:
                # set model input size
                self.input_height = shape[2]
                self.input_width = shape[3]
                cpu = np.empty(shape, dtype)
                status, gpu = cudart.cudaMallocAsync(cpu.nbytes, self.stream)
                assert status.value == 0, f"{LP} failed to allocate memory on GPU"
                # copy the data from the cpu to the gpu
                cudart.cudaMemcpyAsync(
                    gpu,
                    cpu.ctypes.data,
                    cpu.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self.stream,
                )
            else:
                cpu, gpu = np.empty(0), 0
            inp_info.append(Tensor(name, dtype, shape, cpu, gpu))
        for i, name in enumerate(self.output_names):
            i += self.num_inputs
            assert (
                self.runtime_engine.get_binding_name(i) == name
            ), f"{LP} binding name mismatch"
            dtype = trt.nptype(self.runtime_engine.get_binding_dtype(i))
            shape = tuple(self.runtime_engine.get_binding_shape(i))
            #
            if not dynamic:
                cpu = np.empty(shape, dtype=dtype)
                status, gpu = cudart.cudaMallocAsync(cpu.nbytes, self.stream)
                assert status.value == 0
                cudart.cudaMemcpyAsync(
                    gpu,
                    cpu.ctypes.data,
                    cpu.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self.stream,
                )
                out_ptrs.append(gpu)
            else:
                cpu, gpu = np.empty(0), 0
            out_info.append(Tensor(name, dtype, shape, cpu, gpu))

        self.is_dynamic = dynamic
        self.inp_info = inp_info
        self.out_info = out_info
        self.out_ptrs = out_ptrs
        logger.debug(
            f"{LP} initialized input/output bindings in {time.perf_counter() - _start:.3f} s. Dynamic input?: {dynamic}"
        )

    def __warm_up(self) -> None:
        if self.is_dynamic:
            logger.warning(
                f"{LP} The engine ({self.config.input}) has dynamic axes, you are responsible for warming "
                f"up the engine"
            )
            return
        logger.debug(f"{LP} Warming up the engine with 10 iterations...")
        _start = time.perf_counter()
        for _ in range(10):
            inputs = []
            for i in self.inp_info:
                inputs.append(i.cpu)
            self.__call__(inputs)
        logger.debug(
            f"perf:{LP} Engine warmed up in {time.perf_counter() - _start:.3f} seconds"
        )

    async def detect(self, input_image: np.ndarray) -> DetectionResults:
        labels = np.array([], dtype=np.int32)
        confs = np.array([], dtype=np.float32)
        b_boxes = np.array([], dtype=np.float32)
        # bgr, ratio, dwdh = letterbox(input_image, self.inp_info[0].shape[-2:])
        # dw, dh = int(dwdh[0]), int(dwdh[1])
        rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        self.img_height, self.img_width = rgb.shape[:2]
        # resize image to network size
        rgb = cv2.resize(rgb, (self.input_width, self.input_height))
        tensor = blob(rgb)
        # dwdh = np.array(dwdh * 2, dtype=np.float32)
        tensor = np.ascontiguousarray(tensor)
        # inference
        _start = time.perf_counter()
        try:
            self.acquire_lock()
            data = self.__call__(tensor)
        except Exception as all_ex:
            logger.error(f"{LP} EXCEPTION! {all_ex}")
            return DetectionResults(
                success=False,
                type=self.config.type_of,
                processor=self.processor,
                name=self.name,
                results=[],
            )
        else:
            logger.debug(
                f"perf:{LP} '{self.name}' inference took {time.perf_counter() - _start:.5f} seconds"
            )

        finally:
            self.release_lock()

        b_boxes, confs, labels = self.process_output(list(data))
        # logger.debug(f"{LP} {lbls = } -- {confs = } -- {b_boxes = }")
        result = DetectionResults(
            success=True if labels else False,
            type=self.config.type_of,
            processor=self.processor,
            name=self.name,
            results=[
                Result(
                    label=self.config.labels[labels[i]],
                    confidence=confs[i],
                    bounding_box=b_boxes[i],
                )
                for i in range(len(labels))
            ],
        )

        return result

    def set_profiler(self, profiler: Optional[trt.IProfiler]) -> None:
        self.context.profiler = (
            profiler
            if (profiler is not None and isinstance(profiler, trt.IProfiler))
            else trt.Profiler()
        )

    def __call__(self, *inputs) -> Union[Tuple, np.ndarray]:
        assert len(inputs) == self.num_inputs, f"{LP} incorrect number of inputs"
        contiguous_inputs: List[np.ndarray] = [np.ascontiguousarray(i) for i in inputs]
        _start = time.perf_counter()
        for i in range(self.num_inputs):
            if self.is_dynamic:
                logger.debug(f"{LP} setting binding shape for input layer: {i}")
                self.context.set_binding_shape(i, tuple(contiguous_inputs[i].shape))
                status, self.inp_info[i].gpu = cudart.cudaMallocAsync(
                    contiguous_inputs[i].nbytes, self.stream
                )
                assert (
                    status.value == 0
                ), f"{LP} failed to allocate memory on GPU for dynamic input"
            cudart.cudaMemcpyAsync(
                self.inp_info[i].gpu,
                contiguous_inputs[i].ctypes.data,
                contiguous_inputs[i].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            )
            # copy the gpu pointer to the bindings
            self.bindings[i] = self.inp_info[i].gpu
        output_gpu_ptrs: List[int] = []
        outputs: List[np.ndarray] = []

        for i in range(self.num_outputs):
            j = i + self.num_inputs
            if self.is_dynamic:
                shape = tuple(self.context.get_binding_shape(j))
                dtype = self.out_info[i].dtype
                cpu = np.empty(shape, dtype=dtype)
                status, gpu = cudart.cudaMallocAsync(cpu.nbytes, self.stream)
                assert (
                    status.value == 0
                ), f"{LP} failed to allocate memory on GPU from dynamic input"
                cudart.cudaMemcpyAsync(
                    gpu,
                    cpu.ctypes.data,
                    cpu.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self.stream,
                )
            else:
                cpu = self.out_info[i].cpu
                gpu = self.out_info[i].gpu
            outputs.append(cpu)
            output_gpu_ptrs.append(gpu)
            self.bindings[j] = gpu
        _s = time.perf_counter()
        self.context.execute_async_v2(self.bindings, self.stream)
        cudart.cudaStreamSynchronize(self.stream)

        for i, o in enumerate(output_gpu_ptrs):
            cudart.cudaMemcpyAsync(
                outputs[i].ctypes.data,
                o,
                outputs[i].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            )
        # for output in outputs:
        #     if isinstance(output, np.ndarray):
        #         logger.debug(f"{LP} {output.shape = }")

        # return tuple(outputs) if len(outputs) > 1 else outputs[0]
        return outputs

    def process_output(
        self, output: List[Optional[np.ndarray]]
    ) -> Tuple[List, List, List]:
        return_empty: bool = False
        boxes: np.ndarray = np.array([], dtype=np.float32)
        scores: np.ndarray = np.array([], dtype=np.float32)
        labels: np.ndarray = np.array([], dtype=np.int32)
        nms, conf = self.options.nms, self.options.confidence
        if output:
            # Fixme: allow for custom class length (80 is for COCO)
            output_shape = [o.shape for o in output]
            legacy_shape = [(1, 8400, 4), (1, 8400, 80)]
            new_flat_shape = [(1, 8400, 80), (1, 8400, 4)]
            num_outputs = len(output)
            logger.debug(
                f"{LP} '{self.name}' output shapes ({num_outputs}): {output_shape}"
            )
            shape_str = ""
            # Deci AI super-gradients new model.export() has FLAT (n, 7) and BATCHED outputs
            if num_outputs == 1:
                if isinstance(output[0], np.ndarray):
                    # prettrained: (1, 84, 8400)
                    # dfo with 2 classes and 1 background: (1, 6, 8400)
                    if output[0].shape == (1, 84, 8400):
                        # v8
                        # (1, 84, 8400) -> (8400, 84)
                        predictions = np.squeeze(output[0]).T
                        logger.debug(
                            f"{LP} yolov8 output shape = (1, <X>, 8400) detected!"
                        )
                        # logger.debug(f"{LP} predictions.shape = {predictions.shape} --- {predictions =}")
                        # Filter out object confidence scores below threshold
                        scores = np.max(predictions[:, 4:], axis=1)

                        if len(scores) == 0:
                            logger.debug(
                                f"{LP} '{self.name}' no scores above confidence threshold"
                            )
                            return_empty = True

                        # Get bounding boxes for each object
                        boxes = self.extract_boxes(predictions)
                        # Get the class ids with the highest confidence
                        class_ids = np.argmax(predictions[:, 4:], axis=1)
                    elif len(output[0].shape) == 2 and output[0].shape[1] == 7:
                        logger.debug(
                            f"{LP} YOLO-NAS model.export() FLAT output detected!"
                        )
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
                if output_shape == legacy_shape or output_shape == new_flat_shape:
                    # NAS - .convert_to_onnx() output = [(1, 8400, 4), (1, 8400, 80)] / NEW "FLAT" = [(1, 8400, 80), (1, 8400, 4)]
                    if output_shape == legacy_shape:
                        shape_str = "convert_to_onnx() [Legacy]"
                        _boxes, raw_scores = output

                    elif output_shape == new_flat_shape:
                        shape_str = "export() [NEW Flat]"
                        raw_scores, _boxes = output
                    else:
                        shape_str = "<UNKNOWN>"
                        _boxes, raw_scores = output
                    # YOLO-NAS
                    logger.debug(f"{LP} YOLO-NAS model.{shape_str} output detected!")
                    _boxes: np.ndarray
                    raw_scores: np.ndarray
                    # get boxes and scores from outputs
                    # find max from scores and flatten it [1, n, num_class] => [n]
                    scores = raw_scores.max(axis=2).flatten()
                    if len(scores) == 0:
                        return_empty = True
                    # squeeze boxes [1, n, 4] => [n, 4]
                    _boxes = np.squeeze(_boxes, 0)
                    boxes = self.rescale_boxes(_boxes)
                    # find index from max scores (class_id) and flatten it [1, n, num_class] => [n]
                    class_ids = np.argmax(raw_scores, axis=2).flatten()
                else:
                    logger.warning(
                        f"{LP} '{self.name}' has unknown output shape: {output_shape}, should be: Legacy: {legacy_shape} or New Flat: {new_flat_shape}"
                    )
            elif num_outputs == 4:
                # NAS model.export() batch output len = 4
                # num_predictions [B, 1]
                # pred_boxes [B, N, 4]
                # pred_scores [B, N]
                # pred_classes [B, N]
                # Here B corresponds to batch size and N is the maximum number of detected objects per image
                if (
                    len(output[0].shape) == 2
                    and len(output[1].shape) == 3
                    and len(output[2].shape) == 2
                    and len(output[3].shape) == 2
                ):
                    logger.debug(
                        f"{LP} YOLO-NAS model.export() BATCHED output detected!"
                    )
                    batch_size = output[0].shape[0]
                    max_detections = output[1].shape[1]
                    num_predictions, pred_boxes, pred_scores, pred_classes = output
                    assert (
                        num_predictions.shape[0] == 1
                    ), "Only batch size of 1 is supported by this function"

                    num_predictions = int(num_predictions.item())
                    boxes = pred_boxes[0, :num_predictions]
                    scores = pred_scores[0, :num_predictions]
                    class_ids = pred_classes[0, :num_predictions]
        else:
            return_empty = True

        if return_empty:
            logger.warning(f"{LP} '{self.name}' return_empty = True !!")
            return [], [], []

        indices = cv2.dnn.NMSBoxes(
            boxes, scores, conf, nms
        )
        if len(indices) == 0:
            logger.debug(f"{LP} no detections after filter by NMS ({nms}) and confidence ({conf})")
            return [], [], []

        boxes = (boxes[indices],)
        scores = scores[indices]
        class_ids = class_ids[indices]
        if isinstance(boxes, tuple):
            if len(boxes) == 1:
                boxes = boxes[0]
        if isinstance(scores, tuple):
            if len(scores) == 1:
                scores = scores[0]
        return (
            boxes.astype(np.int32).tolist(),
            scores.astype(np.float32).tolist(),
            class_ids.astype(np.int32).tolist(),
        )

    def extract_boxes(self, predictions):
        """Extract boxes from predictions, scale them and convert from xywh to xyxy format"""
        # Extract boxes from predictions
        boxes = predictions[:, :4]
        logger.debug(f"{LP} {boxes.shape = }")
        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)
        # Convert boxes to xyxy format
        from ..onnx_runtime import xywh2xyxy

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
