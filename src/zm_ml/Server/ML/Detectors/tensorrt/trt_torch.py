"""
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
import uuid
from collections import namedtuple
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Tuple, List, AnyStr, Union

import numpy as np
from .....Server.Log import SERVER_LOGGER_NAME

logger = logging.getLogger(SERVER_LOGGER_NAME)

try:
    import torch
    from torchvision.ops import nms
except ImportError:
    torch = None

try:
    import tensorrt as trt
except ImportError:
    trt = None


from .....Server.Models.config import TRTModelConfig, TRTModelOptions
from .....Server.ML.file_locks import FileLock
from .....Shared.Models.config import DetectionResults
from .....Shared.Models.Enums import ModelProcessor


if TYPE_CHECKING:
    from src.zm_ml.Shared.configs import GlobalConfig

g: Optional[GlobalConfig] = None
LP: str = "TRT:torch::"


class TensorRtTorchDetector(FileLock):
    from .trt_base import TensorRtDetector

    class TRTModule(torch.nn.Module):
        dtypeMapping = {
            trt.bool: torch.bool,
            trt.int8: torch.int8,
            trt.int32: torch.int32,
            trt.float16: torch.float16,
            trt.float32: torch.float32
        }

        def __init__(self, weight: Union[str, Path],
                     device: Optional[torch.device]) -> None:
            if not torch:
                logger.error(f"{LP} Torch not installed, cannot use Torch detectors")
                return
            if not trt:
                logger.error(f"{LP} TensorRT not installed, cannot use TensorRT detectors")
                return
            super(self).__init__()
            self.weight = Path(weight) if isinstance(weight, str) else weight
            self.device = device if device is not None else torch.device('cuda:0')
            self.stream = torch.cuda.Stream(device=device)
            self.__init_engine()
            self.__init_bindings()

        def __init_engine(self) -> None:
            logger = trt.Logger(trt.Logger.WARNING)
            trt.init_libnvinfer_plugins(logger, namespace='')
            with trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(self.weight.read_bytes())

            context = model.create_execution_context()
            num_bindings = model.num_bindings
            num_inputs, num_outputs = 0, 0
            names = []
            for i in range(num_bindings):
                if model.binding_is_input(i):
                    num_inputs += 1
                else:
                    num_outputs += 1
                names.append(model.get_binding_name(i))

            self.bindings: List[int] = [0] * num_bindings

            self.num_bindings = num_bindings
            self.num_inputs = num_inputs
            self.num_outputs = num_outputs
            self.model = model
            self.context = context
            self.input_names = names[:num_inputs]
            self.output_names = names[num_inputs:]
            self.idx = list(range(self.num_outputs))

        def __init_bindings(self) -> None:
            idynamic = odynamic = False
            Tensor = namedtuple('Tensor', ('name', 'dtype', 'shape'))
            inp_info = []
            out_info = []
            for i, name in enumerate(self.input_names):
                assert self.model.get_binding_name(i) == name
                dtype = self.dtypeMapping[self.model.get_binding_dtype(i)]
                shape = tuple(self.model.get_binding_shape(i))
                if -1 in shape:
                    idynamic |= True
                inp_info.append(Tensor(name, dtype, shape))
            for i, name in enumerate(self.output_names):
                i += self.num_inputs
                assert self.model.get_binding_name(i) == name
                dtype = self.dtypeMapping[self.model.get_binding_dtype(i)]
                shape = tuple(self.model.get_binding_shape(i))
                if -1 in shape:
                    odynamic |= True
                out_info.append(Tensor(name, dtype, shape))

            if not odynamic:
                self.output_tensor = [
                    torch.empty(info.shape, dtype=info.dtype, device=self.device)
                    for info in out_info
                ]
            self.idynamic = idynamic
            self.odynamic = odynamic
            self.inp_info = inp_info
            self.out_info = out_info

        def set_profiler(self, profiler: Optional[trt.IProfiler]):
            self.context.profiler = profiler \
                if profiler is not None else trt.Profiler()

        def set_desired(self, desired: Optional[Union[List, Tuple]]):
            if isinstance(desired,
                          (list, tuple)) and len(desired) == self.num_outputs:
                self.idx = [self.output_names.index(i) for i in desired]

        def forward(self, *inputs) -> Union[Tuple, torch.Tensor]:

            assert len(inputs) == self.num_inputs
            contiguous_inputs: List[torch.Tensor] = [
                i.contiguous() for i in inputs
            ]

            for i in range(self.num_inputs):
                self.bindings[i] = contiguous_inputs[i].data_ptr()
                if self.idynamic:
                    self.context.set_binding_shape(
                        i, tuple(contiguous_inputs[i].shape))

            outputs: List[torch.Tensor] = []

            for i in range(self.num_outputs):
                j = i + self.num_inputs
                if self.odynamic:
                    shape = tuple(self.context.get_binding_shape(j))
                    output = torch.empty(size=shape,
                                         dtype=self.out_info[i].dtype,
                                         device=self.device)
                else:
                    output = self.output_tensor[i]
                self.bindings[j] = output.data_ptr()
                outputs.append(output)

            self.context.execute_async_v2(self.bindings, self.stream.cuda_stream)
            self.stream.synchronize()

            return tuple(outputs[i]
                         for i in self.idx) if len(outputs) > 1 else outputs[0]

    def __call__(self, image):
        return self.detect(image)

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

    def _prepare_tensor(self, image: np.ndarray) -> np.ndarray:
        # bgr to RGB
        image = image[:, :, ::-1]
        image = image.transpose([2, 0, 1])
        image = image[np.newaxis, ...]
        image = np.ascontiguousarray(image).astype(np.float32) / 255
        return torch.asarray(image, device=self.device)


    def __init__(self, model_config: TRTModelConfig):
        assert model_config, f"{LP} no config passed!"
        super().__init__()
        self.config: TRTModelConfig = model_config
        self.options: TRTModelOptions = self.config.detection_options
        self.processor: ModelProcessor = ModelProcessor.GPU
        self.name: str = self.config.name
        self.model: Optional[str] = None
        self.id: uuid.uuid4 = self.config.id
        self.description: Optional[str] = model_config.description
        self.LP: str = LP

        self.img_height: int = 0
        self.img_width: int = 0

        self.input_shape: Tuple[int, int, int] = (0, 0, 0)
        self.input_height: int = 0
        self.input_width: int = 0
        self.device: torch.device = self._get_device()

        self.input_names: List[Optional[AnyStr]] = []
        self.output_names: Optional[List[Optional[AnyStr]]] = None
        self.engine: Optional[TensorRtTorchDetector.TRTModule] = None
        self.context: Optional[trt.IExecutionContext] = None

        self.conf_th = self.options.confidence
        self.nms_threshold = self.options.nms
        self.trt_logger: TensorRtTorchDetector.TensorRtDetector.TrtLogger = self.TensorRtDetector.TrtLogger()
        logger.debug(f"{LP} about to call _load_engine()")

        self.engine = self.TRTModule(self.config.input, self.device)
        if not self.engine:
            # remove from available models
            logger.error(f"{LP} failed to load engine for '{self.name}'")
            from ....app import get_global_config

            for _model in get_global_config().available_models:
                if _model.id == self.id:
                    get_global_config().available_models.remove(_model)
                    break
            logger.error(
                f"cannot create detector for {self.config.name}"
            )
            return



        try:
            self.context = self.engine.create_execution_context()
        except Exception as e:
            logger.error(e, exc_info=True)
            raise RuntimeError("fail to allocate CUDA resources") from e


    def detect(self, image: np.ndarray):
        tensor = self._prepare_tensor(image)
        data = self.engine(tensor)
        bboxes, scores, labels = self.det_postprocess(data)
        if bboxes.numel() == 0:
            # if no bounding box
            logger.debug(f'{LP} no object!')
            return DetectionResults()
        bboxes -= dwdh
        bboxes /= ratio

    def _postprocess(self, data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        assert len(data) == 4
        iou_thres: float = 0.65
        num_dets, bboxes, scores, labels = data[0][0], data[1][0], data[2][
            0], data[3][0]
        # check score negative
        scores[scores < 0] = 1 + scores[scores < 0]
        nums = num_dets.item()
        if nums == 0:
            return bboxes.new_zeros((0, 4)), scores.new_zeros(
                (0,)), labels.new_zeros((0,))
        bboxes = bboxes[:nums]
        scores = scores[:nums]
        labels = labels[:nums]

        # add nms
        idx = nms(bboxes, scores, self.options.nms
                  )
        bboxes, scores, labels = bboxes[idx], scores[idx], labels[idx]
        return bboxes, scores, labels

