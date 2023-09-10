from enum import Enum
from typing import Any
import logging

from pydantic_core import CoreSchema, core_schema

from pydantic import BaseModel, GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue

from ...Server.Log import SERVER_LOGGER_NAME

logger = logging.getLogger(SERVER_LOGGER_NAME)


class MyEnumBase(Enum):
    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any, _handler: GetCoreSchemaHandler) -> CoreSchema:

        return core_schema.no_info_after_validator_function(
            lambda x: getattr(cls, x.upper()),
            core_schema.any_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(lambda x: x),
        )

    @classmethod
    def __get_pydantic_json_schema__(cls, _core_schema: CoreSchema, _handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        print(f"DEBUG: {type(cls) = } -- {cls = } -- {cls.__dict__ = }")
        return {'enum': [m.name for m in cls], 'type': 'string'}


class ModelType(str, Enum):
    OBJECT = "object"
    FACE = "face"
    ALPR = "alpr"
    DEFAULT = OBJECT

    def __repr__(self):
        return f"<{self.__class__.__name__}: {str(self.name).lower()} detection>"

    def __str__(self):
        return self.__repr__()


class OpenCVSubFrameWork(str, Enum):
    DARKNET = "darknet"
    CAFFE = "caffe"
    TENSORFLOW = "tensorflow"
    TORCH = "torch"
    VINO = "vino"
    ONNX = "onnx"
    DEFAULT = DARKNET

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name}>"

    def __str__(self):
        return f"{self.name}"


class HTTPSubFrameWork(str, Enum):
    NONE = "none"
    VIREL = "virel"
    REKOGNITION = "rekognition"
    DEFAULT = VIREL


class ALPRSubFrameWork(str, Enum):
    OPENALPR = "openalpr"
    PLATE_RECOGNIZER = "plate_recognizer"
    REKOR = "rekor"
    DEFAULT = OPENALPR
    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name}>"

    def __str__(self):
        return f"{self.name}"


class ModelFrameWork(str, Enum):
    ULTRALYTICS = "ultralytics"
    OPENCV = "opencv"
    HTTP = "http"
    CORAL = "coral"
    TORCH = "torch"
    DEEPFACE = "deepface"
    ALPR = "alpr"
    FACE_RECOGNITION = "face_recognition"
    REKOGNITION = "rekognition"
    ORT = "ort"
    TRT = "trt"
    DEFAULT = OPENCV


class UltralyticsSubFrameWork(str, Enum):
    OBJECT = "object"
    SEGMENTATION = "segmentation"
    POSE = "pose"
    CLASSIFICATION = "classification"


class ModelProcessor(str, Enum):
    NONE = "none"
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    DEFAULT = CPU

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name}>"

    def __str__(self):
        return f"{self.name}"


class FaceRecognitionLibModelTypes(str, Enum):
    CNN = "cnn"
    HOG = "hog"
    DEFAULT = CNN


class ALPRAPIType(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"
    DEFAULT = LOCAL


class ALPRService(str, Enum):
    OPENALPR = "openalpr"
    PLATE_RECOGNIZER = "plate_recognizer"
    SCOUT = OPENALPR
    DEFAULT = OPENALPR
