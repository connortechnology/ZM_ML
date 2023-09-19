import logging
from enum import Enum

from ...Server.Log import SERVER_LOGGER_NAME

logger = logging.getLogger(SERVER_LOGGER_NAME)


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
