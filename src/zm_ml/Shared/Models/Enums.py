from enum import Enum


class ModelType(str, Enum):
    OBJECT = "object"
    FACE = "face"
    ALPR = "alpr"
    DEFAULT = OBJECT

    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name} ({str(self.name).lower()} detection)"

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


class ModelFrameWork(str, Enum):
    OPENCV = "opencv"
    HTTP = "http"
    CORAL = "coral"
    TENSORFLOW = "tensorflow"
    TORCH = "torch"
    DEEPFACE = "deepface"
    ALPR = "alpr"
    FACE_RECOGNITION = "face_recognition"
    DEFAULT = OPENCV
    REKOGNITION = "rekognition"
    AWS = REKOGNITION


class ModelProcessor(str, Enum):
    NONE = "none"
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    DEFAULT = CPU


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
