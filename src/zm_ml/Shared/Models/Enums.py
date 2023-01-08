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


class ModelFrameWork(str, Enum):
    OPENCV = "opencv"
    YOLO = "yolo"
    CV_YOLO = YOLO
    CORAL = "coral"
    PYCORAL = CORAL
    EDGETPU = CORAL
    EDGE_TPU = CORAL
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    DEEPFACE = "deepface"
    ALPR = "alpr"
    FACE_RECOGNITION = "face_recognition"
    DEFAULT = CV_YOLO
    REKOGNITION = "rekognition"
    AWS = REKOGNITION


class ModelProcessor(str, Enum):
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
