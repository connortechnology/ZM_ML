"""YOLO v5/8 support"""

try:
    import ultralytics
except ImportError:
    ultralytics = None


from ..file_locks import FileLock
from ....Shared.Models.Enums import ModelType, ModelProcessor
from zm_ml.Server.app import SERVER_LOGGER_NAME
from ....Shared.Models.config import DetectionResults, Result

class UltralyticsDetector