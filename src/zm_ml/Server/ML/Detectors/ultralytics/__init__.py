from typing import Dict, List, AnyStr

from .yolo.ultra_base import UltralyticsYOLODetector

PRETRAINED_MODEL_NAMES: Dict[AnyStr, List[AnyStr]] = {
    "yolov5u": [
        "yolov5nu",
        "yolov5su",
        "yolov5mu",
        "yolov5lu",
        "yolov5xu",
        "yolov5n6u",
        "yolov5s6u",
        "yolov5m6u",
        "yolov5l6u",
        "yolov5x6u",
    ],
    "yolov8": [
        "yolov8n",
        "yolov8s",
        "yolov8m",
        "yolov8l",
        "yolov8x",
    ],
    "nas": [
        "yolo_nas_s",
        "yolo_nas_m",
        "yolo_nas_l",
        ],
    "yolov8-seg": [
        "yolov8n-seg.pt",
        "yolov8s-seg.pt",
        "yolov8m-seg.pt",
        "yolov8l-seg.pt",
        "yolov8x-seg.pt",
    ],
    "yolov8-pose": [
        "yolov8n-pose.pt",
        "yolov8s-pose.pt",
        "yolov8m-pose.pt",
        "yolov8l-pose.pt",
        "yolov8x-pose.pt",
        "yolov8x-pose-p6",
    ],
    "yolov8-cls": [
        "yolov8n-cls.pt",
        "yolov8s-cls.pt",
        "yolov8m-cls.pt",
        "yolov8l-cls.pt",
        "yolov8x-cls.pt",
    ],
}
