from typing import Optional, List, Dict, Any, Union

from zm_ml.Server.Models.config import *


class UltralyticsConfig(BaseModelConfig):
    """Configuration for the Detector"""

    _model_names: Dict[str, List[str]] = Field(
        {
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
                "yolov8n.pt",
                "yolov8s.pt",
                "yolov8m.pt",
                "yolov8l.pt",
                "yolov8x.pt",
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

        }
    )
    is_nas: bool = Field(
        False, description="Is this a NAS model?", repr=False, init=False
    )

    model_name: str = Field(
        ...,
        description="Name of the ultralytics model",
    )
    model_type: ModelType = Field(
        ModelType.OBJECT, description="Type of the model", init=False
    )
    model_processor: ModelProcessor = Field(
        ModelProcessor.CPU, description="Processor for the model"
    )

    def __repr__(self):
        return f"<{self.__class__.__name__} @ {self.model_config}>"

    def __str__(self):
        return f"API{{{self.model_config} ({self.model_type}:{self.model_processor})}}"
