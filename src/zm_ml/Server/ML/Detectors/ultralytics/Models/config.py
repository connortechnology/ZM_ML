from typing import Optional, List, Dict, Any, Union

from pydantic import field_validator, FieldValidationInfo

from zm_ml.Server.Models.config import *

class UltralyticsModelConfig(BaseModelConfig):
    """Configuration for the Detector"""

    model_name: str = Field(
        "yolo_nas_s",
        description="Name of the ultralytics model",
    )
    detection_options = None
    model_type = ModelType.OBJECT

    @field_validator("model_name", mode="before", always=True)
    def _validate_model_name(cls, v: Optional[str], info: FieldValidationInfo) -> str:
        assert info.config is not None
        # print(info.config.get('title'))
        # > Model
        # print(cls.model_fields[info.field_name].is_required())
        if not v:
            v = "yolo_nas_s"
        if isinstance(v, str):
            v = v.casefold()
            from ...ultralytics import PRETRAINED_MODEL_NAMES

            _type = info.config.get("sub_framework")
            model_names = []
            if _type == UltralyticsSubFrameWork.OBJECT:
                model_names.append(PRETRAINED_MODEL_NAMES.get("yolov8", []))
                model_names.append(PRETRAINED_MODEL_NAMES.get("yolov5u", []))
                model_names.append(PRETRAINED_MODEL_NAMES.get("nas", []))
            elif _type == UltralyticsSubFrameWork.SEGMENTATION:
                model_names.append(PRETRAINED_MODEL_NAMES.get("yolov8-seg", []))
            elif _type == UltralyticsSubFrameWork.POSE:
                model_names.append(PRETRAINED_MODEL_NAMES.get("yolov8-pose", []))
            elif _type == UltralyticsSubFrameWork.CLASSIFICATION:
                model_names.append(PRETRAINED_MODEL_NAMES.get("yolov8-cls", []))
            if v not in model_names:
                raise ValueError(f"Invalid model name: {v}, can only be one of {' ,'.join(PRETRAINED_MODEL_NAMES)}")
        return v

    def __repr__(self):
        return f"<{self.__class__.__name__} @ {self.model_config}>"

    def __str__(self):
        return f"API{{{self.model_config} ({self.model_type}:{self.model_processor})}}"
