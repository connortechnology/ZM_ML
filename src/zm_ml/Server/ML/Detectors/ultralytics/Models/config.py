from zm_ml.Server.Models.config import *

from ......Shared.Models.config import DefaultEnabled


class UltralyticsModelConfig(BaseModelConfig):
    """Configuration for the Detector"""

    class PreTrained(DefaultEnabled):
        name: str = Field(
            "yolo_nas_s",
            description="Name of the ultralytics model",
        )

    pretrained: PreTrained = Field(default_factory=PreTrained)
    gpu_idx: Optional[int] = None

    _model_type: ModelType = ModelType.OBJECT

    @model_validator(mode="after")
    def _validate_pretrained_name(self):
        if self.pretrained:
            if self.pretrained.enabled is True:
                v = self.pretrained.name
                if not v:
                    v = "yolo_nas_s"
                if isinstance(v, str):
                    v = v.casefold()
                    from ...ultralytics import PRETRAINED_MODEL_NAMES

                    _type = self.sub_framework
                    model_names = []

                    if _type:
                        if isinstance(_type, str):
                            _type = UltralyticsSubFrameWork(_type)
                        if _type == UltralyticsSubFrameWork.OBJECT:
                            model_names.extend(PRETRAINED_MODEL_NAMES.get("yolov8", []))
                            model_names.extend(PRETRAINED_MODEL_NAMES.get("yolov5u", []))
                            model_names.extend(PRETRAINED_MODEL_NAMES.get("nas", []))
                        elif _type == UltralyticsSubFrameWork.SEGMENTATION:
                            model_names.extend(PRETRAINED_MODEL_NAMES.get("yolov8-seg", []))
                        elif _type == UltralyticsSubFrameWork.POSE:
                            model_names.extend(PRETRAINED_MODEL_NAMES.get("yolov8-pose", []))
                        elif _type == UltralyticsSubFrameWork.CLASSIFICATION:
                            model_names.extend(PRETRAINED_MODEL_NAMES.get("yolov8-cls", []))
                        if v not in model_names:
                            raise ValueError(
                                f"Invalid model name: {v}, can only be one of {model_names}")

                    else:
                        raise ValueError(f"sub_framework is not defined, cannot ascertain model name")

        return self


