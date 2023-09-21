import enum
import json
import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Union, Dict, List, Optional, Any, TYPE_CHECKING

import numpy as np
import yaml
from pydantic import (
    BaseModel,
    Field,
    FieldValidationInfo,
    field_validator,
    model_validator,
    IPvAnyAddress,
    PositiveInt,
    SecretStr,
    IPvAnyNetwork,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from .validators import validate_model_labels
from ..Log import SERVER_LOGGER_NAME
from ...Server.Models.DEFAULTS import *
from ...Shared.Models.Enums import (
    ModelType,
    ModelFrameWork,
    ModelProcessor,
    FaceRecognitionLibModelTypes,
    ALPRAPIType,
    ALPRService,
    HTTPSubFrameWork,
    OpenCVSubFrameWork,
    ALPRSubFrameWork,
    UltralyticsSubFrameWork,
)
from ...Shared.Models.config import Testing, LoggingSettings, DefaultEnabled
from ...Shared.Models.validators import (
    str2path,
    validate_file,
)

if TYPE_CHECKING:
    from ..ML.Detectors.ultralytics.Models.config import UltralyticsModelConfig
    from ..auth import UserDB

logger = logging.getLogger(SERVER_LOGGER_NAME)


class LockSetting(BaseModel):
    """Default file lock options"""

    max: int = Field(
        1, ge=1, le=100, description="Maximum number of parallel processes"
    )
    timeout: int = Field(
        30, ge=1, le=480, description="Timeout in seconds for acquiring a lock"
    )
    name: Optional[str] = Field(None, description="Name of the lock file")


class LockSettings(DefaultEnabled):
    dir: Path = Field(
        None,
        description="Directory for the lock files",
    )
    gpu: LockSetting = Field(
        default_factory=LockSetting, description="GPU Lock Settings"
    )
    cpu: LockSetting = Field(
        default_factory=LockSetting, description="CPU Lock Settings"
    )
    tpu: LockSetting = Field(
        default_factory=LockSetting, description="TPU Lock Settings"
    )

    @field_validator("gpu", "cpu", "tpu", mode="before")
    @classmethod
    def set_lock_name(cls, v, info: FieldValidationInfo):
        if v:
            assert isinstance(v, (LockSetting, Dict)), f"Invalid type: {type(v)}"
            if isinstance(v, Dict):
                v = LockSetting(**v)
            v.name = f"zm-mlapi_{info.field_name}"
        return v

    def get(self, device: str) -> LockSetting:
        if device:
            device = device.casefold()
            if device == "gpu":
                return self.gpu
            elif device == "cpu":
                return self.cpu
            elif device == "tpu":
                return self.tpu
            else:
                raise ValueError(f"Invalid device type: {device}")
        else:
            raise ValueError("Device type must be specified")

    @field_validator("dir", mode="before")
    @classmethod
    def validate_lock_dir(cls, v):
        if not v:
            v = f"{tempfile.gettempdir()}/zm_mlapi/locks"
        temp_dir = Path(v)
        if not temp_dir.exists():
            logger.debug(f"Creating lock directory: {temp_dir}")
            temp_dir.mkdir(parents=True)
        return v


class ServerSettings(BaseModel):
    class AuthSettings(DefaultEnabled):
        db_file: Path = Field(..., description="TinyDB user DB file")
        expire_after: int = Field(60, ge=1, description="JWT Expiration time in minutes")
        sign_key: SecretStr = Field("CHANGE ME!!!!", description="JWT Sign Key")
        algorithm: str = Field("HS256", description="JWT Algorithm")


    address: Union[IPvAnyAddress] = Field(
        "0.0.0.0", description="Server listen address"
    )

    port: PositiveInt = Field(5000, description="Server listen port")
    reload: bool = Field(
        default=False, description="Uvicorn reload - For development only"
    )
    debug: bool = Field(
        default=False, description="Uvicorn debug mode - For development only"
    )
    auth: AuthSettings = Field(default_factory=AuthSettings, description="JWT Settings")


class DetectionResult(BaseModel):
    success: bool = False
    type: ModelType = None
    processor: ModelProcessor = None
    _model_name: str = Field(None, alias="model_name")

    label: List[str] = None
    confidence: List[Union[float, int]] = None
    bounding_box: List[List[Union[float, int]]] = None


class BaseModelOptions(BaseModel):
    confidence: Optional[float] = Field(
        0.2, ge=0.0, le=1.0, descritpiton="Confidence Threshold"
    )


class ORTModelOptions(BaseModelOptions):
    nms: Optional[float] = Field(
        0.4, ge=0.0, le=1.0, description="Non-Maximum Suppression Threshold (IoU)"
    )


class TRTModelOptions(BaseModelOptions):
    nms: Optional[float] = Field(
        0.4, ge=0.0, le=1.0, description="Non-Maximum Suppression Threshold (IoU)"
    )


class UltralyticsModelOptions(BaseModelOptions):
    nms: Optional[float] = Field(
        0.4, ge=0.0, le=1.0, description="Non-Maximum Suppression Threshold (IoU)"
    )


class CV2YOLOModelOptions(BaseModelOptions):
    nms: Optional[float] = Field(
        0.4, ge=0.0, le=1.0, description="Non-Maximum Suppression Threshold"
    )


class TPUModelOptions(BaseModelOptions, extra="allow"):
    class NMSOptions(BaseModel):
        enabled: bool = Field(True, description="Enable Non-Maximum Suppression")
        threshold: Optional[float] = Field(
            0.4, ge=0.0, le=1.0, description="Non-Maximum Suppression Threshold"
        )

    nms: NMSOptions = Field(default_factory=NMSOptions, description="NMS Options")


class FaceRecognitionLibModelDetectionOptions(BaseModelOptions):
    # face_recognition lib config Options
    model: Optional[FaceRecognitionLibModelTypes] = Field(
        FaceRecognitionLibModelTypes.CNN,
        examples=["hog", "cnn"],
        description="Face Detection Model to use. 'cnn' is more accurate but slower on CPUs. "
        "'hog' is faster but less accurate",
    )
    upsample_times: Optional[int] = Field(
        1,
        ge=0,
        description="How many times to upsample the image looking for faces. "
        "Higher numbers find smaller faces but take longer.",
    )
    num_jitters: Optional[int] = Field(
        1,
        ge=0,
        description="How many times to re-sample the face when calculating encoding. "
        "Higher is more accurate, but slower (i.e. 100 is 100x slower)",
    )

    max_size: Optional[int] = Field(
        600,
        ge=100,
        description="Maximum size (Width) of image to load into memory for "
        "face detection (image will be scaled)",
    )
    recognition_threshold: Optional[float] = Field(
        0.6,
        ge=0.0,
        le=1.0,
        description="Recognition distance threshold for face recognition",
    )


class FaceRecognitionLibModelTrainingOptions(BaseModelOptions):
    model: Optional[FaceRecognitionLibModelTypes] = Field(
        FaceRecognitionLibModelTypes.CNN,
        examples=["hog", "cnn"],
        description="Face Detection Model to use. 'cnn' is more accurate but slower on CPUs. "
        "'hog' is faster but less accurate",
    )
    upsample_times: Optional[int] = Field(
        1,
        ge=0,
        description="How many times to upsample the image while detecting faces. "
        "Higher numbers find smaller faces but take longer.",
    )
    num_jitters: Optional[int] = Field(
        1,
        ge=0,
        description="How many times to re-sample the face when calculating encoding. "
        "Higher is more accurate, but slower (i.e. 100 is 100x slower)",
    )

    max_size: Optional[int] = Field(
        600,
        ge=100,
        description="Maximum size (Width) of image to load into memory for "
        "face detection (image will be scaled)",
    )
    dir: Optional[Path] = Field(
        None,
        description="Directory to load training images from. If None, the default directory will be used",
    )


class ALPRModelOptions(BaseModelOptions):
    max_size: Optional[int] = Field(
        600,
        ge=1,
        description="Maximum size (Width) of image to load into memory",
    )


class OpenALPRLocalModelOptions(ALPRModelOptions):
    binary_path: Optional[str] = Field(
        "alpr", description="OpenALPR binary name or absolute path"
    )
    binary_params: Optional[str] = Field(
        "-d",
        description="OpenALPR binary parameters (-j is ALWAYS passed)",
        example="-p ca -c US",
    )


class OpenALPRCloudModelOptions(ALPRModelOptions):
    # For an explanation of params, see http://doc.openalpr.com/api/?api=cloudapi
    recognize_vehicle: Optional[bool] = Field(
        True,
        description="If True, will attempt to recognize the vehicle type (ie: Ford Mustang)",
    )
    country: Optional[str] = Field(
        "us",
        description="Country of license plate to recognize.",
    )
    state: Optional[str] = Field(
        None,
        description="State of license plate to recognize.",
    )


class PlateRecognizerModelOptions(BaseModelOptions):
    stats: Optional[bool] = Field(
        False,
        description="Return stats about the plate recognition request",
    )
    payload: Optional[Dict] = Field(
        default_factory=dict,
        description="Override the payload sent to the Plate Recognizer API, must be a JSON serializable string",
    )
    config: Optional[Dict] = Field(
        None,
        description="Override the config sent to the Plate Recognizer API, must be a JSON serializable string",
    )
    # If you want to specify regions. See http://docs.platerecognizer.com/#regions-supported
    regions: List[str] = Field(
        None,
        example=["us", "cn", "kr", "ca"],
        description="List of regions to search for plates in. If not specified, "
        "all regions are searched. See http://docs.platerecognizer.com/#regions-supported",
    )
    # minimal confidence for actually detecting a plate (the presence of a license plate, not the actual plate text)
    min_dscore: float = Field(
        0.1,
        ge=0,
        le=1.0,
        description="Minimal confidence for detecting a license plate in the image",
    )
    # minimal confidence for the translated text (plate number)
    min_score: float = Field(
        0.5,
        ge=0,
        le=1.0,
        description="Minimal confidence for the translated text from the plate",
    )
    max_size: int = Field(
        1600,
        ge=1,
        description="Maximum size (Width) of image",
    )

    @field_validator("config", "payload")
    @classmethod
    def check_json_serializable(cls, v):
        try:
            json.dumps(v)
        except TypeError:
            raise ValueError("Must be JSON serializable, check formatting...")
        return v


class TorchModelOptions(BaseModelOptions):
    nms: Optional[float] = Field(
        0.3, ge=0.0, le=1.0, description="Non-Maximum Suppression Threshold"
    )


class BaseModelConfig(BaseModel):
    """
    Base Model Config

    This is the base model config that all models inherit from.

    :param id: Unique ID of the model
    :param name: model name
    :param enabled: model enabled
    :param description: model description
    :param framework: model framework
    :param model_type: model type (object, face, alpr)
    :param processor: Processor to use for model
    :param sub_framework: sub-framework to use for model
    :param detection_options: Model options (if any)
    """

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4, description="Unique ID of the model", init=False
    )
    name: str = Field(..., description="model name")
    enabled: bool = Field(True, description="model enabled")
    description: str = Field(None, description="model description")
    framework: ModelFrameWork = Field(
        default=ModelFrameWork.OPENCV, description="model framework"
    )
    type_of: ModelType = Field(
        ModelType.OBJECT,
        description="model type (object, face, alpr)",
        alias="model_type",
    )
    processor: Optional[ModelProcessor] = Field(
        ModelProcessor.CPU, description="Processor to use for model"
    )

    # todo: add validator that detects framework and sets a default sub framework if it is None.
    sub_framework: Optional[
        Union[
            OpenCVSubFrameWork,
            HTTPSubFrameWork,
            ALPRSubFrameWork,
            UltralyticsSubFrameWork,
        ]
    ] = Field(OpenCVSubFrameWork.DARKNET, description="sub-framework to use for model")

    detection_options: Union[
        BaseModelOptions,
        FaceRecognitionLibModelDetectionOptions,
        OpenALPRLocalModelOptions,
        OpenALPRCloudModelOptions,
        PlateRecognizerModelOptions,
        ALPRModelOptions,
        TPUModelOptions,
        TorchModelOptions,
        UltralyticsModelOptions,
        TRTModelOptions,
        ORTModelOptions,
        None,
    ] = Field(
        default_factory=BaseModelOptions,
        description="Default Configuration for the model",
    )

    @field_validator("name")
    @classmethod
    def check_name(cls, v):
        v = str(v).strip().casefold()
        return v


class TPUModelConfig(BaseModelConfig):
    input: Optional[Path] = Field(None, description="model file/dir path (Optional)")
    config: Optional[Path] = Field(
        None, description="model config file path (Optional)"
    )
    classes: Optional[Path] = Field(
        None, description="model classes file path (Optional)"
    )
    height: Optional[int] = Field(
        416, ge=1, description="Model input height (resized for model)"
    )
    width: Optional[int] = Field(
        416, ge=1, description="Model input width (resized for model)"
    )
    square: Optional[bool] = Field(
        False, description="Zero pad the image to be a square; 1920x1080 = 1920x1920"
    )

    labels: List[str] = Field(
        default=None,
        description="model labels parsed into a list of strings",
        repr=False,
        exclude=True,
    )

    _validate_labels = field_validator("labels")(validate_model_labels)

    @field_validator("config", "input", "classes", mode="before")
    @classmethod
    def str_to_path(cls, v, info: FieldValidationInfo) -> Optional[Path]:
        msg = f"{info.field_name} must be a path or a string of a path"
        model_name = info.data.get("name", "Unknown Model")
        model_input: Optional[Path] = info.data.get("input")
        lp = f"Model Name: {model_name} ->"

        if v is None:
            return v
        elif not isinstance(v, (Path, str)):
            raise ValueError(msg)
        elif isinstance(v, str):
            v = Path(v)
        if info.field_name == "config":
            if model_input and model_input.suffix == ".weights":
                msg = f"'{info.field_name}' is required when 'input' is a DarkNet .weights file"
        return v


class ORTModelConfig(BaseModelConfig):
    input: Path = Field(None, description="model file/dir path (Optional)")
    classes: Path = Field(default=None, description="model labels file path (Optional)")
    height: Optional[int] = Field(
        416, ge=1, description="Model input height (resized for model)"
    )
    width: Optional[int] = Field(
        416, ge=1, description="Model input width (resized for model)"
    )
    square: Optional[bool] = Field(
        False, description="Zero pad the image to be a square"
    )

    labels: List[str] = Field(
        default=None,
        description="model labels parsed into a list of strings",
        repr=False,
        exclude=True,
    )
    extra: Optional[str] = None

    @model_validator(mode="after")
    def _validate_model(self):
        model_name = self.name
        model_input: Optional[Path] = self.input
        labels = self.labels
        self.labels = validate_model_labels(
            labels, info=None, model_name=model_name, labels_file=self.classes
        )
        v = model_input
        assert isinstance(v, (Path, str)), f"Invalid type: {type(v)} for {v}"
        if isinstance(v, str):
            v = Path(v)

        return self


class TRTModelConfig(BaseModelConfig):
    input: Path = Field(None, description="model file/dir path (Optional)")
    classes: Path = Field(default=None, description="model labels file path (Optional)")
    height: Optional[int] = Field(
        416, ge=1, description="Model input height (resized for model)"
    )
    width: Optional[int] = Field(
        416, ge=1, description="Model input width (resized for model)"
    )
    square: Optional[bool] = Field(
        False, description="Zero pad the image to be a square"
    )

    gpu_idx: Optional[int] = 0
    lib_path: Optional[Path] = None

    labels: List[str] = Field(
        default=None,
        description="model labels parsed into a list of strings",
        repr=False,
        exclude=True,
    )

    @model_validator(mode="after")
    def _validate_model(self):
        model_name = self.name
        model_input: Optional[Path] = self.input
        labels = self.labels
        self.labels = validate_model_labels(
            labels, info=None, model_name=model_name, labels_file=self.classes
        )
        v = model_input
        assert isinstance(v, (Path, str)), f"Invalid type: {type(v)} for {v}"
        if isinstance(v, str):
            v = Path(v)

        return self


class CV2YOLOModelConfig(BaseModelConfig):
    class ONNXType(str, enum.Enum):
        v5 = "yolov5"
        v8 = "yolov8"
        nas = "yolo-nas"

    input: Path = Field(None, description="model file/dir path (Optional)")
    classes: Path = Field(default=None, description="model labels file path (Optional)")
    config: Optional[Path] = Field(
        default=None, description="model config file path (Optional)"
    )
    height: Optional[int] = Field(
        416, ge=1, description="Model input height (resized for model)"
    )
    width: Optional[int] = Field(
        416, ge=1, description="Model input width (resized for model)"
    )
    square: Optional[bool] = Field(
        False, description="Zero pad the image to be a square"
    )
    cv2_cuda_fp_16: Optional[bool] = Field(
        False, description="model uses Floating Point 16 Backend (EXPERIMENTAL!)"
    )

    onnx_type: Optional[ONNXType] = Field(None, description="ONNX Model Type")

    labels: List[str] = Field(
        default=None,
        description="model labels parsed into a list of strings",
        repr=False,
        exclude=True,
    )

    @model_validator(mode="after")
    def _validate_model(self):
        model_name = self.name
        model_input: Optional[Path] = self.input
        model_config: Optional[Path] = self.config or None
        labels = self.labels
        self.labels = validate_model_labels(
            labels, info=None, model_name=model_name, labels_file=self.classes
        )

        for v in (model_input, model_config):
            if not v:
                continue
            assert isinstance(v, (Path, str)), f"Invalid type: {type(v)} for {v}"
            if v == model_config:
                if v is None:
                    if model_input and model_input.suffix == ".weights":
                        raise ValueError(
                            f"'{v.__class__.name}' is required when 'input' is a DarkNet .weights file"
                        )
                    return v
            elif isinstance(v, str):
                v = Path(v)

        return self


class FaceRecognitionLibModelConfig(BaseModelConfig):
    """Config cant be changed after loading - Options can be changed"""

    class UnknownFaceSettings(BaseModel):
        enabled: Optional[bool] = Field(
            True,
            description="If True, unknown faces will be saved to disk for possible training",
        )
        label_as: Optional[str] = Field(
            "Unknown",
            description="Label to use for unknown faces. If 'enabled' is False, this is ignored",
        )
        dir: Optional[Path] = Field(
            None,
            description="Directory to save unknown faces to. If 'enabled' is False, this is ignored",
        )
        leeway_pixels: int = Field(
            0,
            description="Unknown faces leeway pixels, used when cropping the image to capture a face",
        )

    detection_options: FaceRecognitionLibModelDetectionOptions = Field(
        default_factory=FaceRecognitionLibModelDetectionOptions,
        description="Default Configuration for the model",
    )

    training_options: Optional[FaceRecognitionLibModelTrainingOptions] = Field(
        default_factory=FaceRecognitionLibModelTrainingOptions,
        description="Configuration for training faces",
    )

    unknown_faces: UnknownFaceSettings = Field(
        default_factory=UnknownFaceSettings,
        description="Settings for handling unknown faces",
    )


class ALPRModelConfig(BaseModelConfig):
    api_type: ALPRAPIType = Field(ALPRAPIType.LOCAL, description="ALPR Service Type")
    service: ALPRService = Field(ALPRService.OPENALPR, description="ALPR Service to use")
    api_key: str = Field(None, description="ALPR API Key (Cloud/Local)")
    api_url: str = Field(None, description="ALPR API URL (Cloud/Local)")


class CV2HOGModelConfig(BaseModelConfig):
    stride: str = None
    padding: str = None
    scale: float = None
    mean_shift: bool = False


class RekognitionModelConfig(BaseModelConfig):
    aws_access_key_id: str = None
    aws_secret_access_key: str = None
    # aws_session_token: str = None
    # aws_profile: str = None
    region_name: str = None


class DeepFaceModelConfig(BaseModelConfig):
    pass


class VirelAIModelConfig(BaseModelConfig):
    pass


from ...Shared.Models.config import DefaultEnabled


class TorchModelConfig(BaseModelConfig):
    class PreTrained(DefaultEnabled):
        name: Optional[str] = Field(
            "default",
            pattern=r"(accurate|fast|default|balanced|high_performance|low_performance)",
        )

        @field_validator("name", mode="before", check_fields=False)
        @classmethod
        def _validate_model_name(
            cls, v: Optional[str], info: FieldValidationInfo
        ) -> str:
            if v is None:
                v = "default"
            return v

    input: Optional[Path] = None
    classes: Optional[Path] = None
    num_classes: Optional[int] = None

    gpu_idx: Optional[int] = None

    pretrained: Optional[PreTrained] = Field(default_factory=PreTrained)

    # this is not to be configured by the user. It is parsed classes from the labels file (or default if no file).
    labels: List[str] = Field(
        default=None,
        description="model labels parsed into a list of strings",
        repr=False,
        exclude=True,
    )

    _validate_labels = field_validator("labels")(validate_model_labels)
    _validate = field_validator("input", "classes", mode="before")(str2path)


class ColorDetectSettings(BaseModel):
    enabled: bool = Field(True, description="Enable Color Detection")
    n_most_common: int = Field(
        4, ge=1, description="Number of dominant colors to detect"
    )
    labels: List[str] = Field(
        None, description="List of labels to run color detection on"
    )


def _replace_vars(search_str: str, var_pool: Dict) -> Dict:
    """Replace variables in a string.


    Args:
        search_str (str): String to search for variables '${VAR_NAME}'.
        var_pool (Dict): Dictionary of variables used to replace.

    """
    import re

    if var_list := re.findall(r"\$\{(\w+)\}", search_str):
        logger.debug(
            f"Found the following substitution variables: {list(set(var_list))}"
        )
        # substitute variables
        _known_vars = []
        _unknown_vars = []
        __seen_vars = []
        for var in var_list:
            if var in var_pool:
                # logger.debug(
                #     f"substitution variable '{var}' IS IN THE sub POOL! VALUE: "
                #     f"{var_pool[var]} [{type(var_pool[var])}]"
                # )
                value = var_pool[var]
                if value is None:
                    value = ""
                elif value is True:
                    value = "yes"
                elif value is False:
                    value = "no"
                search_str = search_str.replace(f"${{{var}}}", value)
                if var not in __seen_vars:
                    _known_vars.append(var)

            else:
                if var not in __seen_vars:
                    _unknown_vars.append(var)
            __seen_vars.append(var)

        if _unknown_vars:
            logger.warning(
                f"The following variables have no configured substitution value: {_unknown_vars}"
            )
        if _known_vars:
            logger.debug(
                f"The following variables have been substituted: {_known_vars}"
            )
    else:
        logger.debug(f"No substitution variables found.")
    return yaml.safe_load(search_str)


class SystemSettings(BaseModel):
    models_dir: Optional[Path] = Field(Path(DEF_SRV_SYS_MODELDIR), alias="model_dir")
    image_dir: Optional[Path] = Field(Path(DEF_SRV_SYS_IMAGEDIR))
    config_path: Optional[Path] = Field(Path(DEF_SRV_SYS_CONFDIR))
    variable_data_path: Optional[Path] = Field(DEF_SRV_SYS_DATADIR)
    tmp_path: Optional[Path] = Field(Path(DEF_SRV_SYS_TMPDIR))
    thread_workers: Optional[int] = Field(DEF_SRV_SYS_THREAD_WORKERS)


class UvicornSettings(BaseModel):
    debug: Optional[bool] = Field(False, description="Uvicorn debug mode")
    proxy_headers: Optional[bool] = False
    forwarded_allow_ips: Optional[List[Union[IPvAnyAddress, IPvAnyNetwork]]] = Field(
        default_factory=list
    )
    grab_cloudflare_ips: Optional[bool] = Field(False)


class Settings(BaseModel, arbitrary_types_allowed=True):
    testing: Testing = Field(default_factory=Testing)
    substitutions: Dict[str, str] = Field(default_factory=dict)
    uvicorn: UvicornSettings = Field(default_factory=UvicornSettings)
    system: SystemSettings = Field(default_factory=SystemSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    locks: LockSettings = Field(
        default_factory=LockSettings, description="Lock Settings", repr=False
    )
    models: List = Field(..., description="Models configuration", exclude=True)

    available_models: List[BaseModelConfig] = Field(
        None, description="Available models"
    )

    def get_lock_settings(self):
        return self.locks

    @model_validator(mode="after")
    def _validate_model_after(self) -> "Settings":
        v = []
        models = self.models
        if models:
            disabled_models: list = []
            for model in models:
                if not model:
                    logger.warning(
                        f"Model is empty, this usually means there is a formatting error "
                        f"(or an empty model; just a '-') in your config file! Skipping"
                    )
                    continue
                final_model_config = None
                if model.get("enabled", True) is True:
                    _model: Union[
                        RekognitionModelConfig,
                        TRTModelConfig,
                        ORTModelConfig,
                        TPUModelConfig,
                        VirelAIModelConfig,
                        FaceRecognitionLibModelConfig,
                        ALPRModelConfig,
                        CV2YOLOModelConfig,
                        "UltralyticsModelConfig",
                        TorchModelConfig,
                        None,
                    ] = None
                    _framework = model.get("framework", ModelFrameWork.OPENCV)
                    _sub_fw = model.get("sub_framework", OpenCVSubFrameWork.DARKNET)
                    _options = model.get("detection_options", {})
                    _type = model.get("model_type", ModelType.OBJECT)
                    logger.debug(
                        f"Settings:: validate available models => {_framework = } -- {_sub_fw = } -- {_type = } -- {_options = } filename: {model.get('input', 'No input file defined!')}"
                    )
                    if _framework == ModelFrameWork.HTTP:
                        if _sub_fw == HTTPSubFrameWork.REKOGNITION:
                            model["model_type"] = ModelType.OBJECT
                            model["processor"] = ModelProcessor.NONE
                            final_model_config = RekognitionModelConfig(**model)
                        elif _sub_fw == HTTPSubFrameWork.VIREL:
                            model["model_type"] = ModelType.OBJECT
                            model["processor"] = ModelProcessor.NONE
                            final_model_config = VirelAIModelConfig(**model)
                    elif _framework == ModelFrameWork.FACE_RECOGNITION:
                        model["model_type"] = ModelType.FACE
                        model[
                            "detection_options"
                        ] = FaceRecognitionLibModelDetectionOptions(**_options)
                        final_model_config = FaceRecognitionLibModelConfig(**model)

                    elif _framework == ModelFrameWork.ALPR:
                        model["model_type"] = ModelType.ALPR
                        api_type = model.get("api_type", "local")
                        if _sub_fw == ALPRSubFrameWork.OPENALPR:
                            config = ALPRModelConfig(**model)
                            if api_type == ALPRAPIType.LOCAL:
                                config.detection_options = OpenALPRLocalModelOptions(
                                    **_options
                                )
                            elif api_type == ALPRAPIType.CLOUD:
                                config.processor = ModelProcessor.NONE
                                config.detection_options = OpenALPRCloudModelOptions(
                                    **_options
                                )
                        elif _sub_fw == ALPRSubFrameWork.PLATE_RECOGNIZER:
                            if api_type == ALPRAPIType.CLOUD:
                                config.processor = ModelProcessor.NONE
                                config.detection_options = PlateRecognizerModelOptions(
                                    **_options
                                )
                            elif api_type == ALPRAPIType.LOCAL:
                                raise NotImplementedError(
                                    "Plate Recognizer Local API not implemented"
                                )

                        # todo: init models and check if success?
                        final_model_config = config
                    elif _framework == ModelFrameWork.ULTRALYTICS:
                        from ..ML.Detectors.ultralytics.Models.config import (
                            UltralyticsModelConfig,
                        )

                        if _sub_fw in [
                            UltralyticsSubFrameWork.POSE,
                            UltralyticsSubFrameWork.SEGMENTATION,
                            UltralyticsSubFrameWork.CLASSIFICATION,
                        ]:
                            logger.warning(f"Not implemented: {_sub_fw}")

                        elif _sub_fw == UltralyticsSubFrameWork.OBJECT:
                            final_model_config = UltralyticsModelConfig(**model)
                            final_model_config.detection_options = (
                                UltralyticsModelOptions(**_options)
                            )

                    elif _framework == ModelFrameWork.ORT:
                        final_model_config = ORTModelConfig(**model)
                        final_model_config.detection_options = ORTModelOptions(
                            **_options
                        )

                    elif _framework == ModelFrameWork.OPENCV:
                        config = None
                        not_impl = [
                            OpenCVSubFrameWork.TENSORFLOW,
                            OpenCVSubFrameWork.TORCH,
                            OpenCVSubFrameWork.CAFFE,
                            OpenCVSubFrameWork.VINO,
                        ]
                        if (
                            _sub_fw == OpenCVSubFrameWork.DARKNET
                            or _sub_fw == OpenCVSubFrameWork.ONNX
                        ):
                            config = CV2YOLOModelConfig(**model)
                            config.detection_options = CV2YOLOModelOptions(**_options)

                        elif _sub_fw in not_impl:
                            raise NotImplementedError(f"{_sub_fw} not implemented")
                        else:
                            raise ValueError(f"Unknown OpenCV sub-framework: {_sub_fw}")
                        if config:
                            final_model_config = config
                    elif _framework == ModelFrameWork.CORAL:
                        config = TPUModelConfig(**model)
                        config.detection_options = TPUModelOptions(**_options)
                        final_model_config = config
                    elif _framework == ModelFrameWork.TRT:
                        config = TRTModelConfig(**model)
                        config.detection_options = TRTModelOptions(**_options)
                        final_model_config = config
                    elif _framework == ModelFrameWork.TORCH:
                        config = TorchModelConfig(**model)
                        config.detection_options = TorchModelOptions(**_options)
                        final_model_config = config
                    else:
                        raise NotImplementedError(
                            f"Framework {_framework} not implemented"
                        )

                    # load model here
                    if final_model_config:
                        logger.debug(
                            f"Settings:: Adding to available models: {final_model_config.name}"
                        )
                        v.append(final_model_config)
                else:
                    disabled_models.append(model.get("name"))
            if disabled_models:
                logger.debug(f"Skipped these disabled models: {disabled_models}")
        if v:
            self.available_models = v
        return self


def parse_client_config_file(cfg_file: Path) -> Optional[Settings]:
    """Parse the YAML configuration file."""

    cfg: Dict = {}
    _start = time.perf_counter()
    raw_config = cfg_file.read_text()

    try:
        cfg = yaml.safe_load(raw_config)
    except yaml.YAMLError:
        logger.error(f"Error parsing the YAML configuration file!")
        raise
    except PermissionError:
        logger.error(f"Error reading the YAML configuration file!")
        raise

    substitutions = cfg.get("substitutions", {})
    testing = cfg.get("testing", {})
    testing = Testing(**testing)
    if testing.enabled:
        logger.info(f"|----- TESTING IS ENABLED! -----|")
        if testing.substitutions:
            logger.info(f"Overriding config:substitutions WITH testing:substitutions")
            substitutions = testing.substitutions

    logger.debug(f"Replacing ${{VARS}} in config:substitutions")
    substitutions = _replace_vars(search_str=str(substitutions), var_pool=substitutions)
    # logger.debug(f"AFTER FIRST SUBSTITUTION: {substitutions = }")
    substitutions = _replace_vars(search_str=str(substitutions), var_pool=substitutions)
    # logger.debug(f"AFTER SECOND SUBSTITUTION: {substitutions = }")
    if inc_file := substitutions.get("IncludeFile"):
        inc_file = Path(inc_file)
        logger.debug(f"PARSING IncludeFile: {inc_file.as_posix()}")
        if inc_file.is_file():
            inc_vars = yaml.safe_load(inc_file.read_text())
            if "server" in inc_vars:
                inc_vars = inc_vars["server"]
                # logger.debug(
                #     f"Loaded {len(inc_vars)} substitution from IncludeFile {inc_file} => {inc_vars}"
                # )
                # check for duplicates
                for k in inc_vars:
                    if k in substitutions:
                        logger.warning(
                            f"Duplicate substitution variable '{k}' in IncludeFile {inc_file} - "
                            f"IncludeFile overrides config file"
                        )

                substitutions.update(inc_vars)
            else:
                logger.warning(
                    f"IncludeFile [{inc_file}] does not have a 'client' section - skipping"
                )
        else:
            logger.warning(f"IncludeFile {inc_file} is not a file!")
    logger.debug(f"Replacing ${{VARS}} in config")
    cfg = _replace_vars(raw_config, substitutions)
    logger.debug(
        f"perf:: Config file loaded in {time.perf_counter() - _start:.5f} seconds"
    )
    _cfg = Settings(**cfg)
    # logger.debug(f"cfg file parsed into a class = {_cfg}")
    return _cfg


class APIDetector:
    """ML detector API class.
    Specify a processor type and then load the model into processor memory. run an inference on the processor
    """

    id: uuid.UUID = Field(None, description="Model ID")
    _config: Union[
        BaseModelConfig,
        CV2YOLOModelConfig,
        ALPRModelConfig,
        FaceRecognitionLibModelConfig,
        DeepFaceModelConfig,
        TPUModelConfig,
        TorchModelConfig,
        "UltralyticsModelConfig",
        "ORTModelConfig",
        "TRTModelConfig",
    ]
    _options: Union[
        BaseModelOptions,
        OpenALPRLocalModelOptions,
        OpenALPRCloudModelOptions,
        PlateRecognizerModelOptions,
        ALPRModelOptions,
        FaceRecognitionLibModelDetectionOptions,
        CV2YOLOModelOptions,
        TorchModelOptions,
        UltralyticsModelOptions,
        "ORTModelOptions",
        "TRTModelOptions",
    ]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.config})"

    def __get__(self, instance, owner):
        return self

    def __init__(
        self,
        model_config: Union[
                BaseModelConfig,
                CV2YOLOModelConfig,
                ALPRModelConfig,
                FaceRecognitionLibModelConfig,
                DeepFaceModelConfig,
                TPUModelConfig,
                TorchModelConfig,
                "UltralyticsModelConfig",
                "ORTModelConfig",
                "TRTModelConfig",
        ],
    ):
        from ..ML.Detectors.opencv.cv_yolo import CV2YOLODetector
        from ..ML.Detectors.coral_edgetpu import TpuDetector
        from ..ML.Detectors.face_recognition import FaceRecognitionLibDetector
        from ..ML.Detectors.alpr import PlateRecognizer, OpenAlprCmdLine, OpenAlprCloud
        from ..ML.Detectors.virelai import VirelAI
        from ..ML.Detectors.aws_rekognition import AWSRekognition
        from ..ML.Detectors.torch.torch_base import TorchDetector
        from ..ML.Detectors.ultralytics.yolo.ultra_base import UltralyticsYOLODetector
        from ..ML.Detectors.onnx_runtime import ORTDetector


        self.config = model_config
        self.id = self.config.id
        self.options = model_config.detection_options
        self.model: Optional[
            Union[
                TorchDetector,
                TpuDetector,
                CV2YOLODetector,
                FaceRecognitionLibDetector,
                PlateRecognizer,
                OpenAlprCmdLine,
                OpenAlprCloud,
                VirelAI,
                AWSRekognition,
                TorchDetector,
                UltralyticsYOLODetector,
                ORTDetector,
                "TensorRtDetector",
            ]
        ] = None
        self._load_model()

    @property
    def options(
        self,
    ) -> Union[
        BaseModelOptions,
        OpenALPRLocalModelOptions,
        OpenALPRCloudModelOptions,
        PlateRecognizerModelOptions,
        ALPRModelOptions,
        FaceRecognitionLibModelDetectionOptions,
        CV2YOLOModelOptions,
        TorchModelOptions,
        UltralyticsModelOptions,
        "ORTModelOptions",
        "TRTModelOptions",
    ]:
        return self._options

    @options.setter
    def options(
        self,
        options: Union[
            BaseModelOptions,
            OpenALPRLocalModelOptions,
            OpenALPRCloudModelOptions,
            PlateRecognizerModelOptions,
            ALPRModelOptions,
            FaceRecognitionLibModelDetectionOptions,
            CV2YOLOModelOptions,
            TorchModelOptions,
            UltralyticsModelOptions,
            "ORTModelOptions",
            "TRTModelOptions",
        ],
    ):
        self._options = options

    def _load_model(
        self,
        config: Optional[
            Union[
                BaseModelConfig,
                CV2YOLOModelConfig,
                ALPRModelConfig,
                FaceRecognitionLibModelConfig,
                DeepFaceModelConfig,
                TPUModelConfig,
                TorchModelConfig,
                "UltralyticsModelConfig",
                "ORTModelConfig",
                "TRTModelConfig",
            ]
        ] = None,
    ):
        """Load the model"""

        from ..ML.Detectors.face_recognition import FaceRecognitionLibDetector
        from ..ML.Detectors.alpr import PlateRecognizer, OpenAlprCmdLine, OpenAlprCloud
        from ..ML.Detectors.virelai import VirelAI
        from ..ML.Detectors.aws_rekognition import AWSRekognition

        self.model = None
        _proc_available: Optional[bool] = False
        if config:
            self.config = config
        if not self.config.processor:
            self.config.processor = ModelProcessor.CPU

        _proc_available = self.is_processor_available()
        if _proc_available is False:
            _failed_proc = self.config.processor.value

            if self.config.framework == ModelFrameWork.CORAL:
                self.config.processor = ModelProcessor.TPU
            elif self.config.framework == ModelFrameWork.HTTP:
                self.config.processor = ModelProcessor.NONE
            else:
                self.config.processor = ModelProcessor.CPU
            logger.warning(
                f"{_failed_proc} is not available to the {self.config.framework} framework! "
                f"Switching to {self.config.processor}"
            )

        elif _proc_available is None:
            from ..app import get_global_config

            for _model in get_global_config().available_models:
                if _model.id == self.id:
                    get_global_config().available_models.remove(_model)
                    break
            raise ImportError(
                f"Library missing, cannot create detector for {self.config.name}"
            )

        try:
            if self.config.framework == ModelFrameWork.OPENCV:
                if self.config.sub_framework == OpenCVSubFrameWork.DARKNET:
                    from ..ML.Detectors.opencv.cv_yolo import CV2YOLODetector

                    self.model = CV2YOLODetector(self.config)

                elif self.config.sub_framework == OpenCVSubFrameWork.ONNX:
                    from ..ML.Detectors.opencv.onnx import CV2ONNXDetector

                    self.model = CV2ONNXDetector(self.config)
            elif self.config.framework == ModelFrameWork.TRT:
                from ..ML.Detectors.tensorrt.trt_base import TensorRtDetector

                self.model = TensorRtDetector(self.config)
            elif self.config.framework == ModelFrameWork.ORT:
                from ..ML.Detectors.onnx_runtime import ORTDetector

                self.model = ORTDetector(self.config)
            elif self.config.framework == ModelFrameWork.HTTP:
                sub_fw = self.config.sub_framework
                if sub_fw == HTTPSubFrameWork.REKOGNITION:
                    self.model = AWSRekognition(self.config)
                elif sub_fw == HTTPSubFrameWork.VIREL:
                    self.model = VirelAI(self.config)
                elif sub_fw == HTTPSubFrameWork.NONE:
                    raise RuntimeError(
                        f"Invalid HTTP sub framework {sub_fw}, YOU MUST CHOOSE A sub_framework!"
                    )
                else:
                    raise RuntimeError(f"Invalid HTTP sub framework: {sub_fw}")
            elif self.config.framework == ModelFrameWork.FACE_RECOGNITION:
                self.model = FaceRecognitionLibDetector(self.config)
            elif self.config.framework == ModelFrameWork.ALPR:
                if self.config.service == ALPRService.PLATE_RECOGNIZER:
                    self.model = PlateRecognizer(self.config)
                elif self.config.service == ALPRService.OPENALPR:
                    if self.config.api_type == ALPRAPIType.LOCAL:
                        self.model = OpenAlprCmdLine(self.config)
                    elif self.config.api_type == ALPRAPIType.CLOUD:
                        self.model = OpenAlprCloud(self.config)

            elif self.config.framework == ModelFrameWork.CORAL:
                from ..ML.Detectors.coral_edgetpu import TpuDetector

                self.model = TpuDetector(self.config)

            elif self.config.framework == ModelFrameWork.TORCH:
                from ..ML.Detectors.torch.torch_base import TorchDetector

                self.model = TorchDetector(self.config)

            elif self.config.framework == ModelFrameWork.ULTRALYTICS:
                from ..ML.Detectors.ultralytics.yolo.ultra_base import (
                    UltralyticsYOLODetector,
                )

                self.model = UltralyticsYOLODetector(self.config)
            else:
                logger.warning(
                    f"CANT CREATE DETECTOR -> Framework NOT IMPLEMENTED!!! {self.config.framework}"
                )
        except Exception as e:
            logger.warning(
                f"Error loading model ({self.config.name}): {e}", exc_info=True
            )
            raise e

    def is_processor_available(self) -> bool:
        """Check if the processor is available"""
        available = False
        processor = self.config.processor
        framework = self.config.framework
        logger.debug(
            f"Checking if {processor} is available to use for {framework} ({self.config.name})"
        )
        if processor == ModelProcessor.NONE:
            available = True
        if framework == ModelFrameWork.ALPR or framework == ModelFrameWork.REKOGNITION:
            available = True
        elif not processor:
            available = True
        elif processor == ModelProcessor.CPU:
            if framework == ModelFrameWork.CORAL:
                logger.error(f"{processor} is not supported for {framework}")
            else:
                available = True

        elif processor == ModelProcessor.TPU:
            if framework == ModelFrameWork.CORAL:
                try:
                    from pycoral.utils.edgetpu import list_edge_tpus
                except ImportError:
                    logger.warning(
                        "pycoral not installed, cannot load any models that use the TPU processor"
                    )
                    available = None
                else:
                    tpus = list_edge_tpus()
                    logger.debug(f"TPU devices found: {tpus}")
                    if tpus:
                        available = True
                    else:
                        logger.warning(
                            "No TPU devices found, cannot load any models that use the TPU processor"
                        )
            else:
                logger.warning("TPU processor is only available for Coral models!")
        elif processor == ModelProcessor.GPU:
            if framework == ModelFrameWork.OPENCV:
                try:
                    import cv2.cuda
                except ImportError:
                    logger.warning(
                        "OpenCV does not have CUDA enabled/compiled, cannot load any models that use the GPU processor"
                    )
                    available = None
                else:
                    # wrap in try block as this will throw an exception if no CUDA devices are found
                    try:
                        if not (cuda_devices := cv2.cuda.getCudaEnabledDeviceCount()):
                            logger.warning(
                                f"No CUDA devices found for '{self.config.name}'"
                            )
                    except Exception as cv2_cuda_exception:
                        logger.warning(
                            f"'{self.config.name}' Error getting CUDA device count: {cv2_cuda_exception}"
                        )
                        available = None
                    else:
                        logger.debug(
                            f"Found {cuda_devices} CUDA device(s) that OpenCV can use for '{self.config.name}'"
                        )
                        available = True
            elif framework == ModelFrameWork.ORT:
                try:
                    import onnxruntime as ort
                except ImportError:
                    logger.warning(
                        "onnxruntime not installed, cannot load any models that use it"
                    )
                    available = None
                else:
                    if ort.get_device() == "GPU":
                        available = True
                    else:
                        logger.warning(f"No GPU devices found for '{self.config.name}'")
            elif framework in (ModelFrameWork.ULTRALYTICS, ModelFrameWork.TORCH):
                try:
                    import torch
                except ImportError:
                    logger.warning(
                        "torch not installed, cannot load any models that use torch"
                    )
                    available = None
                else:
                    if not torch.cuda.is_available():
                        logger.warning(
                            "No CUDA devices found, cannot load any models that use the GPU processor"
                        )
                    else:
                        available = True
            elif framework == ModelFrameWork.FACE_RECOGNITION:
                try:
                    import dlib
                except ImportError:
                    logger.warning(
                        "dlib not installed, cannot load any models that use dlib!"
                    )
                    available = None
                else:
                    try:
                        if dlib.DLIB_USE_CUDA and dlib.cuda.get_num_devices() >= 1:
                            available = True
                    except Exception as dlib_cuda_exception:
                        logger.warning(
                            f"'{self.config.name}' Error getting dlib CUDA device count: {dlib_cuda_exception}"
                        )
                        available = False

            elif framework == ModelFrameWork.DEEPFACE:
                logger.warning("WORKING ON DeepFace models!")
                available = False
            elif framework == ModelFrameWork.TRT:
                available = True
        logger.debug(
            f"{processor} is {'NOT ' if not available else ''}available for {framework} - '{self.config.name}'"
        )
        return available

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect objects in the image"""
        if self.model is None:
            logger.warning(
                f"Detector for {self.config.name} is not loaded, cannot detect objects!"
            )
            return
        return self.model.detect(image)


class GlobalConfig(BaseModel, arbitrary_types_allowed=True):
    available_models: List[
        Union[
            BaseModelConfig,
            CV2YOLOModelConfig,
            ALPRModelConfig,
            FaceRecognitionLibModelConfig,
            TorchModelConfig,
            TRTModelConfig,
            ORTModelConfig,
            TPUModelConfig,
            RekognitionModelConfig,
            VirelAIModelConfig,
        ]
    ] = Field(default_factory=list, description="Available models, call by ID")
    config: Settings = Field(
        default=None, description="Global config, loaded from YAML config file"
    )
    detectors: List[APIDetector] = Field(
        default_factory=list, description="Loaded Detectors"
    )
    user_db: None = Field(None, description="User Database (TinyDB)")


    def get_detector(self, model: BaseModelConfig) -> Optional[APIDetector]:
        """Get a detector by ID (UUID4)"""
        ret_: Optional[APIDetector] = None
        for detector in self.detectors:
            if detector.config.id == model.id:
                ret_ = detector
                break
        if not ret_:
            logger.debug(f"Attempting to create new detector for '{model.name}'")
            try:
                ret_ = APIDetector(model)
            except ImportError as e:
                logger.warning(e, exc_info=True)
            except Exception as e:
                logger.error(e, exc_info=True)
            else:
                self.detectors.append(ret_)
        if not ret_:
            logger.error(f"Unable to create detector for '{model.name}'")
        return ret_


class ServerEnvVars(BaseSettings):
    """Server Environment Variables using pydantic v2 BaseSettings

     NOTE: the name of the attribute must match the name of the environment variable,
    you can set an env-prefix in the model config

    """

    model_config = SettingsConfigDict(
        env_prefix="ML_SERVER_", case_sensitive=True, extra="allow"
    )

    # With the env_prefix set, the env var name will be: ML_SERVER_CONF_FILE
    conf_file: str = Field(None, description="Server YAML config file")
