import json
import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Union, Dict, List, Optional, Any

import numpy as np
import yaml
from pydantic import (
    BaseModel,
    Field,
    validator,
    IPvAnyAddress,
    PositiveInt,
    SecretStr,
    BaseSettings,
    AnyUrl,
)
from pydantic.fields import ModelField

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
)
from ...Shared.Models.config import Testing, LoggingSettings
from ...Shared.Models.validators import (
    validate_no_scheme_url,
    validate_replace_localhost,
    str2path,
)

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


class LockSettings(BaseModel):
    enabled: bool = Field(True, description="Enable file locking")
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

    @validator("gpu", "cpu", "tpu", always=True)
    def set_lock_name(cls, v, field, values):
        # logger.debug(f"locks validator {v = } --- {field.name = } -- {values = }")
        if v:
            v: LockSetting
            v.name = f"zm-mlapi_{field.name}"
        return v

    def get(self, device: str) -> LockSetting:
        device = device.casefold()
        if device == "gpu":
            return self.gpu
        elif device == "cpu":
            return self.cpu
        elif device == "tpu":
            return self.tpu
        else:
            raise ValueError(f"Invalid device type: {device}")

    @validator("dir", pre=True, always=True)
    def validate_lock_dir(cls, v):
        if not v:
            v = f"{tempfile.gettempdir()}/zm_mlapi/locks"
        temp_dir = Path(v)
        if not temp_dir.exists():
            logger.debug(f"Creating lock directory: {temp_dir}")
            temp_dir.mkdir(parents=True)
        return v


class ServerSettings(BaseModel):
    class JWTSettings(BaseModel):
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
    jwt: JWTSettings = Field(default_factory=JWTSettings, description="JWT Settings")

    _validate_address = validator("address", allow_reuse=True, pre=True)(
        validate_replace_localhost
    )


class DetectionResult(BaseModel):
    success: bool = False
    type: ModelType = None
    processor: ModelProcessor = None
    model_name: str = None

    label: List[str] = None
    confidence: List[Union[float, int]] = None
    bounding_box: List[List[Union[float, int]]] = None


class BaseModelOptions(BaseModel):
    confidence: Optional[float] = Field(
        0.2, ge=0.0, le=1.0, descritpiton="Confidence Threshold"
    )


class CV2YOLOModelOptions(BaseModelOptions):
    nms: Optional[float] = Field(
        0.4, ge=0.0, le=1.0, description="Non-Maximum Suppression Threshold"
    )


class TPUModelOptions(BaseModelOptions):
    class NMSOptions(BaseModel):
        enabled: bool = Field(True, description="Enable Non-Maximum Suppression")
        threshold: Optional[float] = Field(
            0.4, ge=0.0, le=1.0, description="Non-Maximum Suppression Threshold"
        )

    nms: NMSOptions = Field(default_factory=NMSOptions, description="NMS Options")

    class Config:
        extra = "allow"


class FaceRecognitionLibModelDetectionOptions(BaseModelOptions):
    # face_recognition lib config Options
    model: Optional[FaceRecognitionLibModelTypes] = Field(
        FaceRecognitionLibModelTypes.DEFAULT,
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
        FaceRecognitionLibModelTypes.DEFAULT,
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

    @validator("config", "payload")
    def check_json_serializable(cls, v):
        try:
            json.dumps(v)
        except TypeError:
            raise ValueError("Must be JSON serializable, check formatting...")
        return v


class DeepFaceModelOptions(BaseModelOptions):
    pass


class CV2TFModelOptions(BaseModelOptions):
    pass


class PyTorchModelOptions(BaseModelOptions):
    nms: Optional[float] = Field(
        0.3, ge=0.0, le=1.0, description="Non-Maximum Suppression Threshold"
    )


class BaseModelConfig(BaseModel):
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4, description="Unique ID of the model"
    )
    name: str = Field(..., description="model name")
    enabled: bool = Field(True, description="model enabled")
    description: str = Field(None, description="model description")
    framework: ModelFrameWork = Field(
        ModelFrameWork.DEFAULT, description="model framework"
    )
    model_type: ModelType = Field(
        ModelType.DEFAULT, description="model type (object, face, alpr)"
    )
    processor: Optional[ModelProcessor] = Field(
        ModelProcessor.CPU, description="Processor to use for model"
    )

    # todo: add validator that detects framework and sets a default sub framework if it is None.
    sub_framework: Optional[
        Union[OpenCVSubFrameWork, HTTPSubFrameWork, ALPRSubFrameWork]
    ] = Field(OpenCVSubFrameWork.DARKNET, description="sub-framework to use for model")

    detection_options: Union[
        BaseModelOptions,
        FaceRecognitionLibModelDetectionOptions,
        OpenALPRLocalModelOptions,
        OpenALPRCloudModelOptions,
        PlateRecognizerModelOptions,
        ALPRModelOptions,
        TPUModelOptions,
        PyTorchModelOptions,
    ] = Field(
        default_factory=BaseModelOptions,
        description="Default Configuration for the model",
    )

    @validator("name")
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

    _validate_labels = validator("labels", always=True, allow_reuse=True)(
        validate_model_labels
    )

    @validator("config", "input", "classes", pre=True, always=True)
    def str_to_path(cls, v, values, field: ModelField) -> Optional[Path]:
        # logger.debug(f"validating {field.name} - {v = } -- {type(v) = } -- {values = }")
        msg = f"{field.name} must be a path or a string of a path"
        model_name = values.get("name", "Unknown Model")
        lp = f"Model Name: {model_name} ->"

        if v is None:
            # logger.debug(f"{lp} {field.name} is None, passing as it is Optional")
            return v
        elif not isinstance(v, (Path, str)):
            raise ValueError(msg)
        elif isinstance(v, str):
            v = Path(v)
        if field.name == "config":
            if values["input"].suffix == ".weights":
                msg = f"'{field.name}' is required when 'input' is a DarkNet .weights file"
        return v


class CV2YOLOModelConfig(BaseModelConfig):
    input: Path = Field(None, description="model file/dir path (Optional)")
    classes: Path = Field(default=None, description="model labels file path (Optional)")
    config: Path = Field(default=None, description="model config file path (Optional)")
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

    labels: List[str] = Field(
        default=None,
        description="model labels parsed into a list of strings",
        repr=False,
        exclude=True,
    )

    _validate_labels = validator("labels", always=True, allow_reuse=True)(
        validate_model_labels
    )

    @validator("config", "input", "classes", pre=True, always=True)
    def str_to_path(cls, v, values, field: ModelField) -> Optional[Path]:
        # logger.debug(f"validating {field.name} - {v = } -- {type(v) = } -- {values = }")
        msg = f"{field.name} must be a path or a string of a path"
        model_name = values.get("name", "Unknown Model")

        if v is None:
            # logger.debug(f"{lp} {field.name} is None, passing as it is Optional")
            return v
        elif not isinstance(v, (Path, str)):
            raise ValueError(msg)
        elif isinstance(v, str):
            v = Path(v)
        if field.name == "config":
            if values["input"].suffix == ".weights":
                msg = f"'{field.name}' is required when 'input' is a DarkNet .weights file"
        return v


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
    service: ALPRService = Field(ALPRService.DEFAULT, description="ALPR Service to use")
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


class CV2TFModelConfig(BaseModelConfig):
    input: Path = Field(
        None,
        description="model file/dir path (Optional)",
        example="/opt/models/frozen_inference_graph.pb",
    )
    classes: Path = Field(default=None, description="model labels file path (Optional)")
    config: Path = Field(
        default=None,
        description="model config file path (Optional)",
        example="/opt/models/ssd_inception_v2_coco_2017_11_17.pbtxt",
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
        False,
        description="model uses Floating Point 16 Backend for GPU (EXPERIMENTAL!)",
    )

    labels: List[str] = Field(
        default=None,
        description="model labels parsed into a list of strings",
        repr=False,
        exclude=True,
    )

    _validate_labels = validator("labels", always=True, allow_reuse=True)(
        validate_model_labels
    )

    @validator("config", "input", "classes", pre=True, always=True)
    def str_to_path(cls, v, values, field: ModelField) -> Optional[Path]:
        # logger.debug(f"validating {field.name} - {v = } -- {type(v) = } -- {values = }")
        msg = f"{field.name} must be a path or a string of a path"
        model_name = values.get("name", "Unknown Model")
        if v is None:
            # logger.debug(f"{lp} {field.name} is None, passing as it is Optional")
            return v
        elif not isinstance(v, (Path, str)):
            raise ValueError(msg)
        elif isinstance(v, str):
            v = Path(v)
        if field.name == "config":
            if values["input"].suffix == ".weights":
                msg = f"'{field.name}' is required when 'input' is a DarkNet .weights file"
        return v


class PyTorchModelConfig(BaseModelConfig):
    input: Optional[Path] = None
    classes: Optional[Path] = None

    num_classes: Optional[int] = None
    gpu_idx: Optional[int] = None
    pretrained: Optional[str] = Field(
        None, regex=r"(accurate|fast|default|balanced|high_performance|low_performance)"
    )

    conf: Optional[float] = Field(None, ge=0, le=1)
    nms: Optional[float] = Field(None, ge=0, le=1)

    # this is not to be configured by the user. It is parsed classes from the labels file (or default if no file).
    labels: List[str] = Field(
        default=None,
        description="model labels parsed into a list of strings",
        repr=False,
        exclude=True,
    )

    _validate_labels = validator("labels", always=True, allow_reuse=True)(
        validate_model_labels
    )
    _validate = validator("input", "classes", pre=True, always=True, allow_reuse=True)(
        str2path
    )


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
    model_dir: Optional[Path] = Field(Path(DEF_SRV_SYS_MODELDIR))
    image_dir: Optional[Path] = Field(Path(DEF_SRV_SYS_IMAGEDIR))
    config_path: Optional[Path] = Field(Path(DEF_SRV_SYS_CONFDIR))
    variable_data_path: Optional[Path] = Field(DEF_SRV_SYS_DATADIR)
    tmp_path: Optional[Path] = Field(Path(DEF_SRV_SYS_TMPDIR))
    thread_workers: Optional[int] = Field(DEF_SRV_SYS_THREAD_WORKERS)


class Settings(BaseModel):
    testing: Testing = Field(default_factory=Testing)
    substitutions: Dict[str, str] = Field(default_factory=dict)
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

    class Config:
        arbitrary_types_allowed = True

    def get_lock_settings(self):
        return self.locks

    @validator("available_models", always=True)
    def validate_available_models(cls, v, values):
        models = values.get("models")
        if models:
            disabled_models: list = []
            v = []
            for model in models:
                if not model:
                    logger.warning(
                        f"Model is empty, this usually means there is a formatting error "
                        f"(or an empty model; just a '-') in your config file! Skipping"
                    )
                    continue
                final_model = None
                if model.get("enabled", True) is True:
                    # logger.debug(f"Adding model: {type(model) = } ------ {model = }")
                    _model: Union[
                        RekognitionModelConfig,
                        VirelAIModelConfig,
                        FaceRecognitionLibModelConfig,
                        ALPRModelConfig,
                        CV2YOLOModelConfig,
                        None,
                    ] = None
                    _framework = model.get("framework", "opencv")
                    _sub_fw = model.get("sub_framework", "darknet")
                    _options = model.get("detection_options", {})
                    _type = model.get("model_type", ModelType.OBJECT)
                    if _framework == ModelFrameWork.HTTP:
                        if _sub_fw == HTTPSubFrameWork.REKOGNITION:
                            model["model_type"] = ModelType.OBJECT
                            model["processor"] = ModelProcessor.NONE
                            final_model = RekognitionModelConfig(**model)
                        elif _sub_fw == HTTPSubFrameWork.VIREL:
                            model["model_type"] = ModelType.OBJECT
                            model["processor"] = ModelProcessor.NONE
                            final_model = VirelAIModelConfig(**model)
                    elif _framework == ModelFrameWork.FACE_RECOGNITION:
                        model["model_type"] = ModelType.FACE
                        model[
                            "detection_options"
                        ] = FaceRecognitionLibModelDetectionOptions(**_options)
                        final_model = FaceRecognitionLibModelConfig(**model)

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
                        final_model = config
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
                            final_model = config
                    elif _framework == ModelFrameWork.CORAL:
                        config = TPUModelConfig(**model)
                        config.detection_options = TPUModelOptions(**_options)
                        final_model = config
                    elif _framework == ModelFrameWork.TORCH:
                        config = PyTorchModelConfig(**model)
                        config.detection_options = PyTorchModelOptions(**_options)
                        final_model = config
                    else:
                        raise NotImplementedError(
                            f"Framework {_framework} not implemented"
                        )

                    # load model here
                    if final_model:
                        logger.debug(f"Adding model: {final_model.name}")
                        v.append(final_model)
                else:
                    disabled_models.append(model.get("name"))
            if disabled_models:
                logger.debug(f"Skipped these disabled models: {disabled_models}")
        return v


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
    ]
    _options: Union[
        BaseModelOptions,
        OpenALPRLocalModelOptions,
        OpenALPRCloudModelOptions,
        PlateRecognizerModelOptions,
        ALPRModelOptions,
        FaceRecognitionLibModelDetectionOptions,
        CV2YOLOModelOptions,
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
            PyTorchModelConfig,
        ],
    ):
        from ..ML.Detectors.opencv.cv_yolo import CV2YOLODetector
        from ..ML.Detectors.coral_edgetpu import TpuDetector
        from ..ML.Detectors.face_recognition import FaceRecognitionLibDetector
        from ..ML.Detectors.alpr import PlateRecognizer, OpenAlprCmdLine, OpenAlprCloud
        from ..ML.Detectors.virelai import VirelAI
        from ..ML.Detectors.aws_rekognition import AWSRekognition
        from ..ML.Detectors.pytorch.torch_base import TorchDetector

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
                VirelAIModelConfig,
                RekognitionModelConfig,
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
            # framework is not installed
            logger.warning(
                f"Library missing, cannot create detector for {self.config.name}"
            )
            self.config = None
            from ..app import get_global_config

            for _model in get_global_config().available_models:
                if _model.id == self.id:
                    get_global_config().available_models.remove(_model)
                    break

            return

        try:
            if self.config.framework == ModelFrameWork.OPENCV:
                if (
                    self.config.sub_framework == OpenCVSubFrameWork.DARKNET
                    or self.config.sub_framework == OpenCVSubFrameWork.ONNX
                ):
                    from ..ML.Detectors.opencv.cv_yolo import CV2YOLODetector

                    self.model = CV2YOLODetector(self.config)
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
                from ..ML.Detectors.pytorch.torch_base import TorchDetector

                self.model = TorchDetector(self.config)

            else:
                logger.warning(
                    f"CANT CREATE DETECTOR -> Framework NOT IMPLEMENTED!!! {self.config.framework}"
                )
        except Exception as e:
            logger.warning(f"Error loading model: {e}")
        else:
            logger.debug(f"APIDetector:_load_model()-> {self.model = }")

    def is_processor_available(self) -> bool:
        """Check if the processor is available"""
        available = False
        processor = self.config.processor
        framework = self.config.framework
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
                                "No CUDA devices found, cannot load any models that use the GPU processor"
                            )
                    except Exception as cv2_cuda_exception:
                        logger.warning(
                            f"Error getting CUDA device count: {cv2_cuda_exception}"
                        )
                        available = None
                    else:
                        logger.debug(
                            f"Found {cuda_devices} CUDA device(s) that OpenCV can use"
                        )
                        available = True
            elif framework == ModelFrameWork.TENSORFLOW:
                try:
                    import tensorflow as tf
                except ImportError:
                    logger.warning(
                        "tensorflow not installed, cannot load any models that use tensorflow GPU processor"
                    )
                    available = None
                else:
                    if not tf.config.list_physical_devices("GPU"):
                        logger.warning(
                            "No CUDA devices found, cannot load any models that use the GPU processor"
                        )
                    else:
                        available = True
            elif framework == ModelFrameWork.TORCH:
                try:
                    import torch
                except ImportError:
                    logger.warning(
                        "pytorch not installed, cannot load any models that use pytorch"
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
                        "dlib not installed, cannot load any models that use dlib GPU processor"
                    )
                    available = None
                else:
                    try:
                        if dlib.DLIB_USE_CUDA and dlib.cuda.get_num_devices() >= 1:
                            available = True
                    except Exception as dlib_cuda_exception:
                        logger.warning(
                            f"Error getting CUDA device count: {dlib_cuda_exception}"
                        )
                        available = False

            elif framework == ModelFrameWork.DEEPFACE:
                logger.warning("WORKING ON DeepFace models!")
                available = False
        return available

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect objects in the image"""
        assert self.model, "model not loaded"
        return self.model.detect(image)


class GlobalConfig(BaseModel):
    available_models: List[BaseModelConfig] = Field(
        default_factory=list, description="Available models, call by ID"
    )
    config: Settings = Field(
        default=None, description="Global config, loaded from YAML config file"
    )
    detectors: List[APIDetector] = Field(
        default_factory=list, description="Loaded Detectors"
    )

    class Config:
        arbitrary_types_allowed = True

    def get_detector(self, model: BaseModelConfig) -> Optional[APIDetector]:
        """Get a detector by ID"""
        ret_: Optional[APIDetector] = None
        for detector in self.detectors:
            if detector.config.id == model.id:
                ret_ = detector
        if not ret_:
            logger.debug(f"Creating new detector for '{model.name}'")
            ret_ = APIDetector(model)
            logger.debug(f"returned detctor -> {ret_}")
            if ret_.config is not None:
                self.detectors.append(ret_)
        if not ret_:
            logger.error(f"Unable to create detector for {model.name}")
        return ret_


class ServerEnvVars(BaseSettings):
    """Server Environment Variables"""

    conf_file: str = Field(
        None, env="ML_SERVER_CONF_FILE", description="Server YAML config file"
    )

    class Config:
        # env_file = ".env"
        # env_file_encoding = "utf-8"
        case_sensitive = True
        # extra = "forbid"
