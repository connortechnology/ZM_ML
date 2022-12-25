import logging
from pathlib import Path
import tempfile
from typing import Union, Dict, List, Optional

from pydantic import BaseModel, BaseSettings, Field, validator
from pydantic.fields import ModelField

from .schemas import ModelFrameWork, ModelType, RekognitionModelConfig, \
    FaceRecognitionLibModelOptions, FaceRecognitionLibModelConfig, ALPRModelConfig, ALPRService, ALPRAPIType, \
    OpenALPRCloudModelOptions, OpenALPRLocalModelOptions, PlateRecognizerModelOptions, \
    CV2YOLOModelOptions, CV2YOLOModelConfig, BaseModelConfig

logger = logging.getLogger("ZM_ML-API")


class ModelConfigFromFile:
    file: Path
    raw: str
    parsed_raw: dict
    parsed: dict

    def __iter__(self):
        return iter(self.parsed.get("models", []))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.file})->{self.parsed.get('models')}"

    def __init__(self, cfg_file: Union[Path, str]):
        self.raw = ""
        self.parsed_raw = {}
        self.parsed = {}
        if not isinstance(cfg_file, (Path, str)):
            raise TypeError(
                f"ModelConfig: The configuration file must be a string or a Path object."
            )
        if isinstance(cfg_file, str):
            cfg_file = Path(cfg_file)
        if cfg_file.exists() and cfg_file.is_file():
            self.file = cfg_file
            with open(self.file, "r") as f:
                self.raw = f.read()
            self.parsed_raw = self.model_config_parser(self.raw)
            self.parsed = self.substitution_vars(self.parsed_raw)
        else:
            logger.error(
                f"ModelConfig: The configuration file '{cfg_file}' does not exist or is not a file."
            )

    def get_model_config(self):
        return self.parsed

    def get_model_config_str(self):
        return self.raw

    def get_model_config_raw(self):
        return self.parsed_raw

    def model_config_parser(self, cfg_str: str) -> dict:
        """Parse the YAML model configuration file.

        Args:
            cfg_str (str): Configuration YAML file as a string.

        """
        cfg: dict = {}
        import yaml

        try:
            cfg = yaml.safe_load(cfg_str)
        except yaml.YAMLError as e:
            logger.error(
                f"model_config_parser: Error parsing the YAML configuration file!"
            )
            raise e
        return cfg

    def substitution_vars(self, cfg: Dict[str, str]) -> Dict[str, str]:
        """Substitute variables in the configuration file.

        Args:
            cfg (Dict[str, str]): The configuration dictionary.

        Returns:
            Dict[str, str]: The configuration dictionary with variables substituted.
        """
        # turn dict into a string to use regex search/replace globally instead of iterating over the dict
        cfg_str = str(cfg)
        # find all variables in the string
        import re

        var_list = re.findall(r"\$\{(\w+)\}", cfg_str)
        if var_list:
            var_list = list(set(var_list))
            logger.debug(
                f"substitution_vars: Found the following variables in the configuration file: {var_list}"
            )
            # substitute variables
            for var in var_list:
                num_var = len(re.findall(f"\${{{var}}}", cfg_str))
                if var in cfg:
                    logger.debug(
                        f"substitution_vars: Found {num_var} occurrence{'s' if num_var != 1 else ''} of '${{{var}}}', "
                        f"Substituting with value '{cfg[var]}'"
                    )
                    cfg_str = cfg_str.replace(f"${{{var}}}", cfg[var])
                else:
                    logger.warning(
                        f"substitution_vars: The variable '${{{var}}}' is not defined."
                    )
            from ast import literal_eval

            return literal_eval(cfg_str)
        else:
            logger.debug(
                f"substitution_vars: No variables for substituting in the configuration file."
            )
        return cfg


class LockSetting(BaseModel):
    max: int = Field(1, ge=1, le=100, description="Maximum number of parallel processes")
    timeout: int = Field(30, ge=1, le=480, description="Timeout in seconds for acquiring a lock")
    name: Optional[str] = Field(None, description="Name of the lock file")

class LockSettings(BaseModel):
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


    @validator("gpu", "cpu", "tpu", pre=True, always=True)
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

from ...Shared.Models.config import Testing, SystemSettings, LoggingSettings
from pydantic import IPvAnyAddress, validator, PositiveInt, SecretStr

class ServerSettings(BaseModel):
    class JWTSettings(BaseModel):
        sign_key: SecretStr = Field("CHANGE ME!!!!", description="JWT Sign Key")
        algorithm: str = Field("HS256", description="JWT Algorithm")

    address: IPvAnyAddress = Field('0.0.0.0', description="Server listen address")
    port: PositiveInt = Field(8000, description="Server listen port")
    model_config: Path = Field(
        ..., description="Path to the model configuration YAML file"
    )
    reload: bool = Field(
        default=False, description="Uvicorn reload - For development only"
    )
    debug: bool = Field(default=False, description="Uvicorn debug mode - For development only")
    jwt: JWTSettings = Field(default_factory=JWTSettings, description="JWT Settings")

class Settings(BaseSettings):
    testing: Testing = Field(default_factory=Testing)
    substitutions: Dict[str, str] = Field(default_factory=dict)
    system: SystemSettings = Field(default_factory=SystemSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    locks: LockSettings = Field(
        default_factory=LockSettings, description="Lock Settings", repr=False
    )

    models: ModelConfigFromFile = Field(
        None, description="ModelConfig object", exclude=True, repr=False
    )
    available_models: List[BaseModelConfig] = Field(
        None, description="Available models"
    )

    def get_lock_settings(self):
        return self.locks

    @validator("available_models")
    def validate_available_models(cls, v, values):
        models = values.get("models")
        # logger.info(f"Settings._validate_available_models: {models = }")
        if models:
            v = []
            for model in models:
                if model.get("enabled", True) is True:
                    # logger.debug(f"Adding model: {type(model) = } ------ {model = }")
                    _framework = model.get("framework", "yolo")
                    _options = model.get("detection_options", {})
                    _type = model.get("model_type", "object")
                    logger.debug(
                        f"DBG<<< {_framework} FRAMEWORK RAW options are {_options}"
                    )
                    if _framework == ModelFrameWork.REKOGNITION:
                        model["model_type"] = ModelType.OBJECT
                        model["processor"] = None
                        v.append(RekognitionModelConfig(**model))

                    if _framework == ModelFrameWork.FACE_RECOGNITION:
                        model["model_type"] = ModelType.FACE
                        model["detection_options"] = FaceRecognitionLibModelOptions(
                            **_options
                        )
                        v.append(FaceRecognitionLibModelConfig(**model))

                    elif _framework == ModelFrameWork.ALPR:
                        model["model_type"] = ModelType.ALPR
                        api_type = model.get("api_type", "local")
                        api_service = model.get("service", "openalpr")
                        logger.debug(
                            f"DEBUG>>> FrameWork: {_framework}  Service: {api_service} [{api_type}]"
                        )
                        config = ALPRModelConfig(**model)
                        if api_service == ALPRService.OPENALPR:
                            if api_type == ALPRAPIType.LOCAL:
                                config.detection_options = OpenALPRLocalModelOptions(
                                    **_options
                                )
                            elif api_type == ALPRAPIType.CLOUD:
                                config.processor = None
                                config.detection_options = OpenALPRCloudModelOptions(
                                    **_options
                                )
                        elif api_service == ALPRService.PLATE_RECOGNIZER:
                            if api_service == ALPRAPIType.CLOUD:
                                config.processor = None
                            config.detection_options = PlateRecognizerModelOptions(
                                **_options
                            )
                        logger.debug(
                            f"DEBUG>>> FINAL ALPR OPTIONS {config.detection_options = }"
                        )
                        v.append(config)
                    elif _framework == ModelFrameWork.CV_YOLO:
                        config = CV2YOLOModelConfig(**model)
                        config.detection_options = CV2YOLOModelOptions(**_options)
                        logger.debug(
                            f"DEBUG>>> FINAL YOLO OPTIONS {config.detection_options = }"
                        )
                        v.append(config)
                    else:
                        logger.debug(
                            f"DEBUG>>> this FRAMEWORK is NOT IMPLEMENTED -> {_framework}"
                        )
                        v.append(BaseModelConfig(**model))
                else:
                    logger.debug(f"Skipping disabled model: {model.get('name')}")

                # logger.info(f"Settings._validate_available_models: {model = }")
        return v

    @validator("debug")
    def check_debug(cls, v, values):
        if v:
            values["reload"] = True
            logger.info(
                f"Debug mode is enabled. The server will reload on every code change and logging level is set to DEBUG."
            )
            logger.setLevel(logging.DEBUG)
        return v

    @validator("models")
    def _validate_model_config(cls, v, field, values):
        model_config = values["model_config"]
        logger.debug(f"parsing model config: {model_config}")
        v = ModelConfigFromFile(model_config)
        return v

    @validator("model_config", "log_dir", pre=True, always=True)
    def _validate_path(cls, v, values, field: ModelField) -> Optional[Path]:
        """Take a Path or str and return a validated Path object"""
        logger.debug(f"validating {field.name} - {v = } -- {type(v) = } -- {values = }")
        if not v:
            raise ValueError(f"{field.name} is required")
        assert isinstance(v, (Path, str)), f"{field.name} must be a Path or str"
        if isinstance(v, str):
            v = Path(v)
        if field.name == "model_config":
            assert v.exists(), "model_config does not exist"
            assert v.is_file(), "model_config is not a file"
        elif field.name == "log_dir":
            if not v.exists():
                logger.warning(
                    f"{field.name} directory: {v} does not exist, creating..."
                )
                v.mkdir(parents=True, exist_ok=True)
            assert v.is_dir(), "log_dir is not a directory"
        return v
