import logging
import sys
from decimal import Decimal
from inspect import getframeinfo, stack
from pathlib import Path
from typing import Union, Any, Dict, List, Type, Optional

from pydantic import BaseSettings, Field, IPvAnyAddress, AnyUrl, SecretStr, validator, BaseModel
from ..Client.Libs.api import ZMApi
from ..Client.Models.config import ConfigFileModel

from .Models.validators import str_to_path

logger = logging.getLogger("ZM_ML-Client")


class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.

    To get the singleton instance, use the `instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    """
    _decorated: Optional[Type] = None
    _instance: Optional[object] = None
    def __init__(self, decorated):
        self._decorated = decorated

    def instance(self, *args, **kwargs):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated(*args, **kwargs)
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `instance()`.')


class ZMEnvVars(BaseSettings):
    zm_conf_dir: Path = Field(
        Path('/etc/zm'), description="Path to ZoneMinder config files", env="CONF_DIR"
    )

    @validator("zm_conf_dir", pre=True, always=True)
    def _validate_conf_path(cls, v, **kwargs):
        if v:
            assert isinstance(v, (str, Path)), f"zm_conf_dir must be a Path or str object, not {type(v)}"
            v = str_to_path(v, **kwargs)
            assert v.exists(), f"ZoneMinder config path [{v}] does not exist"
            assert v.is_dir(), f"ZoneMinder config path [{v}] is not a directory"
        return v
    # def __init__(self, **data: Any):
    #     logger.debug(f"Looking for ZoneMinder config dir via Environment Variable ZM_CONF_DIR")
    #     super().__init__(**data)

    class Config:
        env_prefix = "ZM_"
        check_fields = False


class MLEnvVars(ZMEnvVars):
    ml_conf_dir: Path = Field(
        None, description="Path to ZoneMinder ML config files", env="CONF_DIR"
    )

    @validator("ml_conf_dir", pre=True, always=True)
    def _validate_conf_dir(cls, v, **kwargs):
        field = kwargs["field"]
        if v:
            assert isinstance(v, (str, Path)), f"ENVVAR {field.name} must be a Path or str object"
            v = str_to_path(v, **kwargs)
            assert v.exists(), f"ZM-ML config path [{v}] does not exist"
            assert v.is_dir(), f"ZM-ML config path [{v}] is not a directory"
        return v

    class Config:
        env_prefix = "ML_"
        check_fields = False


class ServerEnvVars(MLEnvVars):
    server_conf_file: Path = Field(
        None, description="Path to ZM-ML SERVER config file", env="CONF_FILE"
    )

    class Config:
        env_prefix = "ML_SERVER_"
        check_fields = False


class ClientEnvVars(MLEnvVars):
    client_conf_file: Path = Field(
        None, description="Path to ZM-ML CLIENT config file", env="CONF_FILE"
    )
    db_host: Union[IPvAnyAddress, AnyUrl] = Field("127.0.0.1", description="Database host", env="DBHOST")
    db_user: str = Field("zmuser", description="Database user", env="DBUSER")
    db_password: SecretStr = Field("zmpass", description="Database password", env="DBPASS")
    db_name: str = Field("zm", description="Database name", env="DBNAME")
    db_driver: str = Field("mysql+pymysql", description="Database driver", env="DBDRIVER")
    class Config:
        env_prefix = "ML_CLIENT_"
        check_fields = False
    @validator("db_host", pre=True)
    def _validate_db_host(cls, v, field, **kwargs):
        # logger.debug(f"Validating ENVVAR {field}: {v}")
        if v:
            if v == 'localhost':
                v = "127.0.0.1"
        return v

    @validator("conf_file", pre=True, always=True, check_fields=False)
    def _validate_conf_file(cls, v, **kwargs):
        field = kwargs.get("field")
        values = kwargs.get("values")
        # logger.debug(f"Validating ENVVAR {field}: {v}")
        if v:
            assert isinstance(v, (str, Path)), f"ENVVAR {field.name} must be a Path or str object"
            v = str_to_path(v, **kwargs)
            assert v.exists(), f"ZM-ML config file [{v}] does not exist"
            assert v.is_file(), f"ZM-ML config file [{v}] is not a file"
        return v


class GlobalConfig(BaseModel):
    api: ZMApi = None
    mid: int = None
    config: ConfigFileModel = None
    config_file: Union[str, Path] = None
    configs_path: Union[str, Path] = None
    eid: int = None
    mon_name: str = None
    mon_post: int = None
    mon_pre: int = None
    mon_fps: Decimal = None
    reason: str = None
    notes: str = None
    event_path: Path = None
    event_cause: str = None
    past_event: bool = False
    Event: Dict = None
    Monitor: Dict = None
    Frame: List = None
    mon_image_buffer_count: int = None
    mon_width: int = None
    mon_height: int = None
    mon_colorspace: int = None
    frame_buffer: Dict = Field(default_factory=dict)

    Environment: ClientEnvVars = None
    imported_zones: list = Field(default_factory=list)
    random: Dict = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
