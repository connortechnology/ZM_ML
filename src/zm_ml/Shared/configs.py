from __future__ import annotations

import logging
from decimal import Decimal
from pathlib import Path
from typing import Union, Dict, List, Type, Optional, TYPE_CHECKING

from pydantic import BaseSettings, Field, IPvAnyAddress, AnyUrl, SecretStr, validator, BaseModel

from .Models.validators import _validate_replace_localhost, _validate_dir, _validate_file
from ..Client.Libs.API import ZMAPI
from ..Client.Log import CLIENT_LOGGER_NAME
from ..Client.Models.config import ConfigFileModel
from ..Client.Libs.DB import ZMDB
from ..Server.Log import SERVER_LOGGER_NAME

if TYPE_CHECKING:
    from ..Client.Models.config import ClientEnvVars

# find loggers
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
if CLIENT_LOGGER_NAME in loggers:
    logger = logging.getLogger(CLIENT_LOGGER_NAME)
elif SERVER_LOGGER_NAME in loggers:
    logger = logging.getLogger(SERVER_LOGGER_NAME)
else:
    logger = logging.getLogger(CLIENT_LOGGER_NAME)


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

class GlobalConfig(BaseModel):
    api: Optional[ZMAPI] = None
    db: Optional[ZMDB] = None
    mid: Optional[int] = None
    config: Optional[ConfigFileModel] = None
    config_file: Union[str, Path, None] = None
    configs_path: Union[str, Path, None] = None
    eid: Optional[int] = None
    mon_name: Optional[str] = None
    mon_post: Optional[int] = None
    mon_pre: Optional[int] = None
    mon_fps: Optional[Decimal] = None
    reason: Optional[str] = None
    notes: Optional[str] = None
    event_path: Optional[Path] = None
    event_cause: Optional[str] = None
    past_event: bool = False
    Event: Optional[Dict] = None
    Monitor: Optional[Dict] = None
    Frame: Optional[List] = None
    mon_image_buffer_count: Optional[int] = None
    mon_width: Optional[int] = None
    mon_height: Optional[int] = None
    mon_colorspace: Optional[int] = None
    frame_buffer: Optional[Dict] = Field(default_factory=dict)

    Environment: Optional[Union["ClientEnvVars"]] = None
    imported_zones: list = Field(default_factory=list)
    random: Dict = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"
