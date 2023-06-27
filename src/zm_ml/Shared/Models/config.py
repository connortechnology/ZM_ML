import logging
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field, validator

from .validators import validate_log_level, str2path, validate_enabled, validate_not_enabled
from ...Server.Models.DEFAULTS import *


class Testing(BaseModel):
    enabled: bool = Field(False)
    substitutions: Dict[str, str] = Field(default_factory=dict)


class DefaultEnabled(BaseModel):
    enabled: bool = Field(True)

    _v = validator('enabled', pre=True, always=True)(validate_enabled)


class DefaultNotEnabled(DefaultEnabled):
    enabled: bool = Field(False)

    _v = validator('enabled', pre=True, always=True)(validate_not_enabled)


class LoggingLevelBase(BaseModel):
    level: Optional[int] = None

    _validate_log_level = validator('level', allow_reuse=True, pre=True, always=True)(validate_log_level)


class LoggingSettings(LoggingLevelBase):
    class ConsoleLogging(DefaultEnabled, LoggingLevelBase):
        pass

    class SyslogLogging(DefaultNotEnabled, LoggingLevelBase):
        address: Optional[str] = Field("")

    class FileLogging(DefaultEnabled, LoggingLevelBase):
        path: Path = Field('/var/log/zm')
        filename_prefix: str = Field("zmML")
        file_name: Optional[str] = None
        user: str = Field(default="www-data")
        group: str = Field(default="www-data")

        _validate_path = validator("path", allow_reuse=True, pre=True)(
            str2path
        )
    class SanitizeLogging(DefaultNotEnabled):
        replacement_str: str = Field(default="<sanitized>")

    class IntegrateZMLogging(DefaultNotEnabled):
        debug_level: int = Field(default=4)

    level = logging.INFO
    console: ConsoleLogging = Field(default_factory=ConsoleLogging)
    syslog: SyslogLogging = Field(default_factory=SyslogLogging)
    integrate_zm: IntegrateZMLogging = Field(default_factory=IntegrateZMLogging)
    file: FileLogging = Field(default_factory=FileLogging)
    sanitize: SanitizeLogging = Field(default_factory=SanitizeLogging)
