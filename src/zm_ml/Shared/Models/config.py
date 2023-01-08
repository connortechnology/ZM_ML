import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field, validator

from .validators import validate_log_level, str_to_path


class Testing(BaseModel):
    enabled: bool = Field(False)
    substitutions: Dict[str, str] = Field(default_factory=dict)


class SystemSettings(BaseModel):
    variable_data_path: Optional[Path] = Field(Path("/var/lib/zm_ml"))
    tmp_path: Optional[Path] = Field(Path(tempfile.gettempdir()) / "zm_ml")
    thread_workers: Optional[int] = Field(4)


class DefaultEnabled(BaseModel):
    enabled: bool = Field(True)


class DefaultNotEnabled(DefaultEnabled):
    enabled: bool = Field(False)


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
        user: str = Field(default="www-data")
        group: str = Field(default="www-data")

        _validate_path = validator("path", allow_reuse=True, pre=True)(
            str_to_path
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
