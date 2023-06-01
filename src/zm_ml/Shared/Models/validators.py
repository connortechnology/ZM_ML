import inspect
import logging
import re
from pathlib import Path
from typing import Union


def _validate_replace_localhost(v, field, values, config):
    # logger.debug(f"Validating ENVVAR {field}: {v}")
    if v:
        if v == 'localhost':
            v = "127.0.0.1"
    return v


def validate_no_scheme_url(v, field, values, config):
    _name_ = inspect.currentframe().f_code.co_name
    # logger.debug(f"{_name_}:: Validating '{field.name}' -> {v}")
    if v:
        import re

        if re.match(r"^(http(s)?)://", v):
            pass
            # logger.debug(f"'{field.name}' is valid with schema: {v}")
        else:
            # logger.debug(
            #     f"No schema in '{field.name}', prepending http:// to make {field.name} a valid URL"
            # )
            v = f"http://{v}"
    return v


def validate_octal(v, **kwargs):
    """Validate and transform octal string into an octal"""
    assert isinstance(v, str)
    if v:
        if re.match(r"^(0o[0-7]+)$", v):
            pass
        else:
            raise ValueError(f"Invalid octal string: {v}")
    return v


def validate_log_level(v, **kwargs):
    """Validate and transform log level string into a log level"""
    if v:
        assert isinstance(v, str)
        v = v.strip().upper()
        if re.match(r"^(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)$", v):
            if v == "WARN":
                v = "WARNING"
            elif v == "FATAL":
                v = "CRITICAL"
            if v == "INFO":
                v = logging.INFO
            elif v == "DEBUG":
                v = logging.DEBUG
            elif v == "WARNING":
                v = logging.WARNING
            elif v == "ERROR":
                v = logging.ERROR
            elif v == "CRITICAL":
                v = logging.CRITICAL
        else:
            raise ValueError(f"Invalid log level string: {v}")
    return v


def str2path(v: Union[str, Path, None], **kwargs):
    """Convert a str to a Path object - pydantic validator

    Args:
        v (str|path|None): string to convert to a Path object
    Keyword Args:
        field (pydantic.fields.ModelField): pydantic field object
        values (Dict): pydantic values dict
        config (pydantic.Config): pydantic config object
        """
    # _name_ = inspect.currentframe().f_code.co_name
    # logger.debug(f"{_name_}:: Validating '{field.name}' -> {v}")
    if v:
        assert isinstance(v, (Path, str))
        v = Path(v)
    v.expanduser().resolve()
    return v


def _validate_dir(v, field=None, values=None, config=None):
    if v:
        v = str2path(v, field=field, values=values, config=config)
        assert v.exists(), f"Path [{v}] does not exist"
        assert v.is_dir(), f"Path [{v}] is not a directory"
    return v


def _validate_file(v, field=None, values=None, config=None):
    if v:
        v = str2path(v, field=field, values=values, config=config)
        assert v.exists(), f"Path [{v}] does not exist"
        assert v.is_file(), f"Path [{v}] is not a file"
    return v
