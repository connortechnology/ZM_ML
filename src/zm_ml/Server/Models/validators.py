"""Pydantic validators for the Models in this package"""
from pathlib import Path
from typing import List, Optional, IO, Dict
import logging

from ..Log import SERVER_LOGGER_NAME
from pydantic import FieldValidationInfo

logger = logging.getLogger(SERVER_LOGGER_NAME)


def validate_model_labels(
    v, info: FieldValidationInfo, **kwargs
) -> Optional[List[str]]:
    logger.debug(f"{kwargs = }\n\n")
    model_name = kwargs.get("model_name", "Unknown")
    labels_file: Optional[Path] = kwargs.get("labels_file", None)
    lp = f"Model Name: {model_name} ->"
    if not labels_file:
        logger.debug(
            f"{lp} 'classes' is not defined. Using *default* COCO 2017 class labels"
        )
        from ..ML.coco17_cv2 import COCO17

        return COCO17
    logger.debug(
        f"{lp} 'classes' is defined. Parsing '{labels_file}' into a list of strings for class identification"
    )
    assert isinstance(labels_file, Path), f"{lp} '{labels_file}' is not a Path object"
    assert labels_file.exists(), f"{lp} '{labels_file}' does not exist"
    assert labels_file.is_file(), f"{lp} '{labels_file}' is not a file"
    with labels_file.open(mode="r") as f:
        f: IO
        v = f.read().splitlines()
    assert isinstance(
        v, list
    ), f"{lp} After parsing the file into a list of strings, {info.name} is not a list"
    return v
