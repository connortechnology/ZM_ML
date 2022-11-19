import logging
import re
import inspect
from pathlib import Path
from typing import Optional

from pydantic import AnyUrl

logger = logging.getLogger("ML-Client")


def percentage_and_pixels_validator(v, values, field):
    # get func name programmatically
    _name_ = inspect.currentframe().f_code.co_name
    if v:
        logger.debug(f"{_name_}:: Validating '{field.name}'[Type: {type(v)}] -> {v}")
        re_match = re.match(r"(0*?1?\.?\d*\.?\d*?)(%|px)?$", str(v), re.IGNORECASE)
        if re_match:
            logger.debug(
                f"{_name_}:: '{field.name}' is valid REGEX re_match: {re_match.groups()}"
            )
            try:
                starts_with: Optional[re.Match] = None
                type_of = ""
                if re_match.group(1):
                    starts_with = re.search(
                        r"(0*\.|1\.)?(\d*\.?\d*?)(%|px)?$",
                        re_match.group(1),
                        re.IGNORECASE,
                    )
                    logger.debug(
                        f"{_name_}:: '{field.name}' checking starts_with(): {starts_with.groups()}"
                    )
                    if re_match.group(2) == "%":
                        # Explicit %
                        logger.debug(f"{_name_}:: '{field.name}' is explicit %")
                        type_of = "Percentage"
                        v = float(re_match.group(1)) / 100.0
                    elif re_match.group(2) == "px":
                        # Explicit px
                        logger.debug(f"{_name_}:: '{field.name}' is explicit px")
                        type_of = "Pixel"
                        v = int(re_match.group(1))
                    elif starts_with and not starts_with.group(1):
                        """
                        'total_max_area' is valid REGEX re_match: ('1.0', None)
                        'total_max_area' checking starts_with(): ('', '1.0', None)
                        """
                        # there is no '%' or 'px' at end and the string does not start with 0*., ., or 1.
                        # consider it a pixel input (basically an int)
                        logger.debug(
                            f"{_name_}:: '{field.name}' :: there is no '%' or 'px' at end and the string "
                            f"does not start with 0*., ., or 1. - CONVERTING TO INT AS PIXEL VALUE"
                        )
                        type_of = "Pixel"
                        v = int(float(re_match.group(1)))
                    else:
                        # String starts with 0*., . or 1. treat as a float type percentile
                        logger.debug(
                            f"{_name_}:: '{field.name}' :: String starts with [0*., ., 1.] treat as "
                            f"a float type percentile"
                        )
                        type_of = "Percentage"
                        v = float(re_match.group(1))
                    logger.debug(f"{type_of} value detected for {field.name} ({v})")
            except TypeError or ValueError as e:
                logger.warning(
                    f"{field.name} value: '{v}' could not be converted to a FLOAT! -> {e} "
                )
                v = 1
            except Exception as e:
                logger.warning(
                    f"{field.name} value: '{v}' could not be converted -> {e} "
                )
                v = 1
        else:
            logger.warning(f"{field.name} value: '{v}' malformed!")
            v = 1
    return v


def no_scheme_url_validator(v, field, values, config):
    _name_ = inspect.currentframe().f_code.co_name
    logger.debug(f"{_name_}:: Validating '{field.name}' -> {v}")
    if v:
        import re

        if re.match(r"^(http(s)?)://", v):
            logger.debug(f"'{field.name}' is valid with schema: {v}")
        else:
            logger.debug(
                f"No schema in '{field.name}', prepending http:// to make {field.name} a valid URL"
            )
            v = f"http://{v}"
    return v


def str_2_path_validator(v, field, values, config):
    _name_ = inspect.currentframe().f_code.co_name
    logger.debug(f"{_name_}:: Validating '{field.name}' -> {v}")
    if v:
        assert isinstance(v, (Path, str))
        v = Path(v)
    return v
