import logging
import re
from pathlib import Path
from typing import Optional

from pydantic import AnyUrl

logger = logging.getLogger("ML-Client")


def percentage_and_pixels_validator(input_, values, field):
    logger.debug(f"Validating '{field.name}' -> {input_}")
    if input_:
        re_match = re.match(r"(0*?\.?\d*\.?\d*?)(%|px)?$", str(input_), re.IGNORECASE)
        if re_match:
            try:
                starts_with: Optional[re.Match] = None
                type_of = ''
                if re_match.group(1):
                    starts_with = re.search(
                        r"(0*\.?)(\d*\.?\d*?)(%)?$", re_match.group(1), re.IGNORECASE
                    )
                    if re_match.group(2) == "%":
                        # Explicit %
                        type_of = "Percentage"
                        input_ = float(re_match.group(1)) / 100.0
                    elif re_match.group(2) == "px":
                        # Explicit px
                        type_of = "Pixel"
                        input_ = int(re_match.group(1))
                    elif starts_with and not starts_with.group(1):
                        # there is no '%' or 'px' at end and the string does not start with 0*. or .
                        # consider it a pixel input (basically an int)
                        type_of = "Pixel"
                        input_ = int(re_match.group(1))
                    else:
                        # String starts with 0*. or . treat as a float type percentile
                        type_of = "Percentage"
                        input_ = float(re_match.group(1))
                    logger.debug(
                        f"{type_of} value detected for {field.name} ({input_})"
                    )
            except TypeError or ValueError as e:
                logger.warning(
                    f"{field.name} value: '{input_}' could not be converted to a FLOAT! -> {e} "
                )
                input_ = 1
        else:
            logger.warning(
                f"{field.name} value: '{input_}' malformed!"
            )
            input_ = 1

    return input_


def no_scheme_url_validator(v, field, values, config):
    logger.debug(f"validating {field.name} - {v}")
    if v:
        import re

        if re.match(r"^(http(s)?)://", v):
            logger.debug(f"'{field.name}' is valid with schema: {v}")
        else:
            logger.debug(
                f"No schema in '{field.name}, prepending http:// to make {field.name} a valid URL"
            )
            v = f"http://{v}"
    return v


def str_2_path_validator(v, field, values, config):
    if v:
        assert isinstance(v, (Path, str))
        v = Path(v)
    return v
