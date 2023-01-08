import logging
import re
import inspect
from typing import Optional, Dict, Tuple

from ..Log import CLIENT_LOGGER_NAME

logger = logging.getLogger(CLIENT_LOGGER_NAME)


def validate_percentage_or_pixels(v, values, field):
    # get func name programmatically
    _name_ = inspect.currentframe().f_code.co_name
    if v:
        # logger.debug(f"{_name_}:: Validating '{field.name}'[Type: {type(v)}] -> {v}")
        re_match = re.match(r"(0*?1?\.?\d*\.?\d*?)(%|px)?$", str(v), re.IGNORECASE)
        if re_match:
            # logger.debug(
            #     f"{_name_}:: '{field.name}' is valid REGEX re_match: {re_match.groups()}"
            # )
            try:
                starts_with: Optional[re.Match] = None
                type_of = ""
                if re_match.group(1):
                    starts_with = re.search(
                        r"(0*\.|1\.)?(\d*\.?\d*?)(%|px)?$",
                        re_match.group(1),
                        re.IGNORECASE,
                    )
                    # logger.debug(
                    #     f"{_name_}:: '{field.name}' checking starts_with(): {starts_with.groups()}"
                    # )
                    if re_match.group(2) == "%":
                        # Explicit %
                        # logger.debug(f"{_name_}:: '{field.name}' is explicit %")
                        type_of = "Percentage"
                        v = float(re_match.group(1)) / 100.0
                    elif re_match.group(2) == "px":
                        # Explicit px
                        # logger.debug(f"{_name_}:: '{field.name}' is explicit px")
                        type_of = "Pixel"
                        v = int(re_match.group(1))
                    elif starts_with and not starts_with.group(1):
                        """
                        'total_max_area' is valid REGEX re_match: ('1.0', None)
                        'total_max_area' checking starts_with(): ('', '1.0', None)
                        """
                        # there is no '%' or 'px' at end and the string does not start with 0*., ., or 1.
                        # consider it a pixel input (basically an int)
                        # logger.debug(
                        #     f"{_name_}:: '{field.name}' :: there is no '%' or 'px' at end and the string "
                        #     f"does not start with 0*., ., or 1. - CONVERTING TO INT AS PIXEL VALUE"
                        # )
                        type_of = "Pixel"
                        v = int(float(re_match.group(1)))
                    else:
                        # String starts with 0*., . or 1. treat as a float type percentile
                        # logger.debug(
                        #     f"{_name_}:: '{field.name}' :: String starts with [0*., ., 1.] treat as "
                        #     f"a float type percentile"
                        # )
                        type_of = "Percentage"
                        v = float(re_match.group(1))
                    # logger.debug(f"{type_of} value detected for {field.name} ({v})")
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


def validate_resolution(v, **kwargs):
    _RESOLUTION_STRINGS: Dict[str, Tuple[int, int]] = {
        # pixel resolution string to tuple, feed it .casefold().strip()'d string's
        "4kuhd": (3840, 2160),
        "uhd": (3840, 2160),
        "4k": (4096, 2160),
        "6MP": (3072, 2048),
        "5MP": (2592, 1944),
        "4MP": (2688, 1520),
        "3MP": (2048, 1536),
        "2MP": (1600, 1200),
        "1MP": (1280, 1024),
        "1440p": (2560, 1440),
        "2k": (2048, 1080),
        "1080p": (1920, 1080),
        "960p": (1280, 960),
        "720p": (1280, 720),
        "fullpal": (720, 576),
        "fullntsc": (720, 480),
        "pal": (704, 576),
        "ntsc": (704, 480),
        "4cif": (704, 480),
        "2cif": (704, 240),
        "cif": (352, 240),
        "qcif": (176, 120),
        "480p": (854, 480),
        "360p": (640, 360),
        "240p": (426, 240),
        "144p": (256, 144),
    }
    logger.debug(f"Validating Monitor Zone resolution: {v}")
    if not v:
        logger.warning("No resolution provided for monitor zone, will not be able to scale "
                       "zone Polygon if resolution changes")
    elif isinstance(v, str):
        v = v.casefold().strip()
        if v in _RESOLUTION_STRINGS:
            v = _RESOLUTION_STRINGS[v]
        elif v not in _RESOLUTION_STRINGS:
            # check for a valid resolution string
            import re
            # WxH
            if re.match(r"^\d+x\d+$", v):
                v = tuple(int(x) for x in v.split("x"))
            # W*H
            elif re.match(r"^\d+\*\d+$", v):
                v = tuple(int(x) for x in v.split("*"))
            # W,H
            elif re.match(r"^\d+,\d+$", v):
                v = tuple(int(x) for x in v.split(","))
            else:
                logger.warning(
                    f"Invalid resolution string: {v}. Valid strings are: W*H WxH W,H OR "
                    f"{', '.join(_RESOLUTION_STRINGS)}"
                )
    logger.debug(f"Validated Monitor Zone resolution: {v}")
    return v


def validate_points(v, field, **kwargs):
    if v:
        orig = str(v)
        if not isinstance(v, (str, list)):
            raise TypeError(
                f"'{field.name}' Can only be List or string! type={type(v)}"
            )
        elif isinstance(v, str):
            v = [tuple(map(int, x.strip().split(","))) for x in v.split(" ")]
        from shapely.geometry import Polygon

        try:
            Polygon(v)
        except Exception as exc:
            logger.warning(f"Zone points unable to form a valid Polygon: {exc}")
            raise TypeError(
                f"The polygon points [coordinates] supplied "
                f"are malformed! -> {orig}"
            )
        else:
            assert isinstance(v, list)

    return v


