from __future__ import annotations

import json
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from platform import python_version
from typing import Union, Dict, List, Optional, Any, TYPE_CHECKING, Annotated, Tuple

from starlette.requests import Request
from starlette.responses import StreamingResponse, Response

try:
    import cv2
except ImportError:
    cv2 = None
    raise ImportError(
        "OpenCV is not installed, please install it by compiling it for "
        "CUDA/cuDNN GPU support/OpenVINO Intel CPU/iGPU/dGPU support or "
        "quickly for only cpu support using 'opencv-contrib-python' package"
    )
import numpy as np

import pydantic
import uvicorn

from fastapi import (
    FastAPI,
    HTTPException,
    __version__ as fastapi_version,
    UploadFile,
    File,
    Body,
    Depends,
)
from fastapi.responses import RedirectResponse, Response
from fastapi.security import OAuth2PasswordRequestForm

from ..Shared.Models.Enums import ModelType, ModelFrameWork, ModelProcessor
from .Log import SERVER_LOGGER_NAME, SERVER_LOG_FORMAT
from ..Shared.Log.handlers import BufferedLogHandler
from .Models.config import BaseModelConfig
from .auth import *

if TYPE_CHECKING:
    from ..Shared.Models.config import DetectionResults
    from .Models.config import (
        GlobalConfig,
        Settings,
        APIDetector,
    )

__version__ = "0.0.1a4"
__version_type__ = "dev"
logger = logging.getLogger(SERVER_LOGGER_NAME)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())
# Control uvicorn logging, what a mess!
uvi_logger = logging.getLogger("uvicorn")
uvi_error_logger = logging.getLogger("uvicorn.error")
uvi_access_logger = logging.getLogger("uvicorn.access")
uvi_loggers = (uvi_logger, uvi_error_logger, uvi_access_logger)
for _ul in uvi_loggers:
    _ul.setLevel(logging.DEBUG)
    _ul.propagate = False

logger.info(
    f"ZM_MLAPI: {__version__} (type: {__version_type__}) [Python: {python_version()} - "
    f"OpenCV: {cv2.__version__} - Numpy: {np.__version__} - FastAPI: {fastapi_version} - "
    f"Pydantic: {pydantic.VERSION}]"
)

app = FastAPI(debug=True, title="ZoMi MLAPI", version=__version__)
g: Optional[GlobalConfig] = None
LP: str = "mlapi:"


def create_logs() -> logging.Logger:
    formatter = SERVER_LOG_FORMAT
    logger = logging.getLogger(SERVER_LOGGER_NAME)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    buffered_log_handler = BufferedLogHandler()
    buffered_log_handler.setFormatter(formatter)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    logger.addHandler(buffered_log_handler)
    for _ul in uvi_loggers:
        _ul.setLevel(logging.DEBUG)
        _ul.addHandler(console_handler)

    return logger


def init_logs(config: Settings) -> None:
    """Initialize the logging system."""
    import getpass
    import grp
    import os
    from ..Shared.Log.handlers import BufferedLogHandler

    lp: str = "init:logs:"
    sys_user: str = getpass.getuser()
    sys_gid: int = os.getgid()
    sys_group: str = grp.getgrgid(sys_gid).gr_name
    sys_uid: int = os.getuid()
    from ..Shared.Models.config import LoggingSettings

    cfg: LoggingSettings = config.logging
    root_level = cfg.level
    logger.debug(
        f"{lp} Setting root logger level to {logging._levelToName[root_level]}"
    )
    logger.setLevel(root_level)
    for _ul in uvi_loggers:
        _ul.setLevel(root_level)

    if cfg.console.enabled is False:
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler):
                if h.stream == sys.stdout:
                    logger.info(f"{lp} Removing console log output!")
                    logger.removeHandler(h)

    if cfg.file.enabled:
        if cfg.file.file_name:
            _filename = cfg.file.file_name
        else:
            from .Models.DEFAULTS import DEF_SRV_LOGGING_FILE_FILENAME

            _filename = DEF_SRV_LOGGING_FILE_FILENAME
        abs_logfile = (cfg.file.path / _filename).expanduser().resolve()
        try:
            if not abs_logfile.exists():
                logger.info(f"{lp} Creating log file [{abs_logfile}]")
                from .Models.DEFAULTS import DEF_SRV_LOGGING_FILE_CREATE_MODE

                abs_logfile.touch(exist_ok=True, mode=DEF_SRV_LOGGING_FILE_CREATE_MODE)
            else:
                # Test if read/write permissions are available
                with abs_logfile.open(mode="a") as f:
                    pass
        except PermissionError:
            logger.warning(
                f"{lp} Logging to file disabled due to permissions"
                f" - No write access to '{abs_logfile.as_posix()}' for user: "
                f"{sys_uid} [{sys_user}] group: {sys_gid} [{sys_group}]"
            )
        else:
            # todo: add timed rotating log file handler if configured (for systems without logrotate)
            file_handler = logging.FileHandler(abs_logfile.as_posix(), mode="a")
            # file_handler = logging.handlers.TimedRotatingFileHandler(
            #     file_from_config, when="midnight", interval=1, backupCount=7
            # )
            file_handler.setFormatter(SERVER_LOG_FORMAT)
            if cfg.file.level:
                logger.debug(f"File logger level CONFIGURED AS {cfg.file.level}")
                # logger.debug(f"Setting file log level to '{logging._levelToName[g.config.logging.file.level]}'")
                file_handler.setLevel(cfg.file.level)
            logger.addHandler(file_handler)
            for _ul in uvi_loggers:
                _ul.addHandler(file_handler)

            # get the buffered handler and call flush with file_handler as a kwarg
            # this will flush the buffer to the file handler
            for h in logger.handlers:
                if isinstance(h, BufferedLogHandler):
                    logger.debug(f"Flushing buffered log handler to file")
                    h.flush(file_handler=file_handler)
                    # Close the buffered handler
                    h.close()
                    break
            logger.debug(
                f"Logging to file '{abs_logfile}' with user: "
                f"{sys_uid} [{sys_user}] group: {sys_gid} [{sys_group}]"
            )
    if cfg.syslog.enabled:
        # enable syslog logging
        syslog_handler = logging.handlers.SysLogHandler(
            address=cfg.syslog.address,
        )
        syslog_handler.setFormatter(SERVER_LOG_FORMAT)
        if cfg.syslog.level:
            logger.debug(
                f"Syslog logger level CONFIGURED AS {logging._levelToName[cfg.syslog.level]}"
            )
            syslog_handler.setLevel(cfg.syslog.level)
        logger.addHandler(syslog_handler)
        logger.debug(f"Logging to syslog at {cfg.syslog.address}")

    logger.info(f"Logging initialized...")


def get_global_config() -> GlobalConfig:
    return g


def create_global_config() -> GlobalConfig:
    """Create the global config object"""
    from .Models.config import GlobalConfig

    global g
    if not isinstance(g, GlobalConfig):
        g = GlobalConfig()
    return get_global_config()


def get_settings() -> Settings:
    return get_global_config().config


def get_available_models() -> List[BaseModelConfig]:
    return get_global_config().available_models


def locks_enabled():
    return get_settings().locks.enabled


def normalize_id(_id: str) -> str:
    return _id.strip().lower()


def get_model(model_hint: Union[str, BaseModelConfig]) -> BaseModelConfig:
    """Get a model based on the hint provided. Hint can be a model name, model id, or a model object"""
    logger.debug(f"get_model: hint TYPE: {type(model_hint)} -> {model_hint}")
    available_models = get_available_models()
    if available_models:
        if isinstance(model_hint, BaseModelConfig):
            for model in available_models:
                if model.id == model_hint.id or model == model_hint:
                    return model
        elif isinstance(model_hint, str):
            for model in available_models:
                identifiers = {normalize_id(model.name), str(model.id)}
                logger.debug(f"get_model: identifiers: {identifiers}")
                if normalize_id(model_hint) in identifiers:
                    return model
    raise HTTPException(status_code=404, detail=f"Model {model_hint} not found")


async def detect(
    _model_hints: List[str],
    images: List[UploadFile],
    return_image: bool = False,
) -> List[Optional[DetectionResults]]:
    available_models = get_global_config().available_models
    _model_hints = [normalize_id(model_hint) for model_hint in _model_hints]
    logger.debug(f"threaded_detect: model_hints -> {_model_hints}")
    detectors: List[APIDetector] = []
    found_models = []
    for model in available_models:
        identifiers = {model.name, str(model.id)}
        if any(
            [normalize_id(model_hint) in identifiers for model_hint in _model_hints]
        ):
            detector = get_global_config().get_detector(model)
            detectors.append(detector)
            found_models.append(f"<'{model.name}' ({model.id})>")
    logger.info(f"Found models: {found_models}")
    images = [load_image_into_numpy_array(await image.read()) for image in images]
    detections: List[List[DetectionResults]] = []
    # logger.info(f"Detectors ({len(Detectors)}) -> {Detectors}")
    #### async
    for image in images:
        img_dets = []
        for detector in detectors:
            # logger.info(f"Starting detection for {detector}")
            img_dets.append(await detector.detect(image))
        detections.append(img_dets)
    ##### ThreadPool
    # futures = []
    # import concurrent.futures
    #
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     for image in images:
    #         for detector in detectors:
    #             # logger.info(f"Starting detection for {detector}")
    #             futures.append(executor.submit(detector.detect, image))
    # for future in futures:
    #     detections.append(future.result())
    # logger.info(f"{LP} ThreadPool detections -> {detections}")
    if return_image:
        return detections, images
    return detections


def load_image_into_numpy_array(data: bytes) -> np.ndarray:
    """Load an uploaded image into a numpy array

    :param data: The image data in bytes
    :return: A cv2 imdecoded numpy array
    """
    np_img = np.frombuffer(data, np.uint8)
    if np_img is None:
        raise RuntimeError("Failed to create numpy array from image data")
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    return frame


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def docs():
    return RedirectResponse(url="/docs")


@app.get("/models/available/all", summary="Get a list of all available models")
async def available_models_all():
    """FastAPI endpoint. Get a list of all available models"""
    try:
        logger.debug(f"About to try and grab available models....")
        x = {"models": get_global_config().available_models}
    except Exception as e:
        logger.error(f"ERROR: {e}", exc_info=True)
        raise e
    else:
        logger.debug(f"Got available models: {x}")
        return x


@app.get(
    "/models/available/type/{model_type}",
    summary="Get a list of available models based on the type",
)
async def available_models_type(model_type: ModelType):
    """FastAPI endpoint. Get a list of available models based on the model type: obj, face, alpr"""
    available_models = get_global_config().available_models
    return {
        "models": [
            model.model_dump()
            for model in available_models
            if model.type_of == model_type
        ]
    }


@app.get(
    "/models/available/proc/{processor}",
    summary="Get a list of available models based on the processor",
)
async def available_models_proc(processor: ModelProcessor):
    """FastAPI endpoint. Get a list of available models based on the processor: cpu, gpu, tpu, none"""
    logger.info(f"available_models_proc: {processor}")
    available_models = get_global_config().available_models
    return {
        "models": [
            model.model_dump()
            for model in available_models
            if model.processor == processor
        ]
    }


@app.get(
    "/models/available/framework/{framework}",
    summary="Get a list of available models based on the framework",
)
async def available_models_framework(framework: ModelFrameWork):
    """FastAPI endpoint. Get a list of available models based on the framework: torch, ort, trt, opencv, etc"""
    logger.info(f"available_models_proc: {framework}")
    available_models = get_global_config().available_models
    return {
        "models": [
            model.model_dump()
            for model in available_models
            if model.framework == framework
        ]
    }


@app.post(
    "/annotate",
    # Set what the media type will be in the autogenerated OpenAPI specification.
    # https://fastapi.tiangolo.com/advanced/additional-responses/#additional-media-types-for-the-main-response
    responses={200: {"content": {"image/jpeg": {}}}},
    # Prevent FastAPI from adding "application/json" as an additional
    # response media type in the autogenerated OpenAPI specification.
    # https://github.com/tiangolo/fastapi/issues/3258
    response_class=Response,
    summary="Return an annotated jpeg image with bounding boxes and labels",
)
async def get_annotated_image(
    user_obj=Depends(verify_token),
    hints_: List[str] = Body(
        ..., example="yolov4, yolo-nas-s trt", description="A list of model names/UUIDs"
    ),
    image: UploadFile = File(...),
):
    """FastAPI endpoint. Return an annotated jpeg image with bounding boxes and labels"""
    logger.info(f"get_annotated_image: model hints (before) = {hints_}")
    if hints_:
        from ..Shared.Models.config import DetectionResults, Result

        hints_ = [x.strip('"').strip("'").strip() for x in hints_]
        if len(hints_) == 1:
            split_hints = hints_[0].split(",")
            if len(split_hints) > 1:
                hints_ = split_hints
        logger.info(f"get_annotated_image: model hints (AFTER) = {hints_}")
        detections: List[List[DetectionResults]]
        detections, images = await detect(hints_, [image], return_image=True)
        i = 0
        SLATE_COLORS: List[Tuple[int, int, int]] = [
            (39, 174, 96),
            (142, 68, 173),
            (0, 129, 254),
            (254, 60, 113),
            (243, 134, 48),
            (91, 177, 47),
        ]

        image_buffer = b""
        img_loop = 0
        num_dets = len(detections)
        logger.debug(f"DBG>>> ANNOTATE: num_dets: {num_dets} ---- {detections = }")
        for img_detection, _image in zip(detections, images):
            img_loop += 1
            for detection in img_detection:
                i += 1
                logger.debug(f"DBG>>> ANNOTATE:detection: {detection} ({i}/{num_dets})")
                if detection and isinstance(detection, DetectionResults):
                    rand_color = random.randrange(len(SLATE_COLORS) - 1)
                    rand_color2 = random.randrange(len(SLATE_COLORS) - 1)
                    logger.debug(
                        f"DBG>>> ANNOTATE: colors chosen randomly: {rand_color} {rand_color2}"
                    )
                    logger.debug(
                        f"DBG>>> ANNOTATE:detection: is a DetectionResults object, num results: {len(detection.results)}"
                    )

                    det: Result
                    for det in detection.results:
                        x1, y1, x2, y2 = det.bounding_box
                        cv2.rectangle(_image, (x1, y1), (x2, y2), rand_color, 2)
                        cv2.putText(
                            _image,
                            f"{det.label} ({det.confidence:.2f})[{detection.name}]",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            rand_color2,
                            2,
                        )

                elif detection and not isinstance(detection, DetectionResults):
                    raise RuntimeError(
                        f"DetectionResults object expected, got {type(detection)}"
                    )
                else:
                    logger.warning(f"DetectionResults object is {type(detection)}")
                is_success, img_buffer = cv2.imencode(".jpg", _image)
                if not is_success:
                    raise RuntimeError("Failed to encode image")
                image_buffer = img_buffer.tobytes()

        # media_type here sets the media type of the actual response sent to the client.
        # Return the results in a header, this allows Swagger UI to display the image
        import json

        results_json = []
        for img_det in detections:
            for _det in img_det:
                results_json.append(_det.model_dump())
        return Response(
            content=image_buffer,
            media_type="image/jpeg",
            headers={"Results": json.dumps(results_json)},
            background=None,
        )


@app.post(
    "/annotate2",
    # Set what the media type will be in the autogenerated OpenAPI specification.
    # https://fastapi.tiangolo.com/advanced/additional-responses/#additional-media-types-for-the-main-response
    responses={200: {"content": {"image/jpeg": {}}}},
    # Prevent FastAPI from adding "application/json" as an additional
    # response media type in the autogenerated OpenAPI specification.
    # https://github.com/tiangolo/fastapi/issues/3258
    response_class=StreamingResponse,
    summary="Return an annotated jpeg image with bounding boxes and labels",
)
async def get_annotated_image2(
    user_obj=Depends(verify_token),
    hints_: List[str] = Body(
        ..., example="yolov4, yolo-nas-s trt", description="A list of model names/UUIDs"
    ),
    images: List[UploadFile] = File(...),
):
    """FastAPI endpoint. EXPERIMENT!!!

    Return a List of annotated jpeg image with bounding boxes and labels. Results are returned in a header
    """
    logger.info(f"get_annotated_image: model hints (before) = {hints_}")
    if hints_:
        from ..Shared.Models.config import DetectionResults, Result

        hints_ = [x.strip('"').strip("'").strip() for x in hints_]
        if len(hints_) == 1:
            split_hints = hints_[0].split(",")
            if len(split_hints) > 1:
                hints_ = split_hints
        logger.info(f"get_annotated_image: model hints (AFTER) = {hints_}")
        detections: List[List[DetectionResults]]
        detections, images = await detect(hints_, images, return_image=True)
        i = 0
        SLATE_COLORS: List[Tuple[int, int, int]] = [
            (39, 174, 96),
            (142, 68, 173),
            (0, 129, 254),
            (254, 60, 113),
            (243, 134, 48),
            (91, 177, 47),
        ]

        image_buffer = b""
        img_loop = 0
        num_dets = len(detections)
        logger.debug(f"DBG>>> ANNOTATE: num_dets: {num_dets} ---- {detections = }")
        for img_detection, _image in zip(detections, images):
            img_loop += 1
            for detection in img_detection:
                i += 1
                logger.debug(f"DBG>>> ANNOTATE:detection: {detection} ({i}/{num_dets})")
                if detection and isinstance(detection, DetectionResults):
                    rand_color = random.randrange(len(SLATE_COLORS) - 1)
                    rand_color2 = random.randrange(len(SLATE_COLORS) - 1)
                    logger.debug(
                        f"DBG>>> ANNOTATE:detection: is a DetectionResults object, num results: {len(detection.results)}"
                    )

                    det: Result
                    for det in detection.results:
                        x1, y1, x2, y2 = det.bounding_box
                        cv2.rectangle(_image, (x1, y1), (x2, y2), rand_color, 2)
                        cv2.putText(
                            _image,
                            f"{det.label} ({det.confidence:.2f})[{detection.name}]",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            rand_color2,
                            2,
                        )

                elif detection and not isinstance(detection, DetectionResults):
                    raise RuntimeError(
                        f"DetectionResults object expected, got {type(detection)}"
                    )
                else:
                    logger.warning(f"DetectionResults object is {type(detection)}")
                is_success, img_buffer = cv2.imencode(".jpg", _image)
                if not is_success:
                    raise RuntimeError("Failed to encode image")
                image_buffer.append(img_buffer.tobytes())

        # media_type here sets the media type of the actual response sent to the client.
        # Return the results in a header, this allows Swagger UI to display the image

        results_json = []
        for img_det in detections:
            for _det in img_det:
                results_json.append(_det.model_dump())

        async def streamer(_images):
            from io import BytesIO

            for image_bytes in _images:
                bio = BytesIO()
                bytes_written = bio.write(image_bytes)
                logger.debug(
                    f"DBG>>> ANNOTATE: bytes_written to BytesIO buffer: {bytes_written}"
                )
                yield bio.getvalue()

        return StreamingResponse(
            streamer(image_buffer),
            headers={"Results": json.dumps(results_json)},
            media_type="image/jpeg",
            background=None,
        )
        return StreamingResponse(
            content=image_buffer,
            media_type="image/jpeg",
            headers={"Results": json.dumps(results_json)},
            background=None,
        )


@app.post(
    "/detect",
    summary="Detect objects in images using a set of threaded models referenced by name/UUID",
)
async def run_detection(
    user_obj=Depends(verify_token),
    hints_model: List[str] = Body(
        ...,
        description="List of model names/UUIDs",
        example="yolov4,97acd7d4-270c-4667-9d56-910e1510e8e8,yolov7 tiny",
    ),
    images: List[UploadFile] = File(..., description="Images to run the ML model on"),
):
    """FastAPI endpoint. Detect objects in images using a set of threaded models referenced by name/UUID"""
    start = time.perf_counter()
    if not images:
        raise HTTPException(status_code=400, detail="No image(s) provided")
    if hints_model:
        if isinstance(hints_model, str):
            # decode from json
            if hints_model.startswith('['):
                hints_model = json.loads(hints_model)
        elif isinstance(hints_model, list):
            if len(hints_model):
                if hints_model[0].startswith("["):
                    hints_model = json.loads(hints_model[0])
                else:
                    hints_model = [
                        hint.strip("'").strip('"').strip().lstrip("[").rstrip("]")
                        for hint in hints_model
                    ]

        if len(hints_model) == 1:
            split_hints = hints_model[0].split(",")
            logger.debug(f"{split_hints = }")
            if len(split_hints) > 0:
                hints_model = split_hints
        logger.debug(f"FINAL: {hints_model = }")
        detections = await detect(hints_model, images)
        logger.debug(f"{LP}DBG>>> /detect results: {detections}")
    else:
        detections = {"error": "No models specified"}
    end = time.perf_counter()
    logger.info(f"perf:{LP} /detect took {end - start:.5f} s")
    return detections


class MLAPI:
    cached_settings: Settings
    cfg_file: Path
    server: uvicorn.Server
    color_detector: Any

    def __init__(self, cfg_file: Union[str, Path], run_server: bool = False):
        """
        Initialize the FastAPI MLAPI server object, read a supplied YAML config file, and start the server if requested.
        :param cfg_file: The config file to read.
        :param run_server: Start the server after initialization.
        """
        create_global_config()
        if not isinstance(cfg_file, (str, Path)):
            raise TypeError(
                f"The YAML config file must be a str or pathlib.Path object, not {type(cfg_file)}"
            )
        # test that the file exists and is a file
        self.cfg_file = Path(cfg_file)
        if not self.cfg_file.exists():
            raise FileNotFoundError(f"'{self.cfg_file.as_posix()}' does not exist")
        elif not self.cfg_file.is_file():
            raise TypeError(f"'{self.cfg_file.as_posix()}' is not a file")
        self.read_settings()
        if run_server:
            self.start()

    def read_settings(self) -> Settings:
        """Read the YAML config file,  initialize the user_db, set the global config (g.config) object and load models"""
        logger.info(f"reading settings from '{self.cfg_file.as_posix()}'")
        from .Models.config import parse_client_config_file

        self.cached_settings = parse_client_config_file(self.cfg_file)
        from .auth import UserDB

        get_global_config().user_db = UserDB(self.cached_settings.server.auth.db_file)
        get_global_config().config = self.cached_settings
        init_logs(self.cached_settings)
        logger.info(f"Starting to load models...")
        available_models = (
            get_global_config().available_models
        ) = self.cached_settings.available_models

        if available_models:
            timer = time.perf_counter()
            _method = "ThreadPool"

            # import uvloop
            # _method = "Async"
            # for model in available_models:
            #     loop = uvloop.new_event_loop()
            #     loop.run_until_complete(get_global_config().create_detector(model))

            futures = []
            with ThreadPoolExecutor() as executor:
                for model in available_models:
                    futures.append(
                        executor.submit(get_global_config().create_detector, model)
                    )
            for future in futures:
                future.result()
            logger.info(
                f"perf:{LP} Loading '{_method}' models took {time.perf_counter() - timer:.5f} s"
            )
        else:
            logger.warning(f"No models found in config file!")

        return self.cached_settings

    def restart(self):
        self.server.shutdown()
        self.read_settings()
        self.start()

    def _get_cf_ip_list(self) -> List[str]:
        """Grab cloudflare ip ranges for IPv4/v6 and store them in a List of strings"""
        import requests

        cf_ipv4_url = "https://www.cloudflare.com/ips-v4"
        cf_ipv6_url = "https://www.cloudflare.com/ips-v6"
        cf_ipv4 = requests.get(cf_ipv4_url).text.splitlines()
        cf_ipv6 = requests.get(cf_ipv6_url).text.splitlines()
        cf_ips = cf_ipv4 + cf_ipv6
        return cf_ips

    def _breakdown_subnets(self, ip_list: List[str]) -> List[str]:
        """Breakdown a list of IP ranges into individual IPs

        gunicorn < 22.0 does not support subnets in forwarded_allow_ips"""
        ips = []
        if ip_list:
            logger.debug(f"Breaking down subnets: {ip_list}")
            from ipaddress import ip_network

            for ip in ip_list:
                if "/" in ip:
                    logger.debug(f"Breaking down subnet: {ip}")
                    _append = [str(ip) for ip in ip_network(ip).hosts()]
                    logger.debug(f"Subnet breakdown: {_append}")
                    ips += _append
                else:
                    ips.append(ip)

        return ips

    def start(self):
        _avail = {}
        forward_ips = self.cached_settings.uvicorn.forwarded_allow_ips
        forwarded_allow = []
        if forward_ips:
            forwarded_allow = [str(x) for x in forward_ips if x]
            logger.debug(f"Added {len(forwarded_allow)} forwarded_allow hosts")
        if self.cached_settings.uvicorn.grab_cloudflare_ips:
            forwarded_allow += self._get_cf_ip_list()
            logger.debug(
                f"Grabbed Cloudflare IP ranges to append to forwarded_allow hosts"
            )
        _proxy_headers = get_global_config().config.uvicorn.proxy_headers
        # logger.debug(f"\n\nforwarded_allow: {forwarded_allow}\n\n{_proxy_headers = }\n")
        for model in get_global_config().available_models:
            _avail[normalize_id(model.name)] = str(model.id)
        logger.info(f"AVAILABLE MODELS! --> {_avail}")
        server_cfg = get_global_config().config.server
        logger.debug(f"Server Config: {server_cfg}")
        # This takes forever when breaking down cloudflare ip subnets.
        # Wait for
        # forwarded_allow = self._breakdown_subnets(forwarded_allow)
        config = uvicorn.Config(
            app="zm_ml.Server.app:app",
            host=str(server_cfg.address),
            port=server_cfg.port,
            forwarded_allow_ips=forwarded_allow,
            log_config={
                "version": 1,
                "disable_existing_loggers": False,
            },
            log_level="debug",
            proxy_headers=_proxy_headers,
            # reload=False,
            # reload_dirs=[
            #     str(self.cfg_file.parent.parent / "src/zm_ml/Server"),
            #     str(self.cfg_file.parent.parent / "src/zm_ml/Shared"),
            # ],
        )
        self.server = uvicorn.Server(config=config)
        try:
            import uvloop

            loop = uvloop.new_event_loop()
            self.server.run()
        except KeyboardInterrupt:
            logger.info("Keyboard Interrupt, shutting down")
        except BrokenPipeError:
            logger.info("Broken Pipe, shutting down")
        except Exception as e:
            logger.exception(f"Shutting down because of Exception: {e}")
        finally:
            logger.info("Shutting down cleanly in finally: logic")
            loop.run_until_complete(self.server.shutdown())


# AUTH STUFF
@app.post(
    "/login",
    response_model=Token,
    summary="Login to get an authentication token",
)
async def login_for_access_token(
    request: Request, form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
):
    """FastAPI endpoint. Login to get an authentication token"""

    logger.debug(
        f"{LP} Access token requested by: {form_data.username} using IP: {request.client.host}"
    )

    user = get_global_config().user_db.authenticate_user(
        form_data.username, form_data.password
    )

    if not user:
        raise credentials_exception
    access_token = create_access_token(data={"sub": user.username})

    return access_token
