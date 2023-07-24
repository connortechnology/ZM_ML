from __future__ import annotations

import datetime
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from platform import python_version
from typing import Union, Dict, List, Optional, Any, TYPE_CHECKING

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
    Path as FastPath,
)
from fastapi.responses import RedirectResponse


from ..Shared.Models.Enums import ModelType, ModelFrameWork, ModelProcessor
from .Log import SERVER_LOGGER_NAME, SERVER_LOG_FORMAT
from zm_ml.Shared.Log.handlers import BufferedLogHandler
from .ML.Detectors.color_detector import ColorDetector

if TYPE_CHECKING:
    from ..Shared.Models.config import DetectionResults
    from .Models.config import (
        GlobalConfig,
        Settings,
        APIDetector,
    )
from .Models.config import BaseModelConfig

__version__ = "0.0.1a"
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

app = FastAPI(debug=True)
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
    logger.debug(f"{lp} Setting root logger level to {logging._levelToName[root_level]}")
    logger.setLevel(root_level)
    for _ul in uvi_loggers:
        _ul.setLevel(root_level)

    if cfg.console.enabled is False:
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler):
                logger.info(f"{lp} Removing console log output!")
                logger.removeHandler(h)

    if cfg.file.enabled:
        if cfg.file.file_name:
            _filename = cfg.file.file_name
        else:
            _filename = f"zmmlServer.log"
        abs_logfile = (cfg.file.path / _filename).expanduser().resolve()
        try:
            if not abs_logfile.exists():
                logger.info(f"{lp} Creating log file [{abs_logfile}]")
                abs_logfile.touch(exist_ok=True, mode=0o644)
            else:
                with abs_logfile.open("a") as f:
                    pass
        except PermissionError:
            logger.warning(
                f"{lp} Logging to file disabled due to permissions"
                f" - No write access to '{abs_logfile.as_posix()}' for user: "
                f"{sys_uid} [{sys_user}] group: {sys_gid} [{sys_group}]"
            )
        else:
            # todo: add timed rotating log file handler if configured
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
    model_hint: str,
    image,
):
    model_hint = normalize_id(model_hint)
    logger.info(f"{LP} detect: model hint {model_hint}")
    model: BaseModelConfig = get_model(model_hint)
    logger.info(f"{LP} found model {model.id} -> {model}")
    detector: APIDetector = get_global_config().get_detector(model)
    image = load_image_into_numpy_array(await image.read())
    timer = time.perf_counter()
    detection: Dict = detector.detect(image)
    logger.info(
        f"perf:{LP} single detection completed in {time.perf_counter() - timer:.5f} s -> {detection}"
    )
    return detection


async def threaded_detect(_model_hints: List[str], image) -> List[Optional[DetectionResults]]:
    available_models = get_global_config().available_models
    _model_hints = [normalize_id(model_hint) for model_hint in _model_hints]
    logger.debug(f"threaded_detect: model_hints -> {_model_hints}")
    detectors: List[APIDetector] = []
    for model in available_models:
        identifiers = {model.name, str(model.id)}
        if any([normalize_id(model_hint) in identifiers for model_hint in _model_hints]):
            logger.info(f"Found model: {model.name} ({model.id})")
            detector = get_global_config().get_detector(model)
            detectors.append(detector)
    image = load_image_into_numpy_array(await image.read())
    detections: List[Dict] = []
    # logger.info(f"Detectors ({len(Detectors)}) -> {Detectors}")
    futures = []
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for detector in detectors:
            # logger.info(f"Starting detection for {detector}")
            futures.append(executor.submit(detector.detect, image))
    for future in futures:
        detections.append(future.result())
    logger.info(f"{LP} ThreadPool detections -> {detections}")

    return detections


def load_image_into_numpy_array(data):
    """Load an uploaded image into a numpy array"""
    npimg = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    # cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def docs():
    return RedirectResponse(url="/docs")


@app.get("/models/available/all", summary="Get a list of all available models")
async def _available_models():
    return {"models": get_global_config().available_models}


@app.get(
    "/models/available/type/{model_type}",
    summary="Get a list of available models based on the type",
)
async def available_models_type(model_type: ModelType):
    available_models = get_global_config().available_models
    return {
        "models": [
            model.dict() for model in available_models if model.model_type == model_type
        ]
    }


@app.get(
    "/models/available/proc/{processor}",
    summary="Get a list of available models based on the processor",
)
async def available_models_proc(processor: ModelProcessor):
    logger.info(f"available_models_proc: {processor}")
    available_models = get_global_config().available_models
    return {
        "models": [
            model.dict() for model in available_models if model.processor == processor
        ]
    }


@app.get(
    "/models/available/framework/{framework}",
    summary="Get a list of available models based on the framework",
)
async def available_models_proc(framework: ModelFrameWork):
    logger.info(f"available_models_proc: {framework}")
    available_models = get_global_config().available_models
    return {
        "models": [
            model.dict() for model in available_models if model.framework == framework
        ]
    }

@app.post('/login')
def login():
    """Processes user's authentication and returns a token
    on successful authentication.

    request body:

    - username: Unique identifier for a user e.g email,
                phone number, name

    - password:
    """
    return "ThisTokenIsFake"

@app.post(
    "/detect/group",
    summary="Detect objects in an image using a set of threaded models referenced by name",
)
async def group_detect(
    hints_model: List[str] = Body(
        ...,
        description="comma seperated model names/UUIDs",
        example="yolov4,97acd7d4-270c-4667-9d56-910e1510e8e8,yolov7 tiny",
    ),
    image: UploadFile = File(...),
):
    logger.info(f"group_detect: {hints_model}")
    hints_model = hints_model[0].strip('"').split(",")
    detections = await threaded_detect(hints_model, image)

    det: DetectionResults
    for det in detections:
        det = det.json()
    return detections


@app.post(
    "/detect/single/{model_hint}",
    summary="Run detection using the specified model on a single image",
    # response_model=DetectionResult,
)
async def single_detection(
    model_hint: str = FastPath(..., description="model name or id", example="yolov4"),
    image: UploadFile = File(..., description="Image to run the ML model on"),
):
    logger.info(f"single_detection: ENDPOINT {model_hint}")
    return await detect(model_hint, image)


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

    def read_settings(self):
        logger.info(f"reading settings from '{self.cfg_file.as_posix()}'")
        from .Models.config import parse_client_config_file

        self.cached_settings = parse_client_config_file(self.cfg_file)
        get_global_config().config = self.cached_settings
        init_logs(self.cached_settings)
        logger.info(f"Starting to load models...")
        available_models = (
            get_global_config().available_models
        ) = self.cached_settings.available_models

        if available_models:
            futures = []
            timer = time.perf_counter()
            with ThreadPoolExecutor() as executor:
                for model in available_models:
                    futures.append(
                        executor.submit(get_global_config().get_detector, model)
                    )
            for future in futures:
                future.result()
            logger.info(
                f"perf:{LP} TOTAL ThreadPool loading models took {time.perf_counter() - timer:.5f} s"
            )
        else:
            logger.warning(f"No models found in config file!")

        return self.cached_settings

    def restart(self):
        self.server.shutdown()
        self.read_settings()
        self.start()

    def start(self):
        _avail = {}
        # logger.debug(f"\n\n{get_global_config().available_models = }\n\n")
        for model in get_global_config().available_models:
            _avail[normalize_id(model.name)] = str(model.id)
        logger.info(f"AVAILABLE MODELS! --> {_avail}")
        server_cfg = get_global_config().config.server
        logger.debug(f"Server Config: {server_cfg}")
        config = uvicorn.Config(
            app="zm_ml.Server.app:app",
            host=str(server_cfg.address),
            port=server_cfg.port,
            log_config={
                "version": 1,
                "disable_existing_loggers": False,
            },
            log_level="debug",
            proxy_headers=True,
            # forwarded_allow_ips="*",
            # reload=False,
            # reload_dirs=[
            #     str(self.cfg_file.parent.parent / "src/zm_ml/Server"),
            #     str(self.cfg_file.parent.parent / "src/zm_ml/Shared"),
            # ],
        )
        self.server = uvicorn.Server(config=config)
        try:
            self.server.run()
        except KeyboardInterrupt:
            logger.info("Keyboard Interrupt, shutting down")
            self.server.shutdown()
        except BrokenPipeError:
            logger.info("Broken Pipe, shutting down")
            self.server.shutdown()
        except Exception as e:
            logger.exception(f"Shutting down because of Exception: {e}")
            self.server.shutdown()
        finally:
            logger.info("Shutting down")
            self.server.shutdown()
