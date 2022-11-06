import json
import logging
import logging.handlers
import os
import sys
from decimal import Decimal
from hashlib import new
from pathlib import Path
from time import perf_counter
from typing import Union, Dict, Optional, List

__version__ = "0.0.1"
__version_type__ = "dev"

import requests
import requests_toolbelt
import yaml
from pydantic import BaseModel, Field
import cv2
import numpy as np

from Media import APIImagePipeLine, SHMImagePipeLine, ZMUImagePipeLine
from api import ZMApi
from config import ConfigFileModel, MLAPIRoute, Testing
from zmdb import ZMDB

logger = logging.getLogger("ZM-ML")
formatter = logging.Formatter(
    "%(asctime)s.%(msecs)04d %(name)s[%(process)s] %(levelname)s %(module)s:%(lineno)d -> %(message)s",
    "%m/%d/%y %H:%M:%S",
)
stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)


class GlobalConfig(BaseModel):
    api: ZMApi = Field(None)
    mid: int = None
    config: ConfigFileModel = None
    config_file: Union[str, Path] = None
    configs_path: Union[str, Path] = None
    eid: int = None
    mon_name: str = None
    mon_post: int = None
    mon_pre: int = None
    mon_fps: Decimal = None
    reason: str = None
    notes: str = None
    event_path: Path = None
    event_cause: str = None
    past_event: bool = False
    Event: Dict = None
    Monitor: Dict = None
    Frame: List = None
    mon_image_buffer_count: int = None
    mon_width: int = None
    mon_height: int = None
    mon_colorspace: int = None
    frame_buffer: list = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


g = GlobalConfig()


def get_global_config() -> GlobalConfig:
    return g


class ZMClient:
    config_file: Union[str, Path]
    config_hash: str
    raw_config: str
    parsed_cfg: Dict
    config: ConfigFileModel
    api: ZMApi
    db: ZMDB
    routes: List[MLAPIRoute]
    mid: int
    eid: int
    image_pipeline: Union[APIImagePipeLine, SHMImagePipeLine, ZMUImagePipeLine]

    def get_hash(
        self,
        read_chunk_size: int = 65536,
        algorithm: str = "sha256",
    ):
        """Hash a file using hashlib.
        Default algorithm is SHA-256

        :param int read_chunk_size: Maximum number of bytes to be read from the file
         at once. Default is 65536 bytes or 64KB
        :param str algorithm: The hash algorithm name to use. For example, 'md5',
         'sha256', 'sha512' and so on. Default is 'sha256'. Refer to
         hashlib.algorithms_available for available algorithms
        """

        lp: str = "conf:hash:"
        checksum = new(algorithm)  # Raises appropriate exceptions.
        self.config_hash = ""
        try:
            with self.config_file.open("rb") as f:
                for chunk in iter(lambda: f.read(read_chunk_size), b""):
                    checksum.update(chunk)
        except Exception as exc:
            logger.warning(
                f"{lp} ERROR while computing {algorithm} hash of "
                f"'{self.config_file.as_posix()}' -> {exc}"
            )
            raise
        else:
            self.config_hash = checksum.hexdigest()
            logger.debug(
                f"{lp} the {algorithm} hex digest for file '{self.config_file.as_posix()}' -> {self.config_hash}"
            )
        return self.config_hash

    @staticmethod
    def compare_hash(hash1: str, hash2: str) -> bool:
        """Compare two hashes"""
        if hash1 == hash2:
            return True
        else:
            return False

    def __init__(self, cfg_file: Union[str, Path]):
        """
        Initialize the ZMDetect class
        :param cfg_file: Path to the config file
        """
        if isinstance(cfg_file, (str, Path)):
            g.config_file = self.config_file = Path(cfg_file)
        else:
            raise TypeError("cfg_file must be a str or Path object")
        g.config = self.config = self.load_config()
        self.sort_routes()

        from concurrent.futures import ThreadPoolExecutor

        tpe = ThreadPoolExecutor()
        futures = []
        with tpe as executor:
            futures.append(executor.submit(self.init_logs))
            futures.append(executor.submit(self.init_db))
            futures.append(executor.submit(self.init_api))
        for future in futures:
            future.result()

    def sort_routes(self):
        self.routes = self.config.mlapi.routes
        if len(self.routes) > 1:
            logger.debug(f"Routes: BEFORE sorting >> {self.routes}")
            self.routes.sort(key=lambda x: x.weight)
            logger.debug(f"Routes: AFTER sorting >> {self.routes}")

    def init_logs(self):
        """Initialize the logging system."""
        level = self.config.logging.level
        level = level.casefold()
        if level == "debug":
            level = logging.DEBUG
        elif level == "warning":
            level = logging.WARNING
        elif level == "error":
            level = logging.ERROR
        elif level == "critical":
            level = logging.CRITICAL
        else:
            level = logging.INFO
        logger.setLevel(level)

        if level == logging.DEBUG:
            logger.debug(f"Logging level set to DEBUG")
        elif self.config.logging.console is True:
            pass
        else:
            logger.removeHandler(stream_handler)
        file_from_config = self.config.logging.dir / self.config.logging.file_name
        if os.access(self.config.logging.dir.as_posix(), os.W_OK):
            file_from_config.touch(exist_ok=True)
            # ZM /var/log/zm is handled by logrotate
            file_handler = logging.FileHandler(file_from_config.as_posix(), mode="a")
            # file_handler = logging.handlers.TimedRotatingFileHandler(
            #     file_from_config, when="midnight", interval=1, backupCount=7
            # )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            # Get user and group id and name
            import getpass
            import grp

            uname = getpass.getuser()
            gname = grp.getgrgid(os.getgid()).gr_name
            uid = os.getuid()
            gid = os.getgid()
            logger.debug(
                f"Logging to file '{file_from_config}' with user: "
                f"{uid} [{uname}] group: {gid} [{gname}] enabled"
            )
        else:
            logger.warning(
                f"Logging to file {file_from_config} disabled due to permissions"
                f" [No write access to {file_from_config.parent.as_posix()}]"
            )
        logger.info(f"Logging initialized...")

    def init_db(self):
        logger.debug("Initializing DB")
        self.db = ZMDB()
        logger.debug("DB initialized...")

    def get_db_data(self, eid: int):
        """Get data from the database"""
        global g
        g.eid = eid
        (
            mid,
            mon_name,
            mon_post,
            mon_pre,
            mon_fps,
            reason,
            event_path,
        ) = self.db.grab_all(g.eid)
        g.mid = mid
        g.mon_name = mon_name
        g.mon_post = mon_post
        g.mon_pre = mon_pre
        g.mon_fps = mon_fps
        g.reason = reason
        g.event_path = event_path

    def init_api(self):
        g.api = self.api = ZMApi(self.config.zoneminder)
        g.api.import_zones()

    def detect(self, eid: int, mid: int):
        global g
        g.mid = mid
        logger.info(f"Running detection for event {eid}, obtaining monitor info...")
        self.get_db_data(eid)
        # get monitor and event info
        monitor_data = self.api.get_monitor_data(mid)
        event_data, event_monitor_data, frame_data = self.api.get_all_event_data(eid)
        # logger.debug(
        #     f"event_monitor_data and monitor_data SAME? :>> {event_monitor_data == monitor_data}"
        # )
        # logger.debug(f"Monitor data: {monitor_data}")
        # logger.debug(f"Event data: {event_data}")
        # if event_monitor_data != monitor_data:
        #     logger.debug(f"Event monitor data: {event_monitor_data}")
        # logger.debug(f"Frame data: {frame_data}")
        # get image and image name
        # FIXME: did __iter__ work?
        # init Image Pipeline
        how = self.config.detection_settings.images.pull_method
        logger.debug(f"DBG>>> Image pull methods: {how}")
        if how.shm is True:
            logger.debug("Using SHM for image source")
            # self.image_pipeline = SHMImagePipeLine()
        elif how.api.enabled is True:
            logger.debug("Using ZM API for image source")
            self.image_pipeline = APIImagePipeLine(how.api)
        elif how.zmu is True:
            logger.debug("Using CLI 'zmu' for image source")
            pass
            # self.image_pipeline = ZMUImagePipeLine()

        models: Optional[Dict] = None
        if g.mid in self.config.monitors:
            models = self.config.monitors.get(g.mid).models
        if not models:
            if self.config.detection_settings.models:
                models = self.config.detection_settings.models
            else:
                models = {"yolov4": {}}
        else:
            _models = self.config.detection_settings.models
            # dont duplicate Models
            models = {**_models, **models}
        model_names = list(models.keys())
        models_str = ",".join(model_names)
        logger.debug(f"{model_names = } --- {models_str = }")

        final_detections: dict = {}
        detections: dict = {}
        _routes = {}
        _start_all = perf_counter()
        # for image, image_name in self.image_pipeline:
        i = 0
        while self.image_pipeline.is_image_stream_active():
            i += 1
            image, image_name = self.image_pipeline.get_image()

            if image is None:
                logger.warning(f"No image returned! trying again...")
                continue
            elif image is False:
                logger.warning(f"Image source exhausted! Checking if more() will stop itself")
                if i > 25:
                    logger.warning(f"More() did not stop itself! stopping after 25 loops")
                    break
                continue
            fid = image_name.split("fid_")[1].split(".")[0]
            if any([g.config.animation.gif.enabled, g.config.animation.mp4.enabled]):
                # Memory expensive?
                # TODO: Add option to write to /tmp for low memory applications
                g.frame_buffer.append((image, fid))
            for route in self.routes:
                if route.enabled:
                    logger.debug(f"MLAPI route '{route.name}' is enabled!")
                    url = f"{route.host}:{route.port}/detect/group"
                    fields = {
                        "model_hints": (
                            None,
                            json.dumps(models_str),
                            "application/json",
                        ),
                        "image": (image_name, image, "image/jpeg"),
                    }
                    multipart_data = (
                        requests_toolbelt.multipart.encoder.MultipartEncoder(
                            fields=fields
                        )
                    )
                    headers = {
                        "Accept": "application/json",
                        "Content-Type": multipart_data.content_type,
                    }
                    if route.name in _routes:
                        url = _routes[route.name]
                    else:
                        # regex for http(s)://
                        import re

                        if re.match(r"^(http(s)?)://", url):
                            logger.debug(f"route.host is valid with schema: {url}")
                        else:
                            logger.debug(
                                f"No http(s):// in route.host, assuming http://"
                            )
                            url = f"http://{url}"
                        _routes[route.name] = url
                    logger.debug(f"Sending image to '{route.name}' @ {url}")
                    _perf = perf_counter()
                    r = requests.post(
                        url,
                        data=multipart_data,
                        headers=headers,
                        timeout=route.timeout,
                    )
                    r.raise_for_status()
                    _end = perf_counter() - _perf
                    if fid in detections:
                        detections[fid].update(r.json())
                    else:
                        detections[fid] = r.json()
                    if route.name in final_detections:
                        final_detections[route.name].update(detections)
                    else:
                        final_detections[route.name] = detections
                    logger.debug(
                        f"perf:: HTTP Detection request to '{route.name}' completed in {_end:.5f} seconds"
                    )
                else:
                    logger.warning(f"Neo-MLAPI route '{route.name}' is disabled!")

        _end_all = perf_counter() - _start_all
        logger.debug(f"perf:: Total detections time {_end_all:.5f} seconds")
        logger.info(f"After all images THESE ARE FINAL detections: {final_detections}")
        # check if any successful detections
        # if self.check_detections(detections):
        #     # filter results
        #     detections = self.filter_detections(detections)
        #     # decide best match
        #     best_match = self.compute_best_match(detections)
        #     # post-processing (annotating, animations, notifications[mqtt, push])
        #     self.post_process(best_match, image, image_name)
        return final_detections

    def load_config(self) -> Optional[ConfigFileModel]:
        """Parse the YAML configuration file. In the future this will read DB values

        Args:
            cfg_file (Path): Configuration YAML file.

        """
        cfg: Dict = {}
        self.raw_config = self.config_file.read_text()

        try:
            cfg = yaml.safe_load(self.raw_config)
        except yaml.YAMLError as e:
            logger.error(
                f"model_config_parser: Error parsing the YAML configuration file!"
            )
            raise e
        substitutions = cfg.get("substitutions", {})
        testing = cfg.get("testing", {})
        testing = Testing(**testing)
        if testing.enabled:
            logger.info(f"|----- TESTING IS ENABLED! -----|")
            if testing.substitutions:
                logger.info(f"Overriding substitutions WITH testing:substitutions")
                substitutions = testing.substitutions

        logger.debug(f"Replacing ${{VARS}} in config:substitutions")
        substitutions = self.replace_vars(str(substitutions), substitutions)
        if inc_file := substitutions.get("IncludeFile"):
            inc_file = Path(inc_file)
            logger.debug(f"PARSING IncludeFile: {inc_file.as_posix()}")
            if inc_file.is_file():
                inc_vars = yaml.safe_load(inc_file.read_text())
                logger.debug(
                    f"model_config_parser: Loaded {len(inc_vars)} substitution from IncludeFile {inc_file}"
                )
                # check for duplicates
                for k in inc_vars:
                    if k in substitutions:
                        logger.warning(
                            f"model_config_parser: Duplicate substitution variable '{k}' in IncludeFile {inc_file} - "
                            f"IncludeFile overrides config file"
                        )

                substitutions.update(inc_vars)
            else:
                logger.warning(
                    f"model_config_parser: IncludeFile {inc_file} is not a file!"
                )

        logger.debug(f"Replacing ${{VARS}} in config")
        cfg = self.replace_vars(self.raw_config, substitutions)
        self.parsed_cfg = dict(cfg)
        logger.debug(f"Config file about to be validated using pydantic!")
        return ConfigFileModel(**cfg)

    @staticmethod
    def replace_vars(search_str: str, var_pool: Dict) -> Dict:
        """Replace variables in a string.


        Args:
            search_str (str): String to search for variables '${VAR_NAME}'.
            var_pool (Dict): Dictionary of variables used to replace.

        """
        import re

        if var_list := re.findall(r"\$\{(\w+)\}", search_str):
            # $ remove duplicates
            var_list = list(set(var_list))
            logger.debug(
                f"substitution_vars: Found the following substitution variables: {var_list}"
            )
            # substitute variables
            _known_vars = []
            _unknown_vars = []
            for var in var_list:
                if var in var_pool:
                    logger.debug(
                        f"substitution variable '{var}' IS IN THE POOL! VALUE: "
                        f"{var_pool[var]} [{type(var_pool[var])}]"
                    )
                    _known_vars.append(var)
                    value = var_pool[var]
                    if value is None:
                        value = ""
                    elif value is True:
                        value = "yes"
                    elif value is False:
                        value = "no"
                    search_str = search_str.replace(f"${{{var}}}", value)
                else:
                    _unknown_vars.append(var)
            if _unknown_vars:
                logger.warning(
                    f"substitution_vars: The following variables have no configured substitution value: {_unknown_vars}"
                )
            if _known_vars:
                logger.debug(
                    f"substitution_vars: The following variables have been substituted: {_known_vars}"
                )
        else:
            logger.debug(f"substitution_vars: No substitution variables found.")

        return yaml.safe_load(search_str)

    def check_detections(self, detections: Dict) -> bool:
        """Check if any detections were successful"""
        _ret = False

        for route_name, detection in detections.items():
            if detection["success"]:
                _ret = True
                break
        return _ret

    def filter_detections(self, detections: Dict) -> Dict:
        """Filter detections"""
        """
                    { "route.name": 
                        {
                        "success": True if labels else False,
                        "type": self.config.model_type,
                        "processor": self.processor,
                        "model_name": self.name,
                        "label": labels,
                        "confidence": confs,
                        "bounding_box": b_boxes,
                        },
                    **repeat for each route** 
                    }
                """
        monitor_ = self.config.monitors.get(g.mid)
        for route_name, _detections in detections.items():
            _detections: list
            for _detection in _detections:
                if not _detection["success"]:
                    continue
                model_type = _detection["type"]
                processor = _detection["processor"]
                model_name = _detection["model_name"]
                labels = _detection["label"]
                confs = _detection["confidence"]
                b_boxes = _detection["bounding_box"]
                _det = {}
                for label, conf, bbox in zip(labels, confs, b_boxes):
                    _det[label] = {"conf": conf, "bbox": bbox}

        return detections

    def compute_best_match(self, detections: Dict) -> Dict:
        """Compute best match"""
        return {}

    def post_process(self, best_match: Dict, image: bytearray, image_name: str):
        """Post-processing"""
        logger.debug(f"in post_process() -> best_match: {best_match}")
        # convert bytearray to cv2 image
        image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        # annotate image
        image = self.annotate_image(best_match, image)
        # create animations
        self.create_animations(best_match, image, image_name)
        # send notifications
        self.send_notifications(best_match, image, image_name)

    def send_notifications(self, best_match: Dict, image: bytearray, image_name: str):
        """Send notifications"""
        # TODO: mqtt, push, email, webhook
        pass

    def create_animations(self, best_match, image, image_name):
        """Create animations"""
        # TODO: create animations
        pass

    def annotate_image(self, best_match: Dict, image: np.ndarray) -> np.ndarray:
        """Annotate image"""

        # TODO draw bounding boxes, label, conf, model, proc
        return image
