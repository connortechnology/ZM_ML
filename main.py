import logging
import logging.handlers
import sys
from decimal import Decimal
from pathlib import Path
from typing import Union, Dict, Optional, List

__version__ = "0.0.1"
__version_type__ = "dev"

import requests
from pydantic import BaseModel
import cv2
import numpy as np

from Media import APIImagePipeLine, SHMImagePipeLine, ZMUImagePipeLine
from libs.api import ZMApi
from models.config import ConfigFileModel, MLAPIRoute
from zmdb import ZMDB

logger = logging.getLogger("zm_ml")
formatter = logging.Formatter(
    "%(asctime)s.%(msecs)04d %(name)s[%(process)s] %(levelname)s %(module)s:%(lineno)d -> %(message)s",
    "%m/%d/%y %H:%M:%S",
)


class GlobalConfig(BaseModel):
    api: ZMApi = None
    mid: int = None
    config: ConfigFileModel = None
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


g = GlobalConfig()


def get_global_config() -> GlobalConfig:
    return g


class ZMDetect:
    config_file: Union[str, Path]
    raw_config: str
    parsed_cfg: Dict
    config: ConfigFileModel
    api: ZMApi
    db: ZMDB
    routes: List[MLAPIRoute]
    mid: int
    eid: int
    image_pipeline: Union[APIImagePipeLine, SHMImagePipeLine, ZMUImagePipeLine]

    def __init__(self, cfg_file: Union[str, Path]):
        self.config_file = Path(cfg_file)
        g.config = self.config = self.load_config(self.config_file)
        self.sort_routes()
        how = self.config.detection_settings.images.pull_method
        if how.shm:
            self.image_pipeline = SHMImagePipeLine()
        elif how.api.enabled:
            self.image_pipeline = APIImagePipeLine(how.api)
        elif how.zmu:
            self.image_pipeline = ZMUImagePipeLine()
        from concurrent.futures import ThreadPoolExecutor

        tpe = ThreadPoolExecutor()
        with tpe as executor:
            executor.submit(self.init_logs)
            executor.submit(self.init_api)
            executor.submit(self.init_db)

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
        elif level == "info":
            level = logging.INFO
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
        file_from_config = self.config.logging.dir / self.config.logging.file_name
        # ZM /var/log/zm is handled by logrotate
        file_handler = logging.FileHandler(file_from_config)
        # file_handler = logging.handlers.TimedRotatingFileHandler(
        #     file_from_config, when="midnight", interval=1, backupCount=7
        # )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        if self.config.logging.console or level == logging.DEBUG:
            logger.debug("Logging to console enabled")
            stream_handler = logging.StreamHandler(stream=sys.stdout)
            stream_handler.setFormatter(formatter)
            stream_handler.setLevel(level)
            logger.addHandler(stream_handler)

    def init_db(self):
        self.db = ZMDB()

    def get_db_data(self, eid: int):
        """Get data from the database"""
        global g
        g.eid = eid
        mid, mon_name, mon_post, mon_pre, mon_fps, reason, event_path = self.db.grab_all(self.eid)
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

    def run(self, eid: int, mid: int):
        global g
        g.mid = mid
        logger.info(
            f"Running detection for event {eid}, obtaining monitor info..."
        )
        self.get_db_data(eid)
        # get monitor and event info
        monitor_data = self.api.get_monitor_data(mid)
        event_data, event_monitor_data, frame_data = self.api.get_all_event_data(eid)
        logger.debug(
            f"event_monitor_data and monitor_data SAME? :>> {event_monitor_data == monitor_data}"
        )
        logger.debug(f"Monitor data: {monitor_data}")
        logger.debug(f"Event data: {event_data}")
        if event_monitor_data != monitor_data:
            logger.debug(f"Event monitor data: {event_monitor_data}")
        logger.debug(f"Frame data: {frame_data}")
        # get image and image name
        # FIXME: did __iter__ work?
        logger.debug(f"DBG>> about to grab image... did __iter__ work?")
        for image, image_name in self.image_pipeline:
            # send to zm_mlapi and get results
            detections: Dict = self.get_detections(image, image_name)
            logger.info(f"Got detections: {detections}")
            # check if any successful detections
            if self.check_detections(detections):
                # filter results
                detections = self.filter_detections(detections)
                # decide best match
                best_match = self.compute_best_match(detections)
                # post-processing (annotating, animations, notifications[mqtt, push])
                self.post_process(best_match, image, image_name)

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

    def get_detections(self, image: bytearray, image_name: str):
        import requests_toolbelt

        # TODO add support for changing model options

        url_end = ''
        models = self.config.monitors.get(self.mid).models
        model_names = list[models.keys()]
        models_str = ",".join(model_names)
        url_end = 'group'
        # routes are sorted by weight already
        # if len(model_names) > 1:
        #     url_end = 'group'
        # else:
        #     url_end = f'single/{model_names[0]}'
        detections = {}
        for route in self.routes:
            if route.enabled:
                logger.info(f"Sending image to {route.name}")
                url = f"{route.host}:{route.port}/detect/group"
                fields = {
                    "model_hints": (None, models_str, "application/json"),
                    "image": (image_name, image, "image/jpeg"),
                }
                multipart_data = requests_toolbelt.multipart.encoder.MultipartEncoder(
                    fields=fields
                )
                headers = {
                    "Accept": "application/json",
                    "Content-Type": multipart_data.content_type,
                }
                r = requests.post(
                    url,
                    data=multipart_data,
                    headers=headers,
                )
                r.raise_for_status()
                detections[route.name] = r.json()
        return detections

    def load_config(self, cfg_file: Path) -> Optional[ConfigFileModel]:
        """Parse the YAML configuration file. In the future this will read DB values

        Args:
            cfg_file (Path): Configuration YAML file.

        """
        cfg: Dict = {}
        cfg_str = cfg_file.read_text()
        self.raw_config = str(cfg_str)
        import yaml

        try:
            cfg = yaml.safe_load(cfg_str)
        except yaml.YAMLError as e:
            logger.error(
                f"model_config_parser: Error parsing the YAML configuration file!"
            )
            raise e
        substitutions = cfg.get("substitutions", {})
        # IncludeFile
        if inc_file := substitutions.get("IncludeFile"):
            inc_file = Path(inc_file)
            # If not absolute, assume relative to the config file
            if not inc_file.is_absolute():
                logger.debug(
                    f"model_config_parser: Relative path to IncludeFile: {inc_file}"
                )
                inc_file = cfg_file.parent / inc_file
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

        # substitutions in substitutions section
        substitutions = self.replace_vars(str(substitutions), substitutions)
        cfg = self.replace_vars(cfg_str, substitutions)
        self.parsed_cfg = dict(cfg)
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
            var_list = [x for n, x in enumerate(var_list) if x not in var_list[:n]]
            logger.debug(
                f"substitution_vars: Found the following variables in the configuration file: {var_list}"
            )
            # substitute variables
            for var in var_list:
                num_var = len(re.findall(f"\${{{var}}}", search_str))
                if var in var_pool:
                    logger.debug(
                        f"substitution_vars: Found {num_var} occurrence{'s' if num_var != 1 else ''} of '${{{var}}}', "
                        f"Substituting with value '{var_pool[var]}'"
                    )
                    search_str = search_str.replace(f"${{{var}}}", var_pool[var])
                else:
                    logger.warning(
                        f"substitution_vars: The variable '${{{var}}}' has no configured substitution value."
                    )

        from ast import literal_eval

        return literal_eval(search_str)
