import copy
import json
import logging
import logging.handlers
import os
import pickle
import sys
import concurrent
from concurrent.futures import Future
from hashlib import new
from pathlib import Path
from shutil import which
from time import perf_counter, sleep, time
from typing import Union, Dict, Optional, List, Any, Tuple

import cv2
import numpy as np
import requests
import requests_toolbelt
import urllib3.exceptions
import yaml
from shapely.geometry import Polygon

from .Libs.Media import APIImagePipeLine, SHMImagePipeLine, ZMUImagePipeLine
from .Libs.api import ZMApi
from .Libs.zmdb import ZMDB
from .Models.config import (
    ConfigFileModel,
    ServerRoute,
    Testing,
    OverRideMatchFilters,
    MonitorZones,
    MatchFilters,
    OverRideStaticObjects,
    OverRideObjectFilters,
    OverRideFaceFilters,
    OverRideAlprFilters,
    MatchStrategy,
)
from ..Shared.configs import ClientEnvVars, GlobalConfig

__version__: str = "0.0.1"
__version_type__: str = "dev"
logger = logging.getLogger("ML-Client")

ZM_INSTALLED: Optional[str] = which("zmpkg.pl")
formatter = logging.Formatter(
    "%(asctime)s.%(msecs)04d %(name)s[%(process)s] %(levelname)s %(module)s:%(lineno)d -> %(message)s",
    "%m/%d/%y %H:%M:%S",
)
stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)

ENV_VARS: Optional[ClientEnvVars] = None
g: Optional[GlobalConfig] = None


def get_global_config() -> GlobalConfig:
    return g


def check_imports():
    try:
        import cv2

        maj, min_, patch = "", "", ""
        x = cv2.__version__.split(".")
        x_len = len(x)
        if x_len <= 2:
            maj, min_ = x
            patch = "0"
        elif x_len == 3:
            maj, min_, patch = x
            patch = patch.replace("-dev", "") or "0"
        else:
            logger.error(f'come and fix me again, cv2.__version__.split(".")={x}')

        cv_ver = int(maj + min_ + patch)
        if cv_ver < 420:
            logger.error(
                f"You are using OpenCV version {cv2.__version__} which does not support CUDA for DNNs. A minimum"
                f" of 4.2 is required. See https://medium.com/@baudneo/install-zoneminder-1-36-x-6dfab7d7afe7"
                f" on how to compile and install openCV 4.5.4 with CUDA"
            )
        del cv2
        try:
            import cv2.dnn
        except ImportError:
            logger.error(
                f"OpenCV does not have DNN support! If you installed from "
                f"pip you need to install 'opencv-contrib-python'. If you built from source, "
                f"you did not compile with CUDA/cuDNN"
            )
            raise
    except ImportError as e:
        logger.error(f"Missing OpenCV 4.2+ (4.5.4+ recommended): {e}")
        raise

    try:
        import numpy as np
    except ImportError as e:
        logger.error(f"Missing numpy: {e}")
        raise
    logger.debug("check imports:: All imports found!")


def _cfg2path(config_file: Union[str, Path]) -> Path:
    if config_file:
        if isinstance(config_file, (str, Path)):
            config_file = Path(config_file)
        else:
            raise TypeError(
                f"config_file must be a string or Path, not {type(config_file)}"
            )
        return Path(config_file)


def static_pickle(
    labels: Optional[List[str]] = None,
    confs: Optional[List] = None,
    bboxs: Optional[List] = None,
    write: bool = False,
) -> Optional[Tuple[List[str], List, List]]:
    """Use the pickle module to save a python data structure to a file

    :param write: save the data to a file
    :param bboxs: list of bounding boxes
    :param confs: list of confidence scores
    :param labels: list of labels
    """
    lp: str = "pickle_static_objects:"
    variable_data_path = g.config.system.variable_data_path
    mon_file = Path(f"{variable_data_path}/mid-{g.mid}_past-detection.pkl")
    logger.debug(f"{lp} mon_file[{type(mon_file)}]={mon_file}")

    if not write:
        logger.debug(
            f"{lp} trying to load previous detection results from file: '{mon_file}'"
        )
        if mon_file.exists():
            try:
                with mon_file.open("rb") as f:
                    labels = pickle.load(f)
                    confs = pickle.load(f)
                    bboxs = pickle.load(f)
            except FileNotFoundError:
                logger.debug(f"{lp}  no history data file found for monitor '{g.mid}'")
            except EOFError:
                logger.debug(f"{lp}  empty file found for monitor '{g.mid}'")
                logger.debug(f"{lp}  going to remove '{mon_file}'")
                try:
                    mon_file.unlink()
                except Exception as e:
                    logger.error(f"{lp}  could not delete: {e}")
            except Exception as e:
                logger.error(f"{lp} error: {e}")
            logger.debug(f"{lp} returning results: {labels}, {confs}, {bboxs}")
        else:
            logger.warning(f"{lp} no history data file found for monitor '{g.mid}'")
    else:
        try:
            mon_file.touch(exist_ok=True)
            with mon_file.open("wb") as f:
                pickle.dump(labels, f)
                pickle.dump(confs, f)
                pickle.dump(bboxs, f)
                logger.debug(
                    f"{lp} saved_event: {g.eid} RESULTS to file: '{mon_file}' ::: {labels}, {confs}, {bboxs}",
                )
        except Exception as e:
            logger.error(
                f"{lp}  error writing to '{mon_file}' past detections not recorded, err msg -> {e}"
            )
    return labels, confs, bboxs


class CFGHash:
    previous_hash: str
    config_file: Path
    hash: str

    def __init__(self, config_file: Union[str, Path, None] = None):
        self.previous_hash = ""
        self.hash = ""
        if config_file:
            self.config_file = _cfg2path(config_file)
        self.compute()

    def compute(
        self,
        input_file: Optional[Union[str, Path]] = None,
        read_chunk_size: int = 65536,
        algorithm: str = "sha256",
    ):
        """Hash a file using hashlib.
        Default algorithm is SHA-256

        :param input_file: File to hash
        :param int read_chunk_size: Maximum number of bytes to be read from the file
         at once. Default is 65536 bytes or 64KB
        :param str algorithm: The hash algorithm name to use. For example, 'md5',
         'sha256', 'sha512' and so on. Default is 'sha256'. Refer to
         hashlib.algorithms_available for available algorithms
        """

        lp: str = f"config file::hash {algorithm}::"
        checksum = new(algorithm)  # Raises appropriate exceptions.
        self.previous_hash = str(self.hash)
        self.hash = ""
        if input_file:
            self.config_file = _cfg2path(input_file)

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
            self.hash = checksum.hexdigest()
            logger.debug(
                f"{lp} the hex-digest for file '{self.config_file.as_posix()}' -> {self.hash}"
            )
        return self.hash

    def compare(self, compare_hash: str) -> bool:
        if self.hash == compare_hash:
            return True
        return False

    def __repr__(self):
        return f"{self.hash}"

    def __str__(self):
        return f"{self.hash}"


class ZMClient:
    config_file: Union[str, Path]
    config_hash: CFGHash
    raw_config: str
    parsed_cfg: Dict
    config: ConfigFileModel
    api: ZMApi
    db: ZMDB
    routes: List[ServerRoute]
    mid: int
    eid: int
    image_pipeline: Union[APIImagePipeLine, SHMImagePipeLine, ZMUImagePipeLine]
    _comb: Dict

    def check_permissions(self):
        lp: str = "check_permissions::"
        if not os.access(g.config.system.variable_data_path, os.W_OK):
            logger.error(
                f"{lp} system:variable_data_path [{g.config.system.variable_data_path}] is not writable by user {self.sys_user}"
            )
            raise PermissionError(
                f"system:variable_data_path [{g.config.system.variable_data_path}] is not writable by user {self.sys_user}"
            )
        if not os.access(g.config.system.variable_data_path, os.R_OK):
            logger.error(
                f"{lp} system:variable_data_path [{g.config.system.variable_data_path}] is not readable by user {self.sys_user}"
            )
            raise PermissionError(
                f"system:variable_data_path [{g.config.system.variable_data_path}] is not readable by user {self.sys_user}"
            )

    def __init__(self, cfg_file: Optional[Union[str, Path]] = None):
        """
        Initialize the ZoneMinder Client
        :param cfg_file: Path to the config file
        """
        logger.debug("Initializing ZMClient")
        self.pushed_labels = []
        from src.zm_ml.Client.Notifications.Pushover import Pushover
        self.pushover: Pushover = Pushover()
        self._comb_filters: Dict = {}
        self.zones: Dict = {}
        self.zone_polygons: List[Polygon] = []
        self.zone_filters: Dict = {}
        self.filtered_labels: Dict = {}
        import getpass
        import grp

        self.sys_user: str = getpass.getuser()
        self.sys_gid: int = os.getgid()
        self.sys_group: str = grp.getgrgid(self.sys_gid).gr_name
        self.sys_uid: int = os.getuid()
        from collections import namedtuple

        self.static_objects = namedtuple(
            "PreviousResults", ["label", "confidence", "bbox"]
        )

        global g, ENV_VARS
        ENV_VARS = ClientEnvVars()
        g = GlobalConfig()
        g.Environment = ENV_VARS
        if not cfg_file:
            logger.warning(
                f"No config file specified, using defaults from ENV -> {g.Environment.conf_file}"
            )
            cfg_file = ENV_VARS.conf_file
        if cfg_file:
            if isinstance(cfg_file, (str, Path)):

                g.config_file = self.config_file = (
                    Path(cfg_file) if isinstance(cfg_file, str) else cfg_file
                )
            else:
                raise TypeError("cfg_file must be a str or Path object")
        assert cfg_file, "No config file specified"
        g.config = self.config = self.load_config()
        check_imports()

        futures: List[concurrent.futures.Future] = []
        _hash: concurrent.futures.Future
        with concurrent.futures.ThreadPoolExecutor(
            thread_name_prefix="init", max_workers=g.config.system.thread_workers
        ) as executor:
            _hash = executor.submit(CFGHash, self.config_file)
            executor.submit(self.check_permissions)
            executor.submit(self._sort_routes)
            futures.append(executor.submit(self._init_logs))
            futures.append(executor.submit(self._init_db))
            futures.append(executor.submit(self._init_api))
            for future in concurrent.futures.as_completed(futures):
                future.result()
        self.config_hash = _hash.result()

    def _sort_routes(self):
        self.routes = self.config.mlapi.routes
        if len(self.routes) > 1:
            logger.debug(f"Routes: BEFORE sorting >> {self.routes}")
            self.routes.sort(key=lambda x: x.weight)
            logger.debug(f"Routes: AFTER sorting >> {self.routes}")

    def _init_logs(self):
        """Initialize the logging system."""
        level = self.config.logging.level
        level = level.casefold().strip()
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

        if self.config.logging.console is False:
            logger.info(f"Removing console log output!")
            logger.removeHandler(stream_handler)
        file_from_config = self.config.logging.dir / self.config.logging.file_name
        if not file_from_config.exists():
            file_from_config.touch(exist_ok=True)

        if os.access(file_from_config, os.W_OK):
            file_from_config.touch(exist_ok=True)
            # ZM /var/log/zm is handled by logrotate
            file_handler = logging.FileHandler(file_from_config.as_posix(), mode="a")
            # file_handler = logging.handlers.TimedRotatingFileHandler(
            #     file_from_config, when="midnight", interval=1, backupCount=7
            # )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            logger.debug(
                f"Logging to file '{file_from_config}' with user: "
                f"{self.sys_uid} [{self.sys_user}] group: {self.sys_gid} [{self.sys_group}]"
            )
        else:
            logger.warning(
                f"Logging to file {file_from_config} disabled due to permissions"
                f" - No write access to {file_from_config.as_posix()} for user: "
                f"{self.sys_uid} [{self.sys_user}] group: {self.sys_gid} [{self.sys_group}]"
            )
        logger.info(f"Logging initialized...")

    def _init_db(self):
        logger.debug("Initializing DB")
        self.db = ZMDB()
        logger.debug("DB initialized...")

    def _get_db_data(self, eid: int):
        """Get data from the database"""
        global g
        g.eid = eid
        (
            g.mid,
            g.mon_name,
            g.mon_post,
            g.mon_pre,
            g.mon_fps,
            g.event_cause,
            g.event_path,
        ) = self.db.grab_all(g.eid)
        logger.debug(
            f"\n\n\n Checking if configured to import zones for monitor {g.mid}\n\n\n\n"
        )
        with concurrent.futures.ThreadPoolExecutor(
            thread_name_prefix="init-2", max_workers=g.config.system.thread_workers
        ) as executor:
            executor.submit(self.api.import_zones)
            executor.submit(lambda: static_pickle)

    def _init_api(self):
        g.api = self.api = ZMApi(self.config.zoneminder)

    def combine_filters(
        self,
        filters_1: Union[Dict, MatchFilters, OverRideMatchFilters],
        filters_2: Union[Dict, OverRideMatchFilters],
    ):
        lp: str = "combine filters::"
        # logger.debug(f"{lp} BASE filters [type: {type(filters_1)}]: {filters_1}")
        # logger.debug(f"{lp} OVERRIDE filters [type: {type(filters_2)}]: {filters_2}")
        if isinstance(filters_1, (MatchFilters, OverRideMatchFilters)):
            # logger.debug(f"{lp} filters_1 is a MatchFilters object, converting to dict")
            output_filters: Dict = filters_1.dict()
        elif isinstance(filters_1, dict):
            output_filters = filters_1
        else:
            raise TypeError("filters_1 must be a dict or (OverRide)MatchFilters object")
        _base_obj_label: Dict = output_filters["object"]["labels"]
        # logger.debug(f"{lp} BASE object.labels: {_base_obj_label}")

        if isinstance(filters_2, (MatchFilters, OverRideMatchFilters)):
            # logger.debug(
            #     f"{lp} filters_2 is a {type(filters_2)} object, converting to dict"
            # )
            override_filters: Dict = filters_2.dict()
        elif isinstance(filters_2, dict):
            override_filters = filters_2
        else:
            raise TypeError("filters_2 must be a dict or OverRideMatchFilters object")
        _override_obj_label: Dict = override_filters["object"]["labels"]
        # logger.debug(f"{lp} OVERRIDE object.labels: {_override_obj_label}")
        # if _base_obj_label is None:
        #     _base_obj_label = {}
        if output_filters:
            if _override_obj_label:
                for label, filter_data in _override_obj_label.items():
                    if output_filters["object"]["labels"] is None:
                        output_filters["object"]["labels"] = {}

                    if _base_obj_label and label in _base_obj_label:
                        for k, v in filter_data.items():
                            if (
                                v is not None
                                and v != output_filters["object"]["labels"][label][k]
                            ):
                                # logger.debug(
                                #     f"{lp} Overriding BASE filter 'object':'labels':'{label}':'{k}' with Monitor {g.mid} "
                                #     f"OVERRIDE filter VALUE '{v}'"
                                # )
                                output_filters["object"]["labels"][label][k] = v
                    else:
                        # logger.debug(
                        #     f"{lp} Adding Monitor {g.mid} OVERRIDE filter 'object':'labels'"
                        #     f":'{label}' with VALUE '{filter_data}'"
                        # )
                        output_filters["object"]["labels"][label] = filter_data

            for filter_type, filter_data in override_filters.items():
                if filter_data is not None:
                    for k, v in filter_data.items():
                        if k == "labels":
                            # Handled in the first loop
                            continue
                        if v is not None and v != output_filters[filter_type][k]:
                            output_filters[filter_type][k] = v
                            # logger.debug(
                            #     f"{lp} Overriding BASE filter '{filter_type}':'{k}' with Monitor {g.mid} "
                            #     f"OVERRIDE filter VALUE '{v}'"
                            # )
            # logger.debug(f"{lp} Final combined output => {output_filters}")
        if not output_filters["object"]["labels"]:
            output_filters["object"]["labels"] = None
        self._comb_filters = output_filters
        return self._comb_filters

    def detect(self, eid: int, mid: Optional[int] = None):
        lp = _lp = "detect::"
        image_name: Optional[Union[int, str]] = None
        _start = perf_counter()
        global g
        g.mid = mid
        logger.info(
            f"{lp} Running detection for event {eid}, obtaining monitor info using DB and API..."
        )
        self._get_db_data(eid)
        if not mid and g.mid:
            logger.debug(
                f"{lp} No monitor ID provided, using monitor ID from DB: {g.mid}"
            )
        elif not mid and not g.mid:
            raise ValueError(
                f"{lp} No monitor ID provided, and no monitor ID from DB: Exiting..."
            )
        # get monitor and event info
        g.Monitor = self.api.get_monitor_data(g.mid)
        g.Event, event_monitor_data, g.Frame, _ = self.api.get_all_event_data(eid)
        # init Image Pipeline
        logger.debug(f"{lp} Initializing Image Pipeline...")
        how = self.config.detection_settings.images.pull_method
        if how.shm is True:
            logger.debug(f"{lp} Using SHM for image source")
            # self.image_pipeline = SHMImagePipeLine()
        elif how.api.enabled is True:
            logger.debug(f"{lp} Using ZM API for image source")
            self.image_pipeline = APIImagePipeLine(how.api)
        elif how.zmu is True:
            logger.debug(f"{lp} Using CLI 'zmu' for image source")
            pass
            # self.image_pipeline = ZMUImagePipeLine()

        models: Optional[Dict] = None
        if g.mid in self.config.monitors and self.config.monitors[g.mid].models:
            logger.debug(f"{lp} Monitor {g.mid} has a config entry for MODELS")
            models = self.config.monitors.get(g.mid).models
        if not models:
            if self.config.detection_settings.models:
                logger.debug(
                    f"{lp} Monitor {g.mid} has NO config entry for MODELS, using global "
                    f"models from detection_settings"
                )
                models = self.config.detection_settings.models
            else:
                logger.debug(
                    f"{lp} Monitor {g.mid} has NO config entry for MODELS, and global "
                    f"models from detection_settings is empty using 'yolov4'"
                )
                models = {"yolov4": {}}
        else:
            # dont duplicate Models
            models = {**self.config.detection_settings.models, **models}
        model_names = list(models.keys())
        models_str = ",".join(model_names)
        logger.debug(
            f"'DBG'>>> {models = } --- {model_names = } --- {models_str = } <<<DBG"
        )

        final_detections: dict = {}
        detections: dict = {}
        _start_detections = perf_counter()
        futures_data = []
        futures = []
        base_filters = g.config.matching.filters
        monitor_filters = g.config.monitors.get(g.mid).filters
        # logger.debug(f"{lp} Combining GLOBAL filters with Monitor {g.mid} filters")
        combined_filters = self.combine_filters(base_filters, monitor_filters)
        if g.mid in self.config.monitors:
            if self.config.monitors.get(g.mid).zones:
                self.zones = self.config.monitors.get(g.mid).zones
        if not self.zones:
            logger.debug(f"{lp} No zones found, adding full image with base filters")
            self.zones["!ZM-ML!_full_image"] = MonitorZones.construct(
                points=[
                    (0, 0),
                    (g.mon_width, 0),
                    (g.mon_width, g.mon_height),
                    (0, g.mon_height),
                ],
                resolution=f"{g.mon_width}x{g.mon_height}",
                object_confirm=False,
                static_objects=OverRideStaticObjects(),
                filters=OverRideMatchFilters(),
            )
        # build each zones filters as they won't change in the loops
        _i = 0
        for zone in self.zones:
            _i += 1
            # Break referencing, we want a copy
            cp_fltrs = copy.deepcopy(combined_filters)
            # logger.debug(
            #     f"'DBG'>>> \nBuilding filters using COMBINED global+monitor and overriding "
            #     f"with zone filters for: {'zone'} BEFORE COMBINED_FILTERS = {cp_fltrs} <<<DBG"
            # )
            self.zone_filters[zone] = self.combine_filters(
                cp_fltrs, self.zones[zone].filters
            )
            # logger.debug(f"'DBG'>>> AFTER COMBINED_FILTERS = {cp_fltrs} <<<DBG")
            cp_fltrs = None

        # logger.debug(f"'DBG'>>> Zone filters: \n\n{self.zone_filters} <<<DBG\n")
        if g.config.matching.static_objects.enabled:
            static_labels, static_conf, static_bbox = static_pickle()
            self.static_objects.labels = static_labels
            self.static_objects.confidence = static_conf
            self.static_objects.bbox = static_bbox
        with concurrent.futures.ThreadPoolExecutor(
            thread_name_prefix="detection-request",
            max_workers=g.config.system.thread_workers,
        ) as executor:
            while self.image_pipeline.is_image_stream_active():
                image, image_name = self.image_pipeline.get_image()

                if image is None:
                    logger.warning(f"{lp} No image returned! trying again...")
                    continue

                if how.api.enabled is True:
                    image_name = str(image_name).split("fid_")[1].split(".")[0]
                cv2_image = np.asarray(bytearray(image), dtype="uint8")
                cv2_image = cv2.imdecode(cv2_image, cv2.IMREAD_COLOR)
                cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
                if any(
                    [g.config.animation.gif.enabled, g.config.animation.mp4.enabled]
                ):

                    if g.config.animation.low_memory:
                        # save to file
                        _tmp = g.config.system.tmp_path / "animations"
                        _tmp.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(_tmp / f"{image_name}.jpg"), cv2_image)
                        # Add Path pbject pointing to the image on disk
                        g.frame_buffer[image_name] = _tmp / f"{image_name}.jpg"
                    else:
                        # Keep images in RAM
                        g.frame_buffer[image_name] = cv2_image

                for route in self.routes:
                    if route.enabled:
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
                        logger.debug(f"Sending image to '{route.name}' @ {url}")
                        _perf = perf_counter()
                        futures.append(
                            executor.submit(
                                requests.post,
                                url,
                                data=multipart_data,
                                headers=headers,
                                timeout=route.timeout,
                            )
                        )
                        futures_data.append(
                            {
                                "image": cv2_image,
                                "route": route,
                                "started": _perf,
                                "image_name": image_name,
                            }
                        )

                    else:
                        logger.warning(f"Neo-MLAPI route '{route.name}' is disabled!")
            for future, f_data in zip(
                concurrent.futures.as_completed(futures), futures_data
            ):
                future: Future
                route = f_data["route"]
                _perf = f_data["started"]
                image = f_data["image"]
                image_name = f_data["image_name"]
                if how.api.enabled is True:
                    image_name = int(image_name)
                else:
                    image_name = image_name
                try:
                    exception_ = future.exception(timeout=60)
                    if exception_:
                        raise exception_
                    r: requests.Response = future.result()
                except requests.exceptions.ConnectionError as e:
                    logger.debug(f"{e.args=}")
                    logger.warning(f"{lp} Route: {route.name} ConnectionError: {e}")
                    continue

                except Exception as e:
                    logger.warning(f"DBG>>>DBG??? {e.args=} {type(e)=}")
                    continue

                logger.debug(
                    f"perf:: HTTP Detection request to '{route.name}' completed in {perf_counter() - _perf:.5f} seconds"
                )

                results: List[Dict[str, Any]] = r.json()
                logger.debug(
                    f"Results: {len(results)} Type: {type(results)} => {results}"
                )
                filter_start = perf_counter()
                filtered_results = self.filter_detections(results, image, image_name)
                final_detections[image_name] = filtered_results
                logger.debug(
                    f"perf:: TOTAL Filtering took {perf_counter() - filter_start:.5f} seconds"
                )

        logger.debug(
            f"perf:: Total detections time {perf_counter() - _start:.5f} seconds"
        )
        # logger.debug(f"\n\n\nFINAL RESULTS: {final_detections}\n\n\n")

        if g.config.matching.static_objects.enabled:
            static_labels, static_conf, static_bbox = [], [], []
            for img_name, _results in final_detections.items():
                for obj in _results:
                    if obj["success"]:
                        static_labels.append(obj["label"])
                        static_conf.append(obj["confidence"])
                        static_bbox.append(obj["bbox"])
            static_pickle(static_labels, static_conf, static_bbox, write=True)

        return final_detections

    def filter_detections(
        self,
        results: List[Dict[str, Any]],
        image: np.ndarray,
        image_name: str,
    ) -> List[Dict[str, Any]]:
        """Filter detections"""
        lp: str = "filter detections::"
        filtered_results: List[Dict[str, Any]] = []
        r_idx = 0
        for result in results:
            r_idx += 1

            if result["success"] is True:
                labels, confidences, bboxes = [], [], []
                final_label, final_confidence, final_bbox = [], [], []
                labels, confidences, bboxs = self._filter(
                    result, image=image, image_name=image_name
                )

                for lbl, cnf, boxes in zip(labels, confidences, bboxs):
                    if cnf not in final_confidence and boxes not in final_bbox:
                        logger.debug(
                            f"DBG>>> \n\n {lbl} {cnf} {boxes} IS NOT IN FINAL LIST... ADDING! \n\n <<<DBG"
                        )
                        final_label.append(lbl)
                        final_confidence.append(cnf)
                        final_bbox.append(boxes)

                    else:
                        logger.debug(
                            f"DBG>>> \n\n {lbl} {cnf} {boxes} IS IN FINAL LIST... "
                            f"SKIPPING! [DONT WANT DUPLICATE] \n\n <<<DBG"
                        )

                filtered_result = {
                    "success": False if not final_label else True,
                    "type": result["type"],
                    "processor": result["processor"],
                    "model_name": result["model_name"],
                    "label": final_label,
                    "confidence": final_confidence,
                    "bounding_box": final_bbox,
                }
                logger.debug(f"DBG>>> FILTERED RESULT: {filtered_result} <<<DBG")
                filtered_results.append(filtered_result)
                # send notifications
                self.send_notifications(
                    (final_label, final_confidence, final_bbox), image, image_name
                )

            else:
                logger.warning(f"{lp} Result was not successful...")

        return filtered_results

    @staticmethod
    def _bbox2points(bbox: List) -> list[tuple[tuple[Any, Any], tuple[Any, Any]]]:
        """Convert bounding box coords to a Polygon acceptable input for shapely."""
        orig_ = list(bbox)
        if isinstance(bbox[0], int):
            it = iter(bbox)
            bbox = list(zip(it, it))
        bbox.insert(1, (bbox[1][0], bbox[0][1]))
        bbox.insert(3, (bbox[0][0], bbox[2][1]))
        # logger.debug(f"convert bbox:: {orig_} to Polygon points: {bbox}")
        return bbox

    def _filter(
        self,
        result: Dict[str, Any],
        image: np.ndarray = None,
        image_name: str = None,
        *args,
        **kwargs,
    ):
        """Filter detections using 2 loops, first loop is filter by object label, second loop is to filter by zone."""
        r_label, r_conf, r_bbox = [], [], []
        zones = self.zones
        zone_filters = self.zone_filters
        object_label_filters = {}
        final_filters = None
        base_filters = None
        strategy: MatchStrategy = g.config.matching.strategy
        type_ = result["type"]
        model_name = result["model_name"]
        processor = result["processor"]
        lp = f"_filter:{image_name}:'{model_name}'::{type_}::"
        passed_zones: bool = False
        # image_polygon = Polygon(
        #     [
        #         (0, 0),
        #         (g.mon_width, 0),
        #         (g.mon_width, g.mon_height),
        #         (0, g.mon_height),
        #     ]
        # )
        label, confidence, bbox = None, None, None
        _zn_tot = len(zones)
        _lbl_tot = len(result["label"])
        idx = 0
        i = 0
        zone_name: str
        zone_data: MonitorZones
        #
        # Outer Loop
        #
        for (label, confidence, bbox) in zip(
            result["label"],
            result["confidence"],
            result["bounding_box"],
        ):
            i += 1

            _lp = f"{lp}'{label}' {i}/{_lbl_tot}::"

            #
            # Inner Loop
            #
            idx = 0
            for zone_name, zone_data in zones.items():
                passed_zones = False
                idx += 1
                __lp = f"{_lp}zone {idx}/{_zn_tot}::"
                if zone_data.enabled is False:
                    logger.debug(f"{__lp} Zone '{zone_name}' is disabled...")
                    continue
                if not zone_data.points:
                    logger.warning(
                        f"{__lp} Zone '{zone_name}' has no points! Did you rename a Zone in ZM"
                        f" or forget to add points? SKIPPING..."
                    )
                    continue
                zone_polygon = Polygon(zone_data.points)
                if zone_polygon not in self.zone_polygons:
                    self.zone_polygons.append(zone_polygon)
                bbox_polygon = Polygon(self._bbox2points(bbox))

                if bbox_polygon.intersects(zone_polygon):
                    logger.debug(
                        f"{__lp} inside of Zone @ {list(zip(*zone_polygon.exterior.coords.xy))[:-1]}"
                    )
                    if zone_name in zone_filters and zone_filters[zone_name]:
                        # logger.debug(f"{lp} zone '{zone_name}' has filters")
                        final_filters = zone_filters[zone_name]
                    else:
                        # logger.debug(
                        #     f"{lp} zone '{zone_name}' has NO filters, using COMBINED global+monitor filters"
                        # )
                        final_filters = self._comb_filters
                    if isinstance(final_filters, dict) and isinstance(
                        final_filters.get("object"), dict
                    ):

                        # logger.debug(
                        #     f"{type(final_filters)=} -- {type(object_label_filters)=}\n\n{final_filters=}\n\n"
                        # )
                        object_label_filters = final_filters["object"]["labels"]
                        # logger.debug(f"\nFINAL FILTERS as DICT {final_filters=}\n\n")
                        if object_label_filters and label in object_label_filters:
                            if object_label_filters[label]:
                                logger.debug(
                                    f"{lp} '{label}' IS IN per label filters for zone '{zone_name}'"
                                )

                                for k, v in object_label_filters[label].items():
                                    if (
                                        v is not None
                                        and final_filters["object"][k] != v
                                    ):
                                        # logger.debug(
                                        #     f"{lp} Overriding object:'{k}' [{final_filters['object'][k]}] "
                                        #     f"with ZONE object:labels:{k} filter VALUE={v}"
                                        # )
                                        final_filters["object"][k] = v
                        else:
                            logger.debug(f"{lp} NOT IN per label filters")

                        final_filters = self.construct_filter(final_filters)
                        # logger.debug(
                        #     f"\n\nFINAL FILTERS 'AFTER' CONSTRUCTING [{type(final_filters)}]\n\n"
                        # )
                        self.zone_filters[zone_name] = final_filters
                        # logger.debug(
                        #     f"\n\n'AFTER' SAVING TO ZONE FILTERS: {self.zone_filters=} \n\n"
                        # )

                    type_filter: Union[
                        OverRideObjectFilters,
                        OverRideFaceFilters,
                        OverRideAlprFilters,
                        None,
                    ] = None
                    if type_ == "object":
                        type_filter = final_filters.object
                    elif type_ == "face":
                        type_filter = final_filters.face
                    elif type_ == "alpr":
                        type_filter = final_filters.alpr

                    # logger.debug(
                    #     f"{_lp} SORTED by {type_} -- FINAL filters => \n{type_filter}\n"
                    # )
                    pattern = type_filter.pattern
                    #
                    # Start filtering
                    #
                    lp = f"{__lp}pattern match::"
                    if match := pattern.match(label):
                        if label in match.groups():
                            logger.debug(
                                f"{lp} matched ReGex pattern [{pattern.pattern}] ALLOWING..."
                            )

                            lp = f"{__lp}min conf::"
                            if confidence >= type_filter.min_conf:
                                logger.debug(
                                    f"{lp} {confidence} IS GREATER THAN OR EQUAL TO "
                                    f"min_conf={type_filter.min_conf}, ALLOWING..."
                                )
                                w, h = g.mon_width, g.mon_height
                                max_object_area_of_image: Optional[
                                    Union[float, int]
                                ] = None
                                min_object_area_of_image: Optional[
                                    Union[float, int]
                                ] = None
                                max_object_area_of_zone: Optional[
                                    Union[float, int]
                                ] = None
                                min_object_area_of_zone: Optional[
                                    Union[float, int]
                                ] = None

                                # check total max area
                                if tma := type_filter.total_max_area:
                                    lp = f"{__lp}total max area::"
                                    if isinstance(tma, float):
                                        if tma >= 1.0:
                                            tma = 1.0
                                            max_object_area_of_image = h * w
                                        else:
                                            max_object_area_of_image = tma * (h * w)
                                            logger.debug(
                                                f"{lp} converted {tma * 100.00}% of {w}*{h}->{w * h:.2f} to "
                                                f"{max_object_area_of_image:.2f} pixels",
                                            )

                                        if max_object_area_of_image > (h * w):
                                            max_object_area_of_image = h * w
                                    elif isinstance(tma, int):
                                        max_object_area_of_image = tma
                                    else:
                                        logger.warning(
                                            f"{lp} Unknown type for total_max_area, defaulting to PIXELS "
                                            f"h*w of image ({h * w})"
                                        )
                                        max_object_area_of_image = h * w
                                    if max_object_area_of_image:
                                        if bbox_polygon.area > max_object_area_of_image:
                                            logger.debug(
                                                f"{lp} {bbox_polygon.area:.2f} is larger then the max allowed: "
                                                f"{max_object_area_of_image:.2f},"
                                                f"\n\nREMOVING REMOVING REMOVING REMOVING REMOVING REMOVING...\n\n"
                                            )
                                            passed_zones = False
                                            continue
                                        else:
                                            logger.debug(
                                                f"{lp} {bbox_polygon.area:.2f} is smaller then the TOTAL (image w*h) "
                                                f"max allowed: {max_object_area_of_image:.2f}, ALLOWING..."
                                            )
                                else:
                                    logger.debug(f"{lp} no total_max_area set")

                                # check total min area
                                if tmia := type_filter.total_min_area:
                                    lp = f"{__lp}total min area=>"
                                    if isinstance(tmia, float):
                                        if tmia >= 1.0:
                                            tmia = 1.0
                                            min_object_area_of_image = h * w
                                        else:
                                            min_object_area_of_image = (
                                                tmia * zone_polygon.area
                                            )
                                            logger.debug(
                                                f"{lp} converted {tmia * 100.00}% of {w}*{h}->{w * h:.2f} to "
                                                f"{min_object_area_of_image:.2f} pixels",
                                            )

                                    elif isinstance(tmia, int):
                                        min_object_area_of_image = tmia
                                    else:
                                        logger.warning(
                                            f"{lp} Unknown type for total_min_area, defaulting to 1 PIXEL"
                                        )
                                        min_object_area_of_image = 1
                                    if min_object_area_of_image:
                                        if (
                                            bbox_polygon.area
                                            >= min_object_area_of_image
                                        ):
                                            logger.debug(
                                                f"{lp} {bbox_polygon.area:.2f} is LARGER THEN OR EQUAL TO the "
                                                f"TOTAL min allowed: {min_object_area_of_image:.2f}, ALLOWING..."
                                            )
                                        else:
                                            logger.debug(
                                                f"{lp} {bbox_polygon.area:.2f} is smaller then the TOTAL min allowed"
                                                f": {min_object_area_of_image:.2f}, "
                                                f"\n\nREMOVING REMOVING REMOVING REMOVING REMOVING REMOVING...\n\n"
                                            )
                                            passed_zones = False
                                            continue
                                else:
                                    logger.debug(f"{lp} no total_min_area set")

                                # check max area
                                if max_area := type_filter.max_area:
                                    lp = f"{__lp}zone max area=>"

                                    if isinstance(max_area, float):
                                        if max_area >= 1.0:
                                            max_area = 1.0
                                            max_object_area_of_zone = zone_polygon.area
                                        else:
                                            max_object_area_of_zone = (
                                                max_area * zone_polygon.area
                                            )
                                            logger.debug(
                                                f"{lp} converted {max_area * 100.00}% of '{zone_name}'->"
                                                f"{zone_polygon.area:.2f} to {max_object_area_of_zone} pixels",
                                            )
                                        if max_object_area_of_zone > zone_polygon.area:
                                            max_object_area_of_zone = zone_polygon.area
                                    elif isinstance(max_area, int):
                                        max_object_area_of_zone = max_area
                                    else:
                                        logger.warning(
                                            f"{lp} Unknown type for max_area, defaulting to PIXELS "
                                            f"of zone ({zone_polygon.area})"
                                        )
                                        max_object_area_of_zone = zone_polygon.area
                                    if max_object_area_of_zone:
                                        if (
                                            bbox_polygon.intersection(zone_polygon).area
                                            > max_object_area_of_zone
                                        ):
                                            logger.debug(
                                                f"{lp} BBOX AREA [{bbox_polygon.area:.2f}] is larger than the "
                                                f"max allowed: {max_object_area_of_zone:.2f},"
                                                f"\n\nREMOVING REMOVING REMOVING REMOVING REMOVING REMOVING...\n\n"
                                            )
                                            passed_zones = False
                                            continue
                                        else:
                                            logger.debug(
                                                f"{lp} '{label}' BBOX AREA [{bbox_polygon.area:.2f}] is smaller than the "
                                                f"max allowed: {max_object_area_of_zone:.2f}, ALLOWING..."
                                            )
                                else:
                                    logger.debug(f"{lp} no max_area set")

                                # check min area
                                if min_area := type_filter.min_area:
                                    lp = f"{__lp}zone min area=>"

                                    if isinstance(min_area, float):
                                        if min_area >= 1.0:
                                            min_area = 1.0
                                            min_object_area_of_zone = zone_polygon.area
                                        else:
                                            min_object_area_of_zone = (
                                                min_area * zone_polygon.area
                                            )
                                            logger.debug(
                                                f"{lp} converted {min_area * 100.00}% of '{zone_name}'->{zone_polygon.area:.5f}"
                                                f" to {min_object_area_of_zone} pixels",
                                            )
                                        if (
                                            min_object_area_of_zone
                                            and min_object_area_of_zone
                                            > zone_polygon.area
                                        ):
                                            min_object_area_of_zone = zone_polygon.area
                                    elif isinstance(min_area, int):
                                        min_object_area_of_zone = min_area
                                    else:
                                        min_object_area_of_zone = 1
                                    if (
                                        min_object_area_of_zone
                                        and bbox_polygon.intersection(zone_polygon).area
                                        > min_object_area_of_zone
                                    ):
                                        logger.debug(
                                            f"{lp} '{label}' BBOX AREA [{bbox_polygon.area:.5f}] is larger then the "
                                            f"min allowed: {min_object_area_of_zone:.5f}, ALLOWING..."
                                        )

                                    else:
                                        logger.debug(
                                            f"{lp} '{label}' BBOX AREA [{bbox_polygon.area:.5f}] is smaller then the "
                                            f"min allowed: {min_object_area_of_zone:.5f},"
                                            f"\n\nNO MATCH, SKIPPING...\n\n"
                                        )
                                        self.filtered_labels[image_name] = (
                                            label,
                                            confidence,
                                            bbox,
                                        )
                                        continue
                                else:
                                    logger.debug(f"{lp} no min_area set")
                                s_o = g.config.matching.static_objects.enabled
                                mon_filt = g.config.monitors.get(g.mid)
                                zone_filt: Optional[MonitorZones] = None
                                if mon_filt and zone_name in mon_filt.zones:
                                    zone_filt = mon_filt.zones[zone_name]

                                # Override with monitor filters than zone filters
                                if not s_o:
                                    if mon_filt and mon_filt.static_objects.enabled:
                                        s_o = True
                                elif s_o:
                                    if mon_filt and not mon_filt.static_objects.enabled:
                                        s_o = False
                                # zone filters override monitor filters
                                if not s_o:
                                    if (
                                        zone_filt
                                        and zone_filt.static_objects.enabled is True
                                    ):
                                        s_o = True
                                elif s_o:
                                    if (
                                        zone_filt
                                        and zone_filt.static_objects.enabled is False
                                    ):
                                        s_o = False
                                if s_o:
                                    logger.debug(
                                        f"{__lp} 'static_objects' enabled, checking for matches"
                                    )
                                    if self.check_for_static_objects(
                                        label, confidence, bbox_polygon, zone_name
                                    ):
                                        # success
                                        pass
                                    else:
                                        # failed
                                        continue
                                else:
                                    logger.debug(
                                        f"{__lp} 'static_objects' disabled, skipping match check"
                                    )
                                # !!!!!!!!!!!!!!!!!!!!
                                # End of all filters
                                # !!!!!!!!!!!!!!!!!!!!

                                passed_zones = True
                                break

                            else:
                                logger.debug(
                                    f"{lp} confidence={confidence} IS LESS THAN "
                                    f"min_confidence={type_filter.min_conf}, "
                                    f"\n\nNO MATCH, SKIPPING...\n\n"
                                )
                                self.filtered_labels[image_name] = (
                                    label,
                                    confidence,
                                    bbox,
                                )
                                continue

                        else:
                            logger.debug(
                                f"{lp} NOT matched in RegEx pattern [{pattern.pattern}], "
                                f"\n\nNO MATCH, SKIPPING...\n\n"
                            )
                            self.filtered_labels[image_name] = (
                                label,
                                confidence,
                                bbox,
                            )
                            continue

                    else:
                        logger.debug(
                            f"{lp} MATCH FAILED [{match = }], "
                            f"\n\nNO MATCH, SKIPPING...\n\n"
                        )
                        self.filtered_labels[image_name] = (
                            label,
                            confidence,
                            bbox,
                        )
                        continue
                else:
                    logger.debug(
                        f"{__lp} NOT in zone [{zone_name}], continuing to next zone..."
                    )

            if passed_zones:
                logger.debug(f"PASSED FILTERING, ADDING TO FINAL LIST")
                passed_zones = False
                r_label.append(label)
                r_conf.append(confidence)
                r_bbox.append(bbox)
                continue

        logger.warning(
            f"{_lp} OUT OF BOTH LABEL AND ZONE LOOP {strategy=} "
            f"FINAL LABELS={r_label}, CONFIDENCE={r_conf}, BBOX={r_bbox}"
        )

        return r_label, r_conf, r_bbox

    def annotate_image(
        self,
        image: np.ndarray,
        labels: List[str],
        confidences: List[float],
        bboxes: List[List[int]],
        filtered_bboxes: List[List[int]],
        zones: List[Polygon],
        model_name: str,
        processor: str,
    ):
        """
        Annotate the image with the labels, confidences and bboxes
        :param image: the image to draw
        :param labels: the labels to draw
        :param confidences: the confidences to draw
        :param bboxes: the bounding boxes to draw
        :param zones: the zones to draw
        :return: the annotated image
        """
        from .Models.utils import draw_bounding_boxes

        lp = f"{self.__class__.__name__}.{sys._getframe().f_code.co_name}() "
        logger.debug(f"{lp} annotating image with labels={labels}")
        img = image.copy()
        img = draw_bounding_boxes(
            img,
            labels,
            confidences,
            bboxes,
            model_name,
            processor,
            g.config.detection_settings.images.annotation.confidence,
            g.config.detection_settings.images.annotation.models.enabled,
            g.config.detection_settings.images.annotation.models.processor,
        )
        if g.config.detection_settings.images.annotation.zones.enabled:
            from src.zm_ml.Client.Models.utils import draw_zones

            img = draw_zones(
                img,
                zones,
                g.config.detection_settings.images.annotation.zones.color,
                g.config.detection_settings.images.annotation.zones.thickness,
            )
        if g.config.detection_settings.images.debug.enabled:
            from src.zm_ml.Client.Models.utils import draw_filtered_bboxes

            debug_image = draw_filtered_bboxes(img, filtered_bboxes)
            from datetime import datetime

            cv2.imwrite(
                g.config.detection_settings.images.debug.path
                / f"debug-img_{datetime.now()}",
                debug_image,
            )

    def create_animations(self, label, confidence, bbox):
        """
        Create animations and save to disk
        :param label:
        :param confidence:
        :param bbox:
        :param zone_name:
        :return:
        """
        lp = f"{g.lp()}create_animations=>"
        logger.debug(f"{lp} STARTED")

    def check_for_static_objects(
        self, current_label, current_confidence, current_bbox_polygon, zone_name
    ) -> bool:
        """Check for static objects in the frame
        :param current_label:
        :param current_confidence:
        :param current_bbox_polygon:
        """

        lp = f"check_for_static_objects::"
        logger.debug(f"{lp} STARTING...")
        aliases: Dict = g.config.label_groups
        mda = g.config.matching.static_objects.difference
        if not mda:
            pass
        mon_filt = g.config.monitors.get(g.mid)
        zone_filt: Optional[MonitorZones] = None
        if mon_filt and zone_name in mon_filt.zones:
            zone_filt = mon_filt.zones[zone_name]
        # Override with monitor filters than zone filters
        if mon_filt and mon_filt.static_objects.difference:
            mda = mon_filt.static_objects.difference
        if zone_filt and zone_filt.static_objects.difference:
            mda = zone_filt.static_objects.difference

        # todo: inherit ignore_labels from monitor and zone
        ignore_labels: Optional[List[str]] = (
            g.config.matching.static_objects.ignore_labels or []
        )
        if mon_filt and mon_filt.static_objects.labels:
            for lbl in mon_filt.static_objects.labels:
                if lbl not in ignore_labels:
                    ignore_labels.append(lbl)
        if zone_filt and zone_filt.static_objects.difference:
            for lbl in zone_filt.static_objects.labels:
                if lbl not in ignore_labels:
                    ignore_labels.append(lbl)

        if ignore_labels and current_label in ignore_labels:
            logger.debug(
                f"{lp} {current_label} is in static_objects:ignore_labels: {ignore_labels}, skipping",
            )
        else:
            logger.debug(
                f"{lp} max difference between current and past object area found! -> {mda}"
            )
            mda = None
            if isinstance(mda, float):
                if mda >= 1.0:
                    mda = 1.0
            elif isinstance(mda, int):
                pass

            else:
                logger.warning(f"{lp} Unknown type for difference, defaulting to 5%")
                mda = 0.05

            for saved_label, saved_conf, saved_bbox in zip(
                self.static_objects.labels,
                self.static_objects.confidence,
                self.static_objects.bbox,
            ):

                # compare current detection element with saved list from file
                found_alias_grouping = False
                # check if it is in a label group
                if saved_label != current_label:
                    if aliases:
                        logger.debug(
                            f"{lp} currently detected object does not match saved object, "
                            f"checking label_groups for an aliased match"
                        )

                        for alias, alias_group in aliases.items():
                            if (
                                saved_label in alias_group
                                and current_label in alias_group
                            ):
                                logger.debug(
                                    f"{lp} saved and current object are in the same label group [{alias}]"
                                )
                                found_alias_grouping = True
                                break

                elif saved_label == current_label:
                    found_alias_grouping = True
                if not found_alias_grouping:
                    continue
                # Found a match by label/group, now compare the area using Polygon
                past_label_polygon = Polygon(self._bbox2points(saved_bbox))
                max_diff_pixels = None
                diff_area = None
                logger.debug(
                    f"{lp} comparing '{current_label}' PAST->{saved_bbox} to CURR->{list(zip(*current_bbox_polygon.exterior.coords.xy))[:-1]}",
                )
                if past_label_polygon.intersects(
                    current_bbox_polygon
                ) or current_bbox_polygon.intersects(past_label_polygon):
                    if past_label_polygon.intersects(current_bbox_polygon):
                        logger.debug(
                            f"{lp} the PAST object INTERSECTS the new object",
                        )
                    else:
                        logger.debug(
                            f"{lp} the current object INTERSECTS the PAST object",
                        )

                    if current_bbox_polygon.contains(past_label_polygon):
                        diff_area = current_bbox_polygon.difference(
                            past_label_polygon
                        ).area
                        if isinstance(mda, float):
                            max_diff_pixels = current_bbox_polygon.area * mda
                            logger.debug(
                                f"{lp} converted {mda*100:.2f}% difference from '{current_label}' "
                                f"is {max_diff_pixels} pixels"
                            )
                        elif isinstance(mda, int):
                            max_diff_pixels = mda
                    else:
                        diff_area = past_label_polygon.difference(
                            current_bbox_polygon
                        ).area
                        if isinstance(mda, float):
                            max_diff_pixels = past_label_polygon.area * mda
                            logger.debug(
                                f"{lp} converted {mda * 100:.2f}% difference from '{saved_label}' "
                                f"is {max_diff_pixels} pixels"
                            )
                        elif isinstance(mda, int):
                            max_diff_pixels = mda
                    if diff_area is not None and diff_area <= max_diff_pixels:
                        logger.debug(
                            f"{lp} removing '{current_label}' as it seems to be approximately in the same spot"
                            f" as it was detected last time based on '{mda}' -> Difference in pixels: {diff_area} "
                            f"- Configured maximum difference in pixels: {max_diff_pixels}"
                        )
                        return False
                        # if saved_bbox not in mpd_b:
                        #     logger.debug(
                        #         f"{lp} appending this saved object to the mpd "
                        #         f"buffer as it has removed a detection and should be propagated "
                        #         f"to the next event"
                        #     )
                        #     mpd_b.append(saved_bs[saved_idx])
                        #     mpd_l.append(saved_label)
                        #     mpd_c.append(saved_cs[saved_idx])
                        # new_err.append(b)
                    elif diff_area is not None and diff_area > max_diff_pixels:
                        logger.debug(
                            f"{lp} allowing '{current_label}' -> the difference in the area of last detection "
                            f"to this detection is '{diff_area:.2f}', a minimum of {max_diff_pixels:.2f} "
                            f"is needed to not be considered 'in the same spot'",
                        )
                        return True
                    elif diff_area is None:
                        logger.debug(f"DEBUG>>>'MPD' {diff_area = } - whats the issue?")
                    else:
                        logger.debug(
                            f"WHATS GOING ON? {diff_area = } -- {max_diff_pixels = }"
                        )
                # Saved does not intersect the current object/label
                else:
                    logger.debug(
                        f"{lp} current detection '{current_label}' is not near enough to '"
                        f"{saved_label}' to evaluate for match past detection filter"
                    )
                    return False
        return True

    @staticmethod
    def construct_filter(filters: Dict) -> OverRideMatchFilters:
        # construct each object label filter
        if filters["object"]["labels"]:
            for label_ in filters["object"]["labels"]:
                filters["object"]["labels"][label_] = OverRideObjectFilters.construct(
                    **filters["object"]["labels"][label_],
                )

        filters["object"] = OverRideObjectFilters.construct(
            **filters["object"],
        )
        filters["face"] = OverRideFaceFilters.construct(
            **filters["face"],
        )
        filters["alpr"] = OverRideAlprFilters.construct(
            **filters["alpr"],
        )
        return OverRideMatchFilters.construct(**filters)

    def load_config(self) -> Optional[ConfigFileModel]:
        """Parse the YAML configuration file. In the future this will read DB values"""
        cfg: Dict = {}
        _start = perf_counter()
        self.raw_config = self.config_file.read_text()

        try:
            cfg = yaml.safe_load(self.raw_config)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing the YAML configuration file!")
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
                if "client" in inc_vars:
                    inc_vars = inc_vars.get("client", {})
                    logger.debug(
                        f"Loaded {len(inc_vars)} substitution from IncludeFile {inc_file} => {inc_vars}"
                    )
                    # check for duplicates
                    for k in inc_vars:
                        if k in substitutions:
                            logger.warning(
                                f"Duplicate substitution variable '{k}' in IncludeFile {inc_file} - "
                                f"IncludeFile overrides config file"
                            )

                    substitutions.update(inc_vars)
                else:
                    logger.warning(
                        f"IncludeFile [{inc_file}] does not have a 'client' section - skipping"
                    )
            else:
                logger.warning(f"IncludeFile {inc_file} is not a file!")
        logger.debug(f"Replacing ${{VARS}} in config")
        cfg = self.replace_vars(self.raw_config, substitutions)
        self.parsed_cfg = dict(cfg)
        _x = ConfigFileModel(**cfg)
        logger.debug(
            f"perf:: Config file loaded and validated in {perf_counter() - _start:.5f} seconds"
        )
        return _x

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
            logger.debug(f"Found the following substitution variables: {var_list}")
            # substitute variables
            _known_vars = []
            _unknown_vars = []
            for var in var_list:
                if var in var_pool:
                    # logger.debug(
                    #     f"substitution variable '{var}' IS IN THE POOL! VALUE: "
                    #     f"{var_pool[var]} [{type(var_pool[var])}]"
                    # )
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
                    f"The following variables have no configured substitution value: {_unknown_vars}"
                )
            if _known_vars:
                logger.debug(
                    f"The following variables have been substituted: {_known_vars}"
                )
        else:
            logger.debug(f"No substitution variables found.")

        return yaml.safe_load(search_str)

    def compute_best_match(self, detections: Dict, type_: str = "zone") -> Dict:
        """Compute best match"""
        if type_ not in ("zone", "label"):
            raise ValueError(f"Invalid type_ '{type_}' [zone|label]")

        return {}

    def send_notifications(
        self,
        best_match: Tuple[List[str], List[float], List[List[int]]],
        image: np.ndarray,
        image_name: str,
    ):
        """Send notifications"""
        # TODO: mqtt, push, email, webhook
        for label, conf, bbox in best_match:
            if label in self.pushed_labels:
                continue
            noti_cfg = g.config.notifications
            if noti_cfg.pushover.enabled:
                self.pushover.send_notification(
                    label, conf, bbox, image, image_name, noti_cfg.pushover
                )


            logger.info(f"Sending notification for label '{label}'")


