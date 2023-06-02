#!/usr/bin/env python3
"""
A script to be used with ZoneMinder filter system. The filtering system can pass some CLI args
to the script. This is the beginning of the exploratory work on how to integrate this.

 This has 2 modes.

 - config mode: you can use ZM ML like monitor config to only annotate certain objects
     - This allows for filtering by label/group/zone
 - all: annotate anything it finds in the event
"""
import argparse
import logging
import sys
import time
import warnings
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import List, Optional, NamedTuple, Tuple, Dict, Union, Any

from zm_ml.Shared.Models.validators import (
    _validate_replace_localhost,
    str2path,
    str2bool, _validate_dir,
)

try:
    import pydantic.fields
    from pydantic import (
        BaseModel as PydanticBaseModel,
        Field,
        AnyUrl,
        IPvAnyAddress,
        validator,
    )
except ImportError:
    warnings.warn("pydantic not installed, please install it to use this script")
    raise
try:
    import ffmpegcv
except ImportError:
    warnings.warn("ffmpegcv not installed, please install it to use this script")
    raise
try:
    import cv2
except ImportError:
    warnings.warn(
        "OpenCV not installed, please install it to use this script (opencv-contrib-python or compile from source with CUDA)"
    )
    raise
try:
    import numpy as np
except ImportError:
    warnings.warn("numpy not installed, please install it to use this script")
    raise
try:
    import zm_ml
except ImportError:
    warnings.warn(
        "zm_ml not installed, please install it to use this script (see https://github.com/baudneo/ZM_ML/wiki/Manual-Installation#client)"
    )
    raise

from zm_ml.Client.Libs.DB import ZMDB
from zm_ml.Client.Models.config import (
    SystemSettings,
    MonitorsSettings,
    MatchFilters,
    ZoneMinderSettings,
    ClientEnvVars,
    ZMDBSettings,
)
from zm_ml.Client.main import parse_client_config_file as parse_cfg, set_logger
from zm_ml.Server.ML.coco17_cv2 import COCO17


LP: str = "filter:annotate:"
logger = logging.getLogger("AnnotateEvents")
logger.setLevel(logging.DEBUG)
formatter: logging.Formatter = logging.Formatter(
    "%(asctime)s.%(msecs)04d %(name)s[%(process)s] %(levelname)s %(module)s:%(lineno)d -> %(message)s",
    "%m/%d/%y %H:%M:%S",
)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)
# zm has logrotation enabled for all files ending in .log in its log folder

set_logger(logger)
SOURCE_DIR: Path
CLASS_NAMES: List[str] = []
# Resize the output video from whatever the source is
CONFIG_PATH: Optional[Path] = None
UPDATE_TIMER: float = 0.0
COLORS: List[Tuple[int, int, int]] = []
ZM_DB: Optional[ZMDB] = None


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True


class FilterConfigFileModel(BaseModel):
    class OutputSettings(BaseModel):
        class ResizeSettings(BaseModel):
            enabled: bool = False
            width: Optional[int] = 0
            height: Optional[int] = 0
            keep_aspect: bool = True

            _validate_1 = validator(
                "enabled", "keep_aspect", allow_reuse=True, pre=True
            )(str2bool)

        resize: ResizeSettings = Field(default_factory=ResizeSettings)

    class StatusUpdateSettings(BaseModel):
        enabled: bool = True
        every: Optional[int] = 10

        _validate_1 = validator("enabled", allow_reuse=True, pre=True)(str2bool)

    class LocalSettings(BaseModel):
        class LocalModelSettings(BaseModel):
            name: str = Field("n/a")
            confidence: float = 0.5
            nms: float = 0.3
            input: Path = Field(...)
            config: Optional[Path] = None
            labels: Optional[Path] = None
            width: int = 416
            height: int = 416


            @validator("input", "config", "labels", always=True)
            def check_path(cls, v, field: pydantic.fields.ModelField, **kwargs):
                if v:
                    v = str2path(v)
                    assert v.is_file(), f"{field.name} must be a valid file"
                return v

        enabled: bool = True
        model: Optional[LocalModelSettings] = None

    class RemoteSettings(BaseModel):
        enabled: bool = False
        host: Union[AnyUrl, IPvAnyAddress, None] = "localhost"
        port: int = 5000
        models: List[str] = Field(default_factory=list)

        @validator("host", allow_reuse=True, pre=True)
        def _validate_host(cls, v, field):
            logger.info(f"Validating KEY: {field.name} VALUE: {v} - {type(v)}")
            if v:
                v = _validate_replace_localhost(v)
                logger.debug(f"After validation: {v} - {type(v)}")
            return v

    class LoggingSettings(BaseModel):
        class LogFileSettings(BaseModel):
            enabled: bool = True
            path: Path = Field("/var/log/zm")
            filename: str = "annotated"

            _validate_1 = validator("enabled", allow_reuse=True, pre=True)(str2bool)
            _validate_2 = validator("path", allow_reuse=True)(str2path)

        debug: bool = False
        file: LogFileSettings = Field(default_factory=LogFileSettings)

    class FrameSkipSettings(BaseModel):
        enabled: bool = False
        every: int = 1

        _validate_1 = validator("enabled", allow_reuse=True, pre=True)(str2bool)

    substitutions: Dict[str, str] = Field(default_factory=dict)
    zm_conf: Path = Field("/etc/zm")
    status_updates: StatusUpdateSettings = Field(default_factory=StatusUpdateSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    annotate_labels: List[str] = Field(default_factory=list)
    db: ZMDBSettings = Field(default_factory=ZMDBSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    local: LocalSettings = Field(default_factory=LocalSettings)
    remote: RemoteSettings = Field(default_factory=RemoteSettings)
    import_zones: bool = False
    frame_skip: int = 1
    # filters: MatchFilters = Field(default_factory=MatchFilters)
    # monitors: Dict[int, MonitorsSettings] = Field(default_factory=dict)

    _validate_1 = validator("import_zones", pre=True, allow_reuse=True)(str2bool)

    @validator("zm_conf", pre=True, always=True)
    def _validate_2(cls, v, field, values, config):
        if v:
            v = str2path(v)
            v = _validate_dir(v)
        return v



class DetectedObject(NamedTuple):
    label: str
    confidence: float
    box: List[int]


class Detection(NamedTuple):
    frame: np.ndarray
    objects: List[DetectedObject]


class DetectMethod(str, Enum):
    LOCAL = "local"
    MLAPI = "mlapi"


class FileType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


def detect(
    image: np.ndarray,
    method: DetectMethod,
    conf: Optional[float] = None,
    nms: Optional[float] = None,
    allowed_labels: Optional[List[str]] = None,
    model: Optional[cv2.dnn.DetectionModel] = None,
):
    """Run detection on a frame"""
    height, width, _ = image.shape

    if not method:
        raise ValueError("No detection method specified")
    elif method == DetectMethod.LOCAL:
        if not conf:
            raise ValueError("No confidence threshold specified")
        if not nms:
            raise ValueError("No NMS threshold specified")

        # run inference locally
        _l, _c, _b = model.detect(image, conf, nms)
        for class_id, confidence, box in zip(_l, _c, _b):
            label = CLASS_NAMES[class_id]

            # TODO: filter if in config mode
            if allowed_labels and (label not in allowed_labels):
                continue
            x, y, _w, _h = (
                int(round(box[0])),
                int(round(box[1])),
                int(round(box[2])),
                int(round(box[3])),
            )
            confidence = float(confidence)
            bbox = [
                x,
                y,
                x + _w,
                y + _h,
            ]
            color = [int(c) for c in COLORS[class_id]]
            cv2.rectangle(image, (x, y), (x + _w, y + _h), color, 2)
            text = f"{label}: {confidence * 100:.2f}%"
            # scale text based on image resolution
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5 * min(width, height) / 1000
            thickness = max(int(font_scale * 2), 1)
            (text_width, text_height), _ = cv2.getTextSize(
                text, font, font_scale, thickness
            )
            cv2.rectangle(
                image,
                (x, y),
                (x + text_width, y - text_height),
                (
                    0,
                    0,
                    0,
                ),
                -1,
            )
            text_x = x
            text_y = y - 5
            cv2.putText(
                image,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )

            # cv2.putText(
            #     image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            # )
    elif method == DetectMethod.MLAPI:
        cfg = g.remote
        # comma separated string of models
        models = f",".join(cfg.models).rstrip(",")
        from requests_toolbelt import MultipartEncoder
        import json
        import requests

        img = cv2.imencode(".jpg", image)[1].tobytes()
        url = f"http://{str(cfg.host).lstrip('http://').lstrip('https://')}:{cfg.port}/detect/group"
        multipart_data = MultipartEncoder(
            fields={
                "model_hints": (None, json.dumps(models), "application/json"),
                "image": ("image.jpg", img, "image/jpeg"),
            }
        )
        headers = {
            "Content-Type": multipart_data.content_type,
        }
        r = requests.post(
            url,
            data=multipart_data,
            headers=headers,
        )
        response: Optional[List[Dict[str, Any]]] = None
        if r.status_code == 200:
            response = r.json()
        else:
            logger.error(f"{LP}detect: Got error from MLAPI (code: {r.status_code}) -> {r.text}")
        if response:
            # [ {response data from 1 model}, {response data from 2 model}, ... ]
            for result in response:
                if result.get('success') is True:
                    # m_type = result.get('type')
                    # m_proc = result.get('processor')
                    # m_name = result.get('model_name')
                    labels = result.get('label')
                    confs = result.get('confidence')
                    boxes = result.get('bounding_box')
                    for _l, _c, _b in zip(labels, confs, boxes):
                        label = _l
                        if allowed_labels and (label not in allowed_labels):
                            continue
                        confidence = _c
                        box = _b

                        x, y, _w, _h = (
                            int(round(box[0])),
                            int(round(box[1])),
                            int(round(box[2])),
                            int(round(box[3])),
                        )
                        confidence = float(confidence)
                        bbox = [
                            x,
                            y,
                            x + _w,
                            y + _h,
                        ]
                        color = [int(c) for c in COLORS[CLASS_NAMES.index(label)]]
                        cv2.rectangle(image, (x, y), (x + _w, y + _h), color, 2)
                        text = f"{label}: {confidence * 100:.2f}%"
                        # scale text based on image resolution
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5 * min(width, height) / 1000
                        thickness = max(int(font_scale * 2), 1)
                        (text_width, text_height), _ = cv2.getTextSize(
                            text, font, font_scale, thickness
                        )
                        cv2.rectangle(
                            image,
                            (x, y),
                            (x + text_width, y - text_height),
                            (
                                0,
                                0,
                                0,
                            ),
                            -1,
                        )
                        text_x = x
                        text_y = y - 5
                        cv2.putText(
                            image,
                            text,
                            (text_x, text_y),
                            font,
                            font_scale,
                            (255, 255, 255),
                            thickness,
                        )

                        # Old method
                        # x, y, _w, _h = (
                        #     int(round(box[0])),
                        #     int(round(box[1])),
                        #     int(round(box[2])),
                        #     int(round(box[3])),
                        # )
                        # confidence = float(confidence)
                        # bbox = [
                        #     x,
                        #     y,
                        #     x + _w,
                        #     y + _h,
                        # ]
                        # color = [int(c) for c in COLORS[CLASS_NAMES.index(label)]]
                        # cv2.rectangle(image, (x, y), (x + _w, y + _h), color, 2)
                        # text = f"{label}: {confidence * 100:.2f}%"
                        # # scale text based on image resolution
                        # font = cv2.FONT_HERSHEY_SIMPLEX
                        # font_scale = 0.5 * min(width, height) / 1000
                        # thickness = max(int(font_scale * 2), 1)
                        # (text_width, text_height), _ = cv2.getTextSize(
                        #     text, font, font_scale, thickness
                        # )
                        # cv2.rectangle(
                        #     image,
                        #     (x, y),
                        #     (x + text_width, y - text_height),
                        #     (
                        #         0,
                        #         0,
                        #         0,
                        #     ),
                        #     -1,
                        # )
                        # text_x = x
                        # text_y = y - 5
                        # cv2.putText(
                        #     image,
                        #     text,
                        #     (text_x, text_y),
                        #     font,
                        #     font_scale,
                        #     (255, 255, 255),
                        #     thickness,
                        # )
                    # {"success":false,"type":"object","processor":"gpu","model_name":"yolov4",
                    # "label":[],
                    # "confidence":[],
                    # "bounding_box":[]}

    else:
        raise ValueError(f"Invalid detection method {method}")
    return image


def get_files(src: Path, file_type: Optional[FileType] = None) -> List[Path]:
    """Get a list of files to process based on the type"""
    if not src.is_dir():
        raise NotADirectoryError(f"Source {src} is not a directory")
    elif not src.exists():
        raise FileNotFoundError(f"Source {src} does not exist")
    elif not src.is_absolute():
        src = src.expanduser().resolve()
    if file_type is None:
        logger.debug(f"{LP}get_files: No type specified, assuming image")
        file_type = FileType.IMAGE

    if file_type == FileType.IMAGE:
        _x = [f for f in src.rglob("*-capture.jpg")]
        return sorted(_x, key=lambda x: x.name)
    elif file_type == FileType.VIDEO:
        return [
            f
            for f in src.rglob("*.mp4")
            if f.name not in ("incomplete.mp4", "annotated.mp4")
        ]

    raise ValueError(f"Invalid file type {file_type}")


def create_writer(
    output, output_fps, gpu: bool = False, codec: Optional[str] = None
) -> Union[ffmpegcv.VideoWriterNV, ffmpegcv.VideoWriter, None]:
    writer: Union[ffmpegcv.VideoWriterNV, ffmpegcv.VideoWriter, None] = None
    if not codec:
        codec = "h264"
    resize: bool = g.output.resize.enabled
    resize_to: Optional[Tuple[int, int]] = g.output.resize.width, g.output.resize.height
    keep_aspect: bool = g.output.resize.keep_aspect
    _func = ffmpegcv.VideoWriterNV if gpu else ffmpegcv.VideoWriter
    try:
        if resize:
            logger.debug(
                f"{LP} Resizing video to width: {resize_to[0]} height: {resize_to[1]}"
            )
            if resize_to:
                writer = _func(
                    output,
                    codec=codec,
                    fps=output_fps,
                    resize=resize_to,
                    resize_keepratio=keep_aspect,
                )
        if not writer:
            writer = _func(output, codec, output_fps)
    except Exception as e:
        logger.error(f"{LP} Error creating writer: {e}")
        writer = None

    return writer


def main():
    global CLASS_NAMES, UPDATE_TIMER, ZM_DB, COLORS

    VC: Optional[cv2.VideoCapture] = None
    writer: Union[ffmpegcv.VideoWriterNV, ffmpegcv.VideoWriter]
    frame_skip: int = g.frame_skip
    method: DetectMethod = DetectMethod.LOCAL
    cfg: Union[FilterConfigFileModel.LocalSettings.LocalModelSettings, FilterConfigFileModel.RemoteSettings]
    if g.remote.enabled:
        method = DetectMethod.MLAPI
    if not CLASS_NAMES:
        logger.debug(f"{LP}main: No labels file specified, using COCO17")
        CLASS_NAMES = COCO17
    if method == DetectMethod.LOCAL:
        cfg = g.local.model
        if cfg.labels:
            if cfg.labels.is_file():
                with open(cfg.labels.as_posix(), "r") as f:
                    CLASS_NAMES = [line.strip() for line in f.readlines()]
        net: cv2.dnn.Net
        model: cv2.dnn.DetectionModel
        load_timer = time.perf_counter()
        net = cv2.dnn.readNet(cfg.input.as_posix(), cfg.config.as_posix())
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        if net is None:
            raise RuntimeError("Failed to load DarkNet network!")
        logger.info(
            f"perf:{LP} Loaded DarkNet network from {cfg.input.as_posix()} in {time.perf_counter() - load_timer:.5f}s"
        )
        model = cv2.dnn.DetectionModel(net)
        model.setInputParams(scale=1 / 255, size=(cfg.height, cfg.width), swapRB=True)
    else:
        cfg = g.remote

    is_video: bool = True
    source: List[Optional[Path]] = []
    video_files = get_files(SOURCE_DIR, FileType.VIDEO)
    image_files = get_files(SOURCE_DIR, FileType.IMAGE)
    if video_files:
        # We can get the encoding info, fps and height/width from the video file
        if len(video_files) > 1:
            logger.warning(
                f"{LP} Multiple video files found in {SOURCE_DIR}, assuming first is the correct one: {video_files[0]}"
            )
        source.append(video_files[0])
    elif image_files:
        # we need to query db for event and monitor info to encode video at correct fps
        source = image_files

    if len(source) == 0:
        logger.error(f"{LP} No source files found in {SOURCE_DIR}")
        sys.exit(1)
    elif len(source) == 1:
        logger.info(f"{LP} Found a SINGLE source file {source[0]}")
        # Make sure its not just 1 capture jpg
        is_video = source[0].suffix != ".jpg"
    elif len(source) > 1:
        logger.error(
            f"{LP} Multiple source files found in {SOURCE_DIR}, assuming jpeg images"
        )
        is_video = False
        # get event data from db
        ENV = ClientEnvVars()
        ENV.zm_conf_dir = g.zm_conf
        ENV.db = g.db
        ZM_DB = ZMDB(ENV)
        if not EVENT_ID:
            raise ValueError(
                f"{LP} No event id specified, cannot continue. Use --eid %EID% and --EPATH %EPATH% for your filter script arguments at a minimum!"
            )
        if not ZM_DB.eid_exists(EVENT_ID):
            raise ValueError(f"{LP} Event id {EVENT_ID} does not exist in db?")
        MONITOR_ID = ZM_DB._mid_from_eid(EVENT_ID)
        output_fps = float(ZM_DB._mon_fps_from_mid(MONITOR_ID))
        logger.debug(
            f"FROM ZMDB -> Target FPS for monitor {MONITOR_ID} is {output_fps}"
        )

    COLORS = np.random.randint(0, 255, size=(len(CLASS_NAMES), 3), dtype="uint8")
    output = SOURCE_DIR / "annotated.mp4"
    if output.exists():
        logger.warning(f"{LP} Output file {output} already exists, deleting")
        output.unlink()

    update_frame_count: int = g.status_updates.every
    status_updates: bool = g.status_updates.enabled
    if is_video:
        _source = source[0]
        logger.info(f"{LP} Processing video file {_source}")
        VC = cv2.VideoCapture(_source.as_posix())
        if not VC.isOpened():
            raise FileNotFoundError(f"{LP} Source video file could not be opened.")
        target_res = (
            int(VC.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(VC.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        logger.debug(f"source width/height: {target_res}")
        total_frames = VC.get(cv2.CAP_PROP_FRAME_COUNT)
        video_fps = VC.get(cv2.CAP_PROP_FPS)
        output_fps = video_fps
        writer = create_writer(output, output_fps)

        video_seconds = total_frames / video_fps
        if video_seconds > 60:
            logger.warning(
                f"{LP} Video file is longer than 60 seconds, if you are not using a GPU, this may take awhile!"
            )
        if not output.exists():
            logger.info(f"{LP} Creating output file {output}")
            output.touch(mode=0o666)
        frames_pulled = 0
        frames_processed = 0
        frames_timer = time.perf_counter()
        logger.debug(f"{LP} Starting video processing loop")
        logger.debug(
            f"skip: {int(frame_skip)} - {output_fps = } - {total_frames = } - {video_fps = } - {video_seconds = }"
        )

        for _ in range(int(total_frames) + 1):
            (grabbed, frame) = VC.read()
            if not grabbed:
                logger.warning(f"{LP} No frame returned from video file, stopping.")
                break
            if frames_pulled == 0:
                logger.debug(f"there is a frame pulled from the source video file!")
            frames_pulled += 1
            if not frames_pulled % int(frame_skip) == 0:
                writer.write(frame)
                continue
            frames_processed += 1
            if status_updates:
                complete = frames_pulled / total_frames * 100
                pull_fps = frames_pulled / (time.perf_counter() - frames_timer)
                proc_fps = frames_processed / (time.perf_counter() - frames_timer)
                eta = (total_frames - frames_pulled) / pull_fps

                if frames_processed % min(update_frame_count, total_frames) == 0:
                    update_str = f" - since last update: {timedelta(seconds=time.perf_counter() - UPDATE_TIMER) if UPDATE_TIMER else 'N/A'}"
                    logger.info(
                        f"Processing:: frame {frames_pulled} / "
                        f"{total_frames} frames ({complete:.2f}%) | "
                        f" Speed: {proc_fps:.2f} FPS (Skip: {int(frame_skip)}) -- ETA: {timedelta(seconds=eta)} -- "
                        f"ELAPSED: {timedelta(seconds=time.perf_counter() - frames_timer)}{update_str if frames_pulled > 1 else ''}"
                    )
                    UPDATE_TIMER = time.perf_counter()
            # Run a detection on the extracted frame
            if cfg.host:
                # remote
                conf = None
                nms = None
                model = None
            else:
                conf = cfg.model.confidence
                nms = cfg.model.nms


            frame = detect(
                frame,
                method,
                conf=conf,
                nms=nms,
                allowed_labels=g.annotate_labels,
                model=model,
            )
            # Write the frame to the output video
            writer.write(frame)

    else:
        logger.info("Processing image...")
        frames_timer = time.perf_counter()
        frames_processed = 0
        frames_pulled = 0
        total_frames = len(source)

        writer = create_writer(output, output_fps, gpu=False)

        if cfg.host:
            # remote
            conf = None
            nms = None
            model = None
        else:
            conf = cfg.model.confidence
            nms = cfg.model.nms
        for _file in source:
            frame = cv2.imread(_file.as_posix())
            frames_pulled += 1
            if frame is None:
                logger.error(f"{LP} Could not read image file {_file.as_posix()}")
                continue
            if g.status_updates:
                complete = frames_pulled / total_frames * 100
                pull_fps = frames_pulled / (time.perf_counter() - frames_timer)
                proc_fps = frames_processed / (time.perf_counter() - frames_timer)
                eta = (total_frames - frames_pulled) / pull_fps

                if frames_processed % min(update_frame_count, total_frames) == 0:
                    update_str = f" - since last update: {timedelta(seconds=time.perf_counter() - UPDATE_TIMER) if UPDATE_TIMER else 'N/A'}"
                    logger.info(
                        f"Processing:: frame {frames_pulled} / "
                        f"{total_frames} frames ({complete:.2f}%) | "
                        f" Speed: {proc_fps:.2f} FPS (Skip: {int(frame_skip)}) -- ETA: {timedelta(seconds=eta)} -- "
                        f"ELAPSED: {timedelta(seconds=time.perf_counter() - frames_timer)}{update_str if frames_pulled > 1 else ''}"
                    )
                    UPDATE_TIMER = time.perf_counter()

            frame = detect(
                frame,
                method,
                conf=conf,
                nms=nms,
                allowed_labels=g.annotate_labels,
                model=model,
            )
            frames_processed += 1

            writer.write(frame)
    if VC:
        VC.release()
    if writer and writer:
        writer.release()


def parse_cli():
    global CONFIG_PATH, EVENT_ID, MONITOR_ID, SOURCE_DIR

    logger.info(f"{LP} Parsing CLI arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--EID",
        "--eid",
        "--event-id",
        type=int,
        help="Event ID",
        dest="eid",
        default=0,
        required=True,
    )
    parser.add_argument(
        "--MID", "--mid", "--monitor-id", help="Monitor ID", dest="mid", default=0
    )
    parser.add_argument(
        "-C", "--config", type=Path, help="Config file", dest="config", required=True
    )
    parser.add_argument(
        "--EPATH",
        "--event-path",
        type=Path,
        help="Event path",
        dest="event_path",
        default=None,
        required=True,
    )
    args, unknown_args = parser.parse_known_args()
    logger.info(f"{LP} Parsed KNOWN CLI arguments: {args}")
    logger.info(f"{LP} Parsed UNKNOWN CLI arguments: {unknown_args}")
    if args.event_path:
        logger.info(f"{LP} Setting SOURCE_DIR to {args.event_path}")
        SOURCE_DIR = args.event_path
    if args.eid:
        EVENT_ID = args.eid
    if args.mid:
        try:
            MONITOR_ID = int(args.mid)
        except ValueError:
            logger.error(
                f"{LP} Monitor ID must be an integer, got {args.mid} ({type(args.mid)})"
            )
    if args.config:
        CONFIG_PATH = args.config


def get_event_data(db: ZMDB, eid: int):
    lp: str = "db:event data:"
    logger.info(f"{lp} Getting event data for event {eid}")
    event_data = db.get_event_data(eid)
    if not event_data:
        logger.error(f"{LP} Could not get event data for event {eid}")
        raise ValueError(f"Could not get event data for event {eid}")
    return event_data


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    # display the command line arguments
    # cli = {
    #     k: v for k, v in dict(enumerate(sys.argv)).items() if k > 0
    # }
    # logger.info(f"{LP} Command line arguments: {cli}")
    parse_cli()
    g: Optional[FilterConfigFileModel] = None
    if CONFIG_PATH and CONFIG_PATH.is_file():
        g = parse_cfg(CONFIG_PATH, FilterConfigFileModel)
        DEBUG = g.logging.debug
        fh = logging.FileHandler(g.logging.file.path / g.logging.file.filename)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        if DEBUG:
            logger.setLevel(logging.DEBUG)
            ch.setLevel(logging.DEBUG)
            fh.setLevel(logging.DEBUG)
        # set zm ml client logger
        logger.info(f"{LP} Parsed config file: {CONFIG_PATH}")
    if not SOURCE_DIR.is_dir():
        raise NotADirectoryError(f"Source {SOURCE_DIR} is not a directory")
    elif not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source {SOURCE_DIR} does not exist")
    elif not SOURCE_DIR.is_absolute():
        SOURCE_DIR = SOURCE_DIR.expanduser().resolve()

    main()
    logger.info(f"{LP} Done! check {SOURCE_DIR}")
