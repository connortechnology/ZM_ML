#!/usr/bin/env python3
"""
A script to be used with ZoneMinder filter system. The filtering system can pass some CLI args
to the script. This is the beggining of the exploratory work on how to integrate this.

 This has 2 modes.

 - config mode: you can use ZM ML like monitor config to only annotate certain objects
     - This allows for filtering by label/group/zone
 - all: annotate anything it finds in the event
"""
import argparse
import logging
import time
import warnings
import os
import sys
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import List, Optional, NamedTuple, Tuple, Dict

from pydantic import BaseModel, Field, validator

from zm_ml.Client.Models.config import DetectionSettings, SystemSettings, ZMAPISettings, \
    MatchingSettings, MonitorsSettings, MatchFilters
from zm_ml.Client.main import _replace_vars, parse_client_config_file as parse_cfg, set_logger
from zm_ml.Server.ML.coco17_cv2 import COCO17
from zm_ml.Shared.Models.config import Testing
# read data from zm db
from zm_ml.Client.Libs.zmdb import ZMDB


try:
    try:
        import cv2
    except ImportError:
        warnings.warn("OpenCV not installed, will not be able to annotate images")
    # Jetbrains hack
except ImportError:
    pass
    if not cv2:
        cv2 = None
    # raise
try:
    import numpy as np
except ImportError:
    warnings.warn("Numpy not installed, will not be able to annotate images")
    # raise

LP: str = "filter:annotate:"
SOURCE_DIR: Path
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
fh = logging.FileHandler("/var/log/zm/annotate_events.log")
fh.setFormatter(formatter)
logger.addHandler(fh)
# set zm ml client logger
set_logger(logger)

NET: Optional[cv2.dnn.Net] = None
MODEL: Optional[cv2.dnn.DetectionModel] = None
# Model class names
CLASS_NAMES: List[str] = COCO17
# Colors for bounding boxes
COLORS: Optional[np.ndarray] = None
# Input model
WEIGHTS: Optional[Path] = Path("/shared/models/yolov4/yolov4_new.weights")
# Input model config file (optional)
MODEL_CFG: Optional[Path] = Path("/shared/models/yolov4/yolov4_new.cfg")
# Model input size
MODEL_H: int = 416
MODEL_W: int = 416
CONFIDENCE_THRESHOLD: float = 0.5
NMS_THRESHOLD: float = 0.3
ALLOWED_LABELS: List[str] = ["person", "car", "truck", "motorbike", "bicycle", "bus", "cat", "dog", "boat"]
WRITER: Optional[cv2.VideoWriter] = None
VC: Optional[cv2.VideoCapture] = None
IS_VIDEO: bool = False
# log status updates of the video file processing
STATUS_UPDATES: bool = True
# how often to log status updates
UPDATE_FRAME_COUNT: int = 10
# Resize the output video from whatever the source is
RESIZE: bool = False
# Resize to this size (H, W)
RESIZE_TO: Optional[Tuple[int, int]] = None
DEBUG: bool = True
EVENT_ID: int = 0
MONITOR_ID: int = 0
CONFIG_PATH: Optional[Path] = None
FRAME_SKIP: int = 1
UPDATE_TIMER: float = 0.0
TARGET_FPS: int = 1
ZM_DB: Optional[ZMDB] = None
# CONFIG_PATH = Path("/opt/zm_ml/configs/example_annotate_filter.yml")


class FilterConfigFileModel(BaseModel):
    testing: Testing = Field(default_factory=Testing)
    substitutions: Dict[str, str] = Field(default_factory=dict)
    system: SystemSettings = Field(default_factory=SystemSettings)
    zoneminder: ZMAPISettings = Field(default_factory=ZMAPISettings)
    label_groups: Dict[str, List[str]] = Field(default_factory=dict)
    models: Dict = Field(default_factory=dict)
    import_zones: bool = Field(False)
    filters: MatchFilters = Field(default_factory=MatchFilters)
    monitors: Dict[int, MonitorsSettings] = Field(default_factory=dict)

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


def detect(image: np.ndarray, method: DetectMethod):
    """Run detection on a frame"""
    global NET, COLORS, CLASS_NAMES, MODEL, WRITER
    height, width, _ = image.shape

    if not method:
        raise ValueError("No detection method specified")
    elif method == DetectMethod.LOCAL:
        # run inference locally
        _l, _c, _b = MODEL.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        for class_id, confidence, box in zip(_l, _c, _b):
            label = CLASS_NAMES[class_id]

            # TODO: filter if in config mode
            if ALLOWED_LABELS and (label not in ALLOWED_LABELS):
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
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(image, (x, y), (x + text_width, y - text_height), (0, 0, 0,), -1)
            text_x = x
            text_y = y - 5
            cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

            # cv2.putText(
            #     image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            # )
    elif method == DetectMethod.MLAPI:
        # Send to zm mlapi server
        pass
    else:
        raise ValueError(f"Invalid detection method {method}")
    return image


def write_frame(frame: np.ndarray):
    """Write a frame to the video writer"""
    if WRITER is None:
        raise ValueError("Video writer not initialized")
    elif frame is None:
        logger.warning(f"{LP}write_frame: Frame is None, skipping")
        return
    # Dont write an empty frame
    if not frame.any():
        logger.warning(f"{LP}write_frame: Empty frame, skipping")
        return
    WRITER.write(frame)


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
        return [f for f in src.rglob("*.mp4") if f.name not in ("incomplete.mp4", "annotated.mp4")]

    raise ValueError(f"Invalid file type {file_type}")


def main():
    global SOURCE_DIR, WRITER, MODEL, NET, COLORS, \
            CLASS_NAMES, WEIGHTS, MODEL_CFG, MODEL_H,\
            MODEL_W, VC, IS_VIDEO, FRAME_SKIP, UPDATE_TIMER, \
            TARGET_FPS
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
        event_info = get_event_info(ZMDB())
        source = image_files

    if len(source) == 0:
        logger.error(f"{LP} No source files found in {SOURCE_DIR}")
        sys.exit(1)
    elif len(source) == 1:
        logger.info(f"{LP} Found a SINGLE source file {source[0]}")
        # Make sure its not just 1 capture jpg
        IS_VIDEO = source[0].suffix != ".jpg"
    elif len(source) > 1:
        logger.error(
            f"{LP} Multiple source files found in {SOURCE_DIR}, assuming jpeg images"
        )
        IS_VIDEO = False

    if not COLORS:
        COLORS = np.random.randint(0, 255, size=(len(CLASS_NAMES), 3), dtype="uint8")

    if not NET:
        load_timer = time.perf_counter()
        NET = cv2.dnn.readNet(WEIGHTS.as_posix(), MODEL_CFG.as_posix())
        NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        NET.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        if NET is None:
            raise RuntimeError("Failed to load DarkNet network!")
        logger.info(
            f"perf:{LP} Loaded DarkNet network from {WEIGHTS.as_posix()} in {time.perf_counter() - load_timer:.5f}s"
        )
    if not MODEL:
        load_timer = time.perf_counter()
        MODEL = cv2.dnn.DetectionModel(NET)
        MODEL.setInputParams(scale=1 / 255, size=(MODEL_H, MODEL_W), swapRB=True)
        logger.info(
            f"perf:{LP} Initializing DetectionModel() completed in {time.perf_counter() - load_timer:.5f}s"
        )

    output = (SOURCE_DIR / "annotated.mp4")
    if output.exists():
        logger.warning(f"{LP} Output file {output} already exists, deleting")
        output.unlink()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if IS_VIDEO:
        _source = source[0]
        logger.info(f"{LP} Processing video file {_source}")
        VC = cv2.VideoCapture(_source.as_posix())
        if not VC.isOpened():
            raise FileNotFoundError(f"{LP} Source video file could not be opened.")
        target_res = (int(VC.get(cv2.CAP_PROP_FRAME_WIDTH)), int(VC.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        logger.debug(f"{LP} Target resolution: {target_res} ({type(target_res)}) ||| fourcc: {fourcc} ({type(fourcc)})")
        total_frames = VC.get(cv2.CAP_PROP_FRAME_COUNT)
        video_fps = VC.get(cv2.CAP_PROP_FPS)
        video_seconds = total_frames / video_fps
        if video_seconds > 60:
            logger.warning(
                f"{LP} Video file is {video_seconds:.2f}s long @ {video_fps} FPS, creating a 1fps video with 1 fps of source video"
            )
            FRAME_SKIP = video_fps / 2
            TARGET_FPS = 1
        else:
            TARGET_FPS = video_fps
        if not output.exists():
            logger.info(f"{LP} Creating output file {output}")
            output.touch(mode=0o666)
        if not WRITER:
            logger.info(f"{LP} Writing output to {output}")
            WRITER = cv2.VideoWriter(
                output.as_posix(),
                fourcc,
                float(TARGET_FPS),
                target_res,
                True,
            )
        frames_pulled = 0
        frames_processed = 0
        frames_timer = time.perf_counter()
        logger.debug(f"{LP} Starting video processing loop")
        logger.debug(f"skip: {int(FRAME_SKIP)} - {TARGET_FPS = } - {total_frames = } - {video_fps = } - {video_seconds = }")

        for _ in range(int(total_frames) + 1):
            (grabbed, frame) = VC.read()
            if not grabbed:
                logger.warning(f"{LP} No frame returned from video file, stopping.")
                break
            if frames_pulled == 0:
                logger.debug(f"there is a frame pulled from the source video file!")
            frames_pulled += 1
            if not frames_pulled % int(FRAME_SKIP) == 0:
                continue
            frames_processed += 1
            if STATUS_UPDATES:
                complete = frames_pulled / total_frames * 100
                pull_fps = frames_pulled / (time.perf_counter() - frames_timer)
                proc_fps = frames_processed / (time.perf_counter() - frames_timer)
                eta = (total_frames - frames_pulled) / pull_fps

                if frames_processed % min(UPDATE_FRAME_COUNT, total_frames) == 0:
                    update_str = f" - since last update: {timedelta(seconds=time.perf_counter() - UPDATE_TIMER) if UPDATE_TIMER else 'N/A'}"
                    logger.info(
                        f"Processing:: frame {frames_pulled} / "
                        f"{total_frames} frames ({complete:.2f}%) | "
                        f" Speed: {proc_fps:.2f} FPS (Skip: {int(FRAME_SKIP)}) -- ETA: {timedelta(seconds=eta)} -- "
                        f"ELAPSED: {timedelta(seconds=time.perf_counter() - frames_timer)}{update_str if frames_pulled > 1 else ''}"
                    )
                    UPDATE_TIMER = time.perf_counter()
            # Resize if needed
            if RESIZE and (RESIZE_TO):
                frame = cv2.resize(frame, RESIZE_TO)  # H, W
            # Run a detection on the extracted frame
            frame = detect(frame, DetectMethod.LOCAL)
            # Write the frame to the output video
            write_frame(frame)

    else:
        logger.info("Processing image...")
        frames_timer = time.perf_counter()
        frames_processed = 0
        frames_pulled = 0
        total_frames = len(source)

        for _file in source:
            frame = cv2.imread(_file.as_posix())
            frames_pulled += 1
            if frame is None:
                logger.error(f"{LP} Could not read image file {_file.as_posix()}")
                continue
            if STATUS_UPDATES:
                complete = frames_pulled / total_frames * 100
                pull_fps = frames_pulled / (time.perf_counter() - frames_timer)
                proc_fps = frames_processed / (time.perf_counter() - frames_timer)
                eta = (total_frames - frames_pulled) / pull_fps

                if frames_processed % min(UPDATE_FRAME_COUNT, total_frames) == 0:
                    update_str = f" - since last update: {timedelta(seconds=time.perf_counter() - UPDATE_TIMER) if UPDATE_TIMER else 'N/A'}"
                    logger.info(
                        f"Processing:: frame {frames_pulled} / "
                        f"{total_frames} frames ({complete:.2f}%) | "
                        f" Speed: {proc_fps:.2f} FPS (Skip: {int(FRAME_SKIP)}) -- ETA: {timedelta(seconds=eta)} -- "
                        f"ELAPSED: {timedelta(seconds=time.perf_counter() - frames_timer)}{update_str if frames_pulled > 1 else ''}"
                    )
                    UPDATE_TIMER = time.perf_counter()
            if RESIZE and (RESIZE_TO):
                frame = cv2.resize(frame, RESIZE_TO)
            if not WRITER:
                logger.info(f"{LP} Writing output to {output}")
                WRITER = cv2.VideoWriter(
                    output.as_posix(),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(10),
                    (frame.shape[1], frame.shape[0]),
                )
            #
            frame = detect(frame, DetectMethod.LOCAL)
            frames_processed += 1

            write_frame(frame)


def parse_cli():
    global CONFIG_PATH, EVENT_ID, MONITOR_ID, SOURCE_DIR

    logger.info(f"{LP} Parsing CLI arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--EID", "--eid", "--event-id", type=int, help="Event ID", dest="eid", default=0)
    parser.add_argument("--MID", "--mid", "--monitor-id", help="Monitor ID", dest="mid", default=0)
    parser.add_argument("-C", "--config", type=Path, help="Config file", dest="config")
    parser.add_argument("--EPATH", "--event-path", type=Path, help="Event path", dest="event_path", default=None)
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
            logger.error(f"{LP} Monitor ID must be an integer, got {args.mid} ({type(args.mid)})")
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
    if CONFIG_PATH and CONFIG_PATH.is_file():
        logger.debug(f"{LP} Config file path: {CONFIG_PATH}")
        parsed_cfg = parse_cfg(CONFIG_PATH, FilterConfigFileModel)
        logger.info(f"{LP} Parsed config file \n\n{parsed_cfg}\n\n")
    if not SOURCE_DIR.is_dir():
        raise NotADirectoryError(f"Source {SOURCE_DIR} is not a directory")
    elif not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source {SOURCE_DIR} does not exist")
    elif not SOURCE_DIR.is_absolute():
        SOURCE_DIR = SOURCE_DIR.expanduser().resolve()

    main()
    if VC:
        VC.release()
    if WRITER:
        WRITER.release()
    logger.info(f"{LP} Done! check {SOURCE_DIR}")
