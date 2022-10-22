from contextlib import contextmanager
from decimal import Decimal
from logging import getLogger
from time import sleep
from typing import Optional, Set, Any

import requests

from main import get_global_config
from models.config import APIPullMethod

logger = getLogger("zm_ml")
LP = "API Images:"
g = get_global_config()


class APIImagePipeLine:
    def __iter__(self):
        logger.debug(f"{LP} __iter__ called on {__class__.__name__}")
        if self.more():
            yield self.get_image()

    def __init__(
        self,
        options: APIPullMethod,
    ):
        lp = f"images:init:"
        if not options:
            raise ValueError(f"{lp} no stream options provided!")
        #  INIT START 
        self.options = options
        self.event_tot_frames: int = 0
        self.has_event_ended: str = ""
        logger.debug(f"{lp} options: {self.options}")

        #  FRAME IDS 
        self._attempted_fids: Set[int] = set()  # All tried
        self._processed_fids: Set[int] = set()  # All successful
        self._skipped_fids: Set[int] = set()  # All skipped

        #  FRAME BUFFER 
        self.total_min_frames: int = 1
        self._current_frame: int = 0
        self.current_snapshot: int = 0
        self.last_fid_read: int = 0
        self.last_snapshot_id: int = 0
        self.fps = int(Decimal(g.mon_fps).quantize(Decimal("1")))
        self.buffer_pre_count = g.mon_pre
        self.buffer_post_count = g.mon_post

        self.max_attempts = options.attempts
        self.max_attempts_delay = options.attempt_delay

        #  Check if delay is configured 
        if not g.past_event and (delay := options.delay):
            if delay:
                logger.debug(
                    f"{LP} a delay is configured, this only applies one time - waiting for {delay} "
                    f"second(s) before starting"
                )
                sleep(delay)

        # Alarm frame is always the first frame, pre count buffer length+1 for alarm frame
        self.current_frame = self.buffer_pre_count + 1
        # The pre- / post-buffers will give the absolute minimum number of frames to grab, assuming no event issues
        self.total_min_frames = int(
            (self.buffer_post_count + self.buffer_pre_count) / self.fps
        )
        # We don't know how long an event will be so set an upper limit of at least
        # pre- + post-buffers calculated as seconds because we are pulling 1 FPS
        self.total_max_frames = max(10, self.total_min_frames)

    def get_image(self):
        def _grab_event_data(msg: Optional[str] = None):
            """Calls global API make_request method to get event data"""
            if not self.event_ended:
                if msg:
                    logger.debug(f"{LP}read>event_data: {msg}")
                try:
                    g.Event, g.Monitor, g.Frame = g.api.get_all_event_data(g.eid)
                except Exception as e:
                    logger.error(f"{lp} error grabbing event data from API -> {e}")
                    raise e
                else:
                    self.event_tot_frames = int(g.Event.get("Frames", 0))
                    self.has_event_ended = g.Event.get("EndDateTime", "")
                    logger.debug(
                        f"{lp} grabbed event data from ZM API for event '{g.eid}' -- event total Frames(g.Event['fr"
                        f"ames']): {self.event_tot_frames} -- EndDateTime: {self.has_event_ended} -- "
                        f"has event ended: {self.event_ended} -- {len(g.Frame) = } -- "
                        f"g.Frame and g.Event['Frame'] equal: {len(g.Frame) == self.event_tot_frames}"
                    )
                    if self.event_ended:
                        logger.debug(
                            f"DBG => THIS EVENT HAS AN EndDateTime ({self.has_event_ended}) checking max_frames "
                            f"and modifying if needed"
                        )
                        new_max = int(self.event_tot_frames / self.fps)
                        logger.debug(
                            f"DEBUG>>>{LP} current max: {self.total_max_frames} new_max: {new_max} (event_total_frames"
                            f"[{self.event_tot_frames}] / fps[{self.fps}])"
                        )
                        if new_max > self.total_max_frames:
                            logger.debug(
                                f"{lp} max_frames ({self.total_max_frames}) is lower than current calculations, "
                                f"setting to {new_max}"
                            )
                            self.total_max_frames = new_max
            else:
                logger.debug(f"{lp} event has ended, no need to grab event data")

        response: Optional[requests.Response] = None
        lp = f"{LP}read:"
        if self.frames_processed > 0:
            logger.debug(
                f"{lp} [{self.frames_processed}/{self.total_max_frames} frames processed: {self._processed_fids}] "
                f"- [{self.frames_skipped}/{self.total_max_frames} frames skipped: {self._skipped_fids}] - "
                f"[{self.frames_attempted}/{self.total_max_frames} frames attempted: {self._attempted_fids}]"
            )
        else:
            logger.debug(f"{lp} processing first frame!")
        curr_snapshot = None
        if self.options.check_snapshots:
            logger.debug(f"{lp} checking snapshot ids enabled!")
            # Check if event data available or get data for snapshot fid comparison
            if (not g.past_event) and (
                (self.frames_processed >= 0 and self.last_snapshot_id >= 0)
                and self.frames_processed % self.options.snapshot_frame_skip
                == 0  # Only run every <x> frames
            ):
                _grab_event_data(msg=f"grabbing data for snapshot comparisons...")
                if curr_snapshot := int(g.Event.get("MaxScoreFrameId", 0)):
                    if self.last_snapshot_id and curr_snapshot > self.last_snapshot_id:
                        logger.debug(
                            f"{lp} current snapshot frame id is not the same as the last snapshot id "
                            f"CURR:{curr_snapshot} - PREV:{self.last_snapshot_id}, grabbing new snapshot image"
                        )
                        self.current_frame = curr_snapshot
                    self.last_snapshot_id = curr_snapshot
                else:
                    logger.warning(
                        f"{lp} Event: {g.eid} - No Snapshot Frame ID found in ZM API? -> {g.Event = }",
                    )
            if not g.Event:
                _grab_event_data(msg="NO EVENT DATA!!! grabbing from API...")

            #  Check if we have already processed this frame ID 
            if self.current_frame in self._processed_fids:
                logger.debug(
                    f"{lp} skipping Frame ID: '{self.current_frame}' as it has already been"
                    f" processed for event {g.eid}"
                )
                return self._process_frame(skip=True)
            #  SET URL TO GRAB IMAGE FROM 
            fid_url = f"{g.api.get_portalbase()}/index.php?view=image&eid={g.eid}&fid={self.current_frame}"

            if g.past_event:
                logger.warning(
                    f"{lp} this is a past event, max image grab attempts set to 1"
                )
                self.max_attempts = 1
            for image_grab_attempt in range(self.max_attempts):
                image_grab_attempt += 1
                logger.debug(
                    f"{lp} attempt #{image_grab_attempt}/{self.max_attempts} to grab image from URL: {fid_url}"
                )
                response = g.api.make_request(fid_url)
                if (
                    response
                    and isinstance(response, requests.Response)
                    and response.status_code == 200
                ):
                    img = bytearray(response.content)
                    return self._process_frame(image=img)
                # response code not 200 or no response
                else:
                    resp_msg = ""
                    if response:
                        resp_msg = f" response code={response.status_code} - response={response}"
                    else:
                        resp_msg = f" no response received!"
                    logger.warning(f"{lp} image was not retrieved!{resp_msg}")

                    _grab_event_data(msg="checking if event has ended...")

                    if self.event_ended:  # Assuming event has ended
                        logger.debug(f"{lp} event has ended, checking OOB status...")
                        # is current frame OOB
                        if self.current_frame > self.event_tot_frames:
                            # We are OOB, so we are done
                            logger.debug(
                                f"{lp} we are OOB in a FINISHED event (current requested fid: {self.current_frame} > "
                                f"total frames in event: {self.event_tot_frames})"
                            )
                            return self._process_frame(end=True)
                    else:
                        logger.debug(
                            f"{lp} event has not ended yet! Total Frames: {self.event_tot_frames}"
                        )
                    if not g.past_event and (image_grab_attempt < self.max_attempts):
                        logger.debug(
                            f"{lp} sleeping for {self.options.attempt_delay} second(s)"
                        )
                        sleep(self.options.attempt_delay)

            return self._process_frame(skip=True)

    def more(self) -> bool:
        logger.debug(
            f"{LP} DBG => more() called {self.frames_processed = } -- {self.total_max_frames = } -- "
            f"{self.frames_processed < self.total_max_frames = }"
        )
        return self.frames_processed < self.total_max_frames

    def _process_frame(
        self,
        image: bytearray = None,
        skip: bool = False,
        end: bool = False,
    ):
        """Process the frame, increment counters, and return the image if there is one"""
        lp = f"{LP}processed_frame:"
        self.last_fid_read = self.current_frame
        self._attempted_fids.add(self.current_frame)
        if skip:
            self._skipped_fids.add(self.current_frame)
        else:
            self._processed_fids.add(self.current_frame)

        if end or self.frames_processed > self.total_max_frames:
            _msg = (
                "end has been called, no more images to process!"
                if end
                else f"max_frames ({self.total_max_frames}) has been reached, stopping!"
            )
            logger.error(f"{lp} {_msg}")
        elif not end:
            self.current_frame = self.current_frame + self.fps
            logger.debug(
                f"{lp} incrementing next frame ID to read by {self.fps} = {self.current_frame}"
            )
        if image:
            # (bytearray, image_file_name)
            return image, f"mid_{g.mid}-eid_{g.eid}-fid_{self.current_frame}.jpg"
        return None, None

    @property
    def event_ended(self):
        if self.has_event_ended:
            return True
        return False

    @property
    def current_frame(self):
        return self._current_frame

    @current_frame.setter
    def current_frame(self, value):
        self._current_frame = value

    @property
    def frames_skipped(self):
        return len(self._skipped_fids) or 0

    @property
    def frames_attempted(self):
        return len(self._attempted_fids) or 0

    @property
    def last_read_fid(self) -> int:
        return self.last_fid_read or 0

    @last_read_fid.setter
    def last_read_fid(self, value):
        self.last_fid_read = value

    @property
    def total_max_frames(self):
        return self._max_frames or 0

    @total_max_frames.setter
    def total_max_frames(self, value):
        self._max_frames = value

    @property
    def frames_processed(self):
        return len(self._processed_fids) or 0


class SHMImagePipeLine:
    offset = None


class ZMUImagePipeLine:
    offset = None



#! /usr/bin/python3
import mmap
import struct
from _ctypes import Structure, Union as CUnion
from ctypes import c_long, c_uint64, c_int64, sizeof
from argparse import ArgumentParser
from collections import namedtuple
from typing import Optional, IO, Union
from sys import maxsize as sys_maxsize

from sqlalchemy import MetaData
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.future import Engine, create_engine, Connection, select


IS_64BITS = sys_maxsize > 2**32
IMAGE_BUFFER_COUNT: int = 5
MID: Union[str, int] = "1"
WIDTH: int = 0
HEIGHT: int = 0
COLORSPACE: int = 4
DB_ENGINE: Optional[Engine] = None
ENV: str = ".env"
# This will compensate for 32/64 bit
struct_time_stamp = r"l"
TIMEVAL_SIZE: int = struct.calcsize(struct_time_stamp)


# Dynamically create timeval Struct to calculate size / 32 == 8 bytes // 64 == 16 bytes
def create_timeval():
    tv_usec = c_long
    # print("BEFORE CHECKING 64 BITNESS -> tv_usec: ", tv_usec)
    if IS_64BITS:
        # print("64 BITS is TRUE")
        tv_usec = c_uint64
    # print("AFTER CHECKING 64 BITNESS -> tv_usec: ", tv_usec)
    _fields = (("tv_sec", c_long,), ("tv_usec", tv_usec,))
    _class = type("timeval", (Structure,), {"_fields_": _fields})
    return _class()

# timeval = create_timeval()
# TIMEVAL_SIZE = sizeof(timeval)


def _db_create() -> Engine:
    lp: str = "ZM-DB:"
    db_config = {
        "dbuser": "zmuser",
        "dbpassword": "zmpass",
        "dbhost": "localhost",
        "dbname": "zm",
        "driver": "mysql+mysqlconnector",
    }
    connection_str = (
        f"{db_config['driver']}://{db_config['dbuser']}"
        f":{db_config['dbpassword']}@{db_config['dbhost']}"
        f"/{db_config['dbname']}"
    )

    try:
        engine: Optional[Engine] = create_engine(connection_str, pool_recycle=3600)
        meta: MetaData = MetaData(engine)
        # New reflection method, only reflect the Events and Monitors tables
        meta.reflect(only=["Monitors"])
        conn: Optional[Connection] = engine.connect()
    except SQLAlchemyError as e:
        conn = None
        engine = None
        print(f"DB configs - {connection_str}")
        print(f"Could not connect to DB, message was: {e}")
    else:


        conn.close()
        return engine

def zm_version(ver: str, minx: Optional[int] = None, patchx: Optional[int] = None) -> int:

    maj, min, patch = "", "", ""
    x = ver.split(".")
    x_len = len(x)
    if x_len <= 2:
        maj, min = x
        patch = "0"
    elif x_len == 3:
        maj, min, patch = x
    else:
        print("come and fix me!?!?!")
    maj = int(maj)
    min = int(min)
    patch = int(patch)
    if minx:
        if minx > min:
            return 1
        elif minx == min:
            if patchx:
                if patchx > patch:
                    return 1
                else:
                    return 0
            else:
                return 0
        else:
            return 0


class ZMMemory:
    def __init__(
        self, path: Optional[str] = None, mid: Optional[Union[int, str]] = None
    ):
        global MID
        if mid:
            MID = mid
        if path is None:
            path = f"/dev/shm"

        self.alarm_state_stages = {
            "STATE_IDLE": 0,
            "STATE_PREALARM": 1,
            "STATE_ALARM": 2,
            "STATE_ALERT": 3,
            "STATE_TAPE": 4,
            "ACTION_GET": 5,
            "ACTION_SET": 6,
            "ACTION_RELOAD": 7,
            "ACTION_SUSPEND": 8,
            "ACTION_RESUME": 9,
            "TRIGGER_CANCEL": 10,
            "TRIGGER_ON": 11,
            "TRIGGER_OFF": 12,
        }
        self.fhandle: Optional[IO] = None
        self.mhandle: Optional[mmap.mmap] = None

        self.fname = f"{path}/zm.mmap.{mid}"
        self.reload()

    def reload(self):
        """Reloads monitor information. Call after you get
        an invalid memory report

        Raises:
            ValueError: if no monitor is provided
        """
        # close file handler
        self.close()
        # open file handler in read binary mode
        self.fhandle = open(self.fname, "r+b")
        # geta rough size of the memory consumed by object (doesn't follow links or weak ref)
        from os.path import getsize
        sz = getsize(self.fname)
        if not sz:
            raise ValueError(f"Invalid size: {sz} of {self.fname}")

        self.mhandle = mmap.mmap(self.fhandle.fileno(), 0, access=mmap.ACCESS_READ)
        self.sd = None
        self.td = None
        self._read()

    def is_valid(self):
        """True if the memory handle is valid

        Returns:
            bool: True if memory handle is valid
        """
        try:
            d = self._read()
            return not d["shared_data"]["size"] == 0
        except Exception as e:
            print(f"ERROR!!!! =-> Memory: {e}")
            return False

    def _read(self):
        self.mhandle.seek(0)  # goto beginning of file
        SharedData = namedtuple(
            "SharedData",
            "size last_write_index last_read_index state capture_fps analysis_fps last_event_id action brightness "
            "hue colour contrast alarm_x alarm_y valid capturing analysing recording signal format imagesize "
            "last_frame_score audio_frequency audio_channels startup_time zmc_heartbeat_time last_write_time "
            "last_read_time last_viewed_time control_state alarm_cause video_fifo_path audio_fifo_path",
        )

        # old_shared_data = r"IIIIQIiiiiii????IIQQQ256s256s"
        # I = uint32 - i = int32 == 4 bytes ; int
        # Q = uint64 - q = int64 == 8 bytes ; long long int
        # L = uint64 - l = int64 == 8 bytes ; long int
        # d = double == 8 bytes ; float
        # s = char[] == n bytes ; string
        # B = uint8 - b = int8 == 1 byte; char

        # shared data bytes is now aligned as of commit 590697b (1.37.19) -> SEE
        # https://github.com/ZoneMinder/zoneminder/commit/590697bd807ab9a74d605122ef0be4a094db9605
        # Before it was 776 for 64bit and 772 for 32 bit
        ZM_VER = "1.37.19"

        shared_data_bytes = 776  # bytes in SharedData
        struct_shared_data = r"I 2i I 2d Q I 6i 6B 4I 5L 256s 256s 64s 64s"  # July 2022

        TriggerData = namedtuple(
            "TriggerData",
            "size trigger_state trigger_score padding trigger_cause trigger_text trigger_showtext",
        )
        struct_trigger_data = r"IIII32s256s256s"
        trigger_data_bytes = 560  # bytes in TriggerData - 32/64 bit SAME

        VideoStoreData = namedtuple(
            "VideoStoreData",
            "size padding current_event event_file recording",
        )
        struct_video_store = r"IIQ4096sq"
        video_store_bytes = 4120 if IS_64BITS else 4116  # bytes in VideoStoreData for 64bit / 32bit = 4116 - July 2022

        TimeStampData = namedtuple(
            "TimeStampData",
            "timeval",
        )
        # struct_time_stamp = r"lq" if IS_64BITS else r"ll"
        # time_stamp_bytes = TIMEVAL_SIZE  # bytes in TimeStampData

        s = SharedData._make(
            struct.unpack(struct_shared_data, self.mhandle.read(shared_data_bytes))
        )
        t = TriggerData._make(
            struct.unpack(struct_trigger_data, self.mhandle.read(trigger_data_bytes))
        )
        v = VideoStoreData._make(
            struct.unpack(struct_video_store, self.mhandle.read(video_store_bytes))
        )

        ts = TimeStampData._make(
            struct.unpack(struct_time_stamp, self.mhandle.read(TIMEVAL_SIZE))
        )
        written_images = s.last_write_index + 1

        self.images_offset = self.mhandle.tell()
        print(f"{written_images = } - {self.images_offset = }")
        print(f"\nSharedData = {s}\n")
        # grab available images? - make context manager to yield images form the ring buffer?
        sh_img_data = self.mhandle.read(written_images * s.imagesize)
        import cv2
        import numpy as np
        print(f"Converting images into cv2")
        self.shared_images = []
        # grab total image buffer
        image_buffer = self.mhandle.read(s.imagesize * written_images)
        # convert bytes to numpy array to cv2 images
        for i in range(written_images):
            img = np.frombuffer(image_buffer[i * s.imagesize : (i + 1) * s.imagesize], dtype=np.uint8)
            if img.size == s.imagesize:
                print(f"Image index {i} is of the correct size, reshaping and converting")
                img = img.reshape((HEIGHT, WIDTH, COLORSPACE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.shared_images.append(img)
            else:
                print(f"Invalid image size: {img.size}")
        # show images
        for img in self.shared_images:
            cv2.imshow("img", img)
            cv2.waitKey(2000)

        self.sd = s._asdict()
        self.td = t._asdict()
        self.vsd = v._asdict()
        self.tsd = ts._asdict()

        self.sd["alarm_cause"] = self.sd["alarm_cause"].split(b"\0", 1)[0].decode()
        self.sd["control_state"] = self.sd["control_state"].split(b"\0", 1)[0].decode()
        self.td["trigger_cause"] = self.td["trigger_cause"].split(b"\0", 1)[0].decode()
        self.td["trigger_text"] = self.td["trigger_text"].split(b"\0", 1)[0].decode()
        self.td["trigger_showtext"] = (
            self.td["trigger_showtext"].split(b"\0", 1)[0].decode()
        )
        return {
            "shared_data": self.sd,
            "trigger_data": self.td,
            "video_store_data": self.vsd,
            "time_stamp_data": self.tsd,
            "shared_images": self.shared_images,
        }

    def close(self):
        """Closes the handle"""
        try:
            if self.mhandle:
                self.mhandle.close()
            if self.fhandle:
                self.fhandle.close()
        except Exception:
            pass


def parse_cli_args():
    """Parse CLI arguments into a dict"""
    global MID, ENV
    ap = ArgumentParser()

    ap.add_argument(
        "-m",
        "--mid",
        "--monitor-id",
        type=int,
        dest="mid",
        help="monitor id - For use by the PERL script (Automatically found)",
    )
    ap.add_argument(
        "-e",
        "--env-file",
        dest="ENV",
        help="environment file to parse"
    )
    args, u = ap.parse_known_args()
    args = vars(args)
    print(f"{args = }")
    if not args.get("mid"):
        print(f'MONITOR ID not passed via CLI, using {MID}')
    else:
        MID = args["mid"]
        print(f'MONITOR ID passed via CLI, using {MID}')
    if not args.get("ENV"):
        print(f'ENVIRONMENT FILE not passed via CLI, using {ENV}')
    else:
        ENV = args["ENV"]
        print(f'ENVIRONMENT FILE passed via CLI, using {ENV}')
    if not args:
        print(f"ERROR-FATAL -> no args!")
        exit(1)
    return args


if __name__ == "__main__":
    print('script started')
    args = parse_cli_args()
    print(f"About to do DB stuff")
    engine = _db_create()
    print(f"{TIMEVAL_SIZE = }")
    print('doing shm stuff')
    zm_mem = ZMMemory(mid=MID)
