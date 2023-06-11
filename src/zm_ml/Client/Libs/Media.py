from __future__ import annotations
import mmap
import struct
from _ctypes import Structure
from collections import namedtuple
from ctypes import c_long, c_uint64
from decimal import Decimal
from enum import IntEnum
import logging
from sys import maxsize as sys_maxsize
from time import sleep
from typing import Optional, IO, Union, TYPE_CHECKING, Set, Tuple, Any, Dict, List

from ..Log import CLIENT_LOGGER_NAME

if TYPE_CHECKING:
    from ...Shared.configs import GlobalConfig

IS_64BITS = sys_maxsize > 2**32
# This will compensate for 32/64 bit
struct_time_stamp = r"l"
TIMEVAL_SIZE: int = struct.calcsize(struct_time_stamp)
logger = logging.getLogger(CLIENT_LOGGER_NAME)
LP = "media::"
g: Optional[GlobalConfig] = None


class APIImagePipeLine:
    from ..Models.config import APIPullMethod

    def __init__(
        self,
        options: APIPullMethod,
    ):
        global g
        from ..main import get_global_config

        g = get_global_config()

        lp = f"{LP}API::init::"
        if not options:
            raise ValueError(f"{lp} no stream options provided!")
        #  INIT START 
        self.options = options
        self.event_tot_frames: int = 0
        self.has_event_ended: str = ""
        logger.debug(f"{lp} options: {self.options}")

        #  FRAME IDS 
        self._attempted_fids: List[int] = list()  # All tried
        self._processed_fids: List[int] = list()  # All successful
        self._skipped_fids: List[int] = list()  # All skipped

        #  FRAME BUFFER 
        self.total_min_frames: int = 1
        self._current_frame: int = 0
        self.current_snapshot: int = 0
        self.last_fid_read: int = 0
        self.last_snapshot_id: int = 0
        mon_fps = g.mon_fps or 1
        self.fps = int(Decimal(mon_fps).quantize(Decimal("1")))
        self.buffer_pre_count = g.mon_pre or 0
        self.buffer_post_count = g.mon_post or 1

        self.max_attempts = options.attempts
        self.max_attempts_delay = options.delay

        # Alarm frame is always the first frame, pre count buffer length+1 for alarm frame
        self.current_frame = self.buffer_pre_count + 1
        # The pre- / post-buffers will give the absolute minimum number of frames to grab, assuming no event issues
        self.total_min_frames = int(
            (self.buffer_post_count + self.buffer_pre_count) / self.fps
        )
        # We don't know how long an event will be so set an upper limit of at least
        # pre- + post-buffers calculated as seconds because we are pulling 1 FPS
        self.total_max_frames = min(self.options.max_frames, self.total_min_frames)

    async def get_image(self) -> Tuple[Optional[Union[bytes, bool]], Optional[str]]:
        if self.frames_processed >= self.total_max_frames:
            logger.error(
                f"max_frames ({self.total_max_frames}) has been reached, stopping!"
            )
            return False, None

        async def _grab_event_data(msg: Optional[str] = None):
            """Calls global API make_request method to get event data"""
            if msg:
                logger.debug(f"{lp}read>event_data: {msg}")
            if not self.event_ended:
                try:
                    g.Event, g.Monitor, g.Frame, _ = await g.api.get_all_event_data(g.eid)
                except Exception as e:
                    logger.error(f"{lp} error grabbing event data from API -> {e}")
                    raise e
                else:
                    self.event_tot_frames = int(g.Event.get("Frames", 0))
                    self.has_event_ended = g.Event.get("EndDateTime", "")
                    logger.debug(
                        f"{lp} grabbed event data from ZM API for event '{g.eid}' -- event total Frames: "
                        f"{self.event_tot_frames} -- EndDateTime: {self.has_event_ended} -- "
                        f"has event ended: {self.event_ended}"
                    )
                    # if self.event_ended:
                    #     logger.debug(
                    #         f"DBG => THIS EVENT HAS AN EndDateTime ({self.has_event_ended}) checking max_frames "
                    #         f"and modifying if needed"
                    #     )
                    #     new_max = int(self.event_tot_frames / self.fps)
                    #     logger.debug(
                    #         f"DEBUG>>>{LP} current max: {self.total_max_frames} new_max: {new_max} (event_total_frames"
                    #         f"[{self.event_tot_frames}] / fps[{self.fps}])"
                    #     )
                    #     if new_max > self.total_max_frames:
                    #         logger.debug(
                    #             f"{lp} max_frames ({self.total_max_frames}) is lower than current calculations, "
                    #             f"setting to {new_max}"
                    #         )
                    #         self.total_max_frames = new_max
            else:
                logger.debug(f"{lp} event has ended, no need to grab event data")
        import aiohttp
        response: Optional[aiohttp.ClientResponse] = None
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
            logger.debug(f"{g.past_event = } || {self.options.snapshot_frame_skip = }")
            # Check if event data available or get data for snapshot fid comparison
            if (
                (not g.past_event)
                and (self.frames_processed > 0)
                and (
                    self.frames_processed % self.options.snapshot_frame_skip == 0
                )  # Only run every <x> frames
            ):
                await _grab_event_data(msg=f"grabbing data for snapshot comparisons...")
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
            # if not g.Event:
            #     _grab_event_data(msg="NO EVENT DATA!!! grabbing from API...")

            #  Check if we have already processed this frame ID 
            if self.current_frame in self._processed_fids:
                logger.debug(
                    f"{lp} skipping Frame ID: '{self.current_frame}' as it has already been"
                    f" processed for event {g.eid}"
                )
                return self._process_frame(skip=True)
            #  SET URL TO GRAB IMAGE FROM 
            logger.debug(f"Calculated Frame ID as {self.current_frame}")
            fid_url = f"{g.api.portal_base_url}/index.php?view=image&eid={g.eid}&fid={self.current_frame}"

            if g.past_event:
                logger.debug(
                    f"{lp} this is a past event, max image grab attempts set to 1"
                )
                self.max_attempts = 1
            for image_grab_attempt in range(self.max_attempts):
                image_grab_attempt += 1
                logger.debug(
                    f"{lp} attempt #{image_grab_attempt}/{self.max_attempts} to grab image ID: {self.current_frame}"
                )
                api_response = await g.api.make_async_request(fid_url)
                if isinstance(api_response, bytes) and api_response.startswith(b'\xff\xd8\xff'):
                    logger.debug(f"ZM API returned a JPEG formatted image!")
                    return self._process_frame(image=api_response)
                else:
                    resp_msg = ""
                    if api_response:
                        resp_msg = f" response code={api_response.status} - response={api_response}"
                    else:
                        resp_msg = f" no response received!"

                    logger.warning(f"{lp} image was not retrieved!{resp_msg}")

                    await _grab_event_data(msg="checking if event has ended...")

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
                            f"{lp} sleeping for {self.options.delay} second(s)"
                        )
                        sleep(self.options.delay)

            return self._process_frame(skip=True)

    def is_image_stream_active(self) -> bool:
        _cont = self.frames_processed < self.total_max_frames
        if not _cont:
            logger.warning(f"Image stream exhausted. Tried: {self.frames_processed} - Max: {self.total_max_frames}")
        return _cont

    def _process_frame(
        self,
        image: bytes = None,
        skip: bool = False,
        end: bool = False,
    ) -> Tuple[Optional[Union[bytes, bool]], Optional[str]]:
        """Process the frame, increment counters, and return the image if there is one"""
        lp = f"{LP}API::processed_frame:"
        self.last_fid_read = self.current_frame
        self._attempted_fids.append(self.current_frame)
        if skip:
            self._skipped_fids.append(self.current_frame)
        else:
            self._processed_fids.append(self.current_frame)

        if end or self.frames_processed > self.total_max_frames:
            _msg = (
                "end has been called, no more images to process!"
                if end
                else f"max_frames ({self.total_max_frames}) has been reached, stopping!"
            )
            self._max_frames = 1
            logger.error(f"{lp} {_msg}")
            return False, None
        elif not end:
            self.current_frame = self.current_frame + self.fps
            logger.debug(
                f"{lp} incrementing next frame ID to read by {self.fps} = {self.current_frame}"
            )
        if image:
            # (bytes, image_file_name)
            return image, f"mid_{g.mid}-eid_{g.eid}-fid_{self.current_frame - self.fps}.jpg"
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


class ZMUImagePipeLine:
    offset = None


# Dynamically create timeval Struct to calculate size / 32 == 8 bytes // 64 == 16 bytes
def create_timeval():
    tv_usec = c_long
    # logger.debug("BEFORE CHECKING 64 BITNESS -> tv_usec: ", tv_usec)
    if IS_64BITS:
        # logger.debug("64 BITS is TRUE")
        tv_usec = c_uint64
    # logger.debug("AFTER CHECKING 64 BITNESS -> tv_usec: ", tv_usec)
    _fields = (
        (
            "tv_sec",
            c_long,
        ),
        (
            "tv_usec",
            tv_usec,
        ),
    )
    _class = type("timeval", (Structure,), {"_fields_": _fields})
    return _class()


# timeval = create_timeval()
# TIMEVAL_SIZE = sizeof(timeval)


def zm_version(
    ver: str, minx: Optional[int] = None, patchx: Optional[int] = None
) -> int:
    maj, min, patch = "", "", ""
    x = ver.split(".")
    x_len = len(x)
    if x_len <= 2:
        maj, min = x
        patch = "0"
    elif x_len == 3:
        maj, min, patch = x
    else:
        logger.debug("come and fix me!?!?!")
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


class ZMAlarmStateChanges(IntEnum):
    STATE_IDLE = 0
    STATE_PREALARM = 1
    STATE_ALARM = 2
    STATE_ALERT = 3
    STATE_TAPE = 4
    ACTION_GET = 5
    ACTION_SET = 6
    ACTION_RELOAD = 7
    ACTION_SUSPEND = 8
    ACTION_RESUME = 9
    TRIGGER_CANCEL = 10
    TRIGGER_ON = 11
    TRIGGER_OFF = 12


# Read images from a video/image file
class FileImagePipeLine:
    config: Optional[Dict[str, Any]] = None


class SHMImagePipeLine:
    def __init__(self):
        global g
        from ..main import get_global_config

        g = get_global_config()
        path = "/dev/shm"

        self.alarm_state_stages = ZMAlarmStateChanges
        self.file_handle: Optional[IO] = None
        self.mem_handle: Optional[mmap.mmap] = None

        self.file_name = f"{path}/zm.mmap.{g.mid}"

    def reload(self):
        """Reloads monitor information. Call after you get
        an invalid memory report

        Raises:
            ValueError: if no monitor is provided
        """
        # close file handler
        self.close()
        # open file handler in read binary mode
        self.file_handle = open(self.file_name, "r+b")
        # geta rough size of the memory consumed by object (doesn't follow links or weak ref)
        from os.path import getsize

        sz = getsize(self.file_name)
        if not sz:
            raise ValueError(f"Invalid size: {sz} of {self.file_name}")

        self.mem_handle = mmap.mmap(
            self.file_handle.fileno(), 0, access=mmap.ACCESS_READ
        )
        self.sd = None
        self.td = None
        self.get_image()

    def is_valid(self):
        """True if the memory handle is valid

        Returns:
            bool: True if memory handle is valid
        """
        try:
            d = self.get_image()
            return not d["shared_data"]["size"] == 0
        except Exception as e:
            logger.debug(f"ERROR!!!! =-> Memory: {e}")
            return False

    def get_image(self):
        self.mem_handle.seek(0)  # goto beginning of file
        # import proper class that contains mmap data
        from ..Models.shm_data import Dot3725

        x = Dot3725()
        sd_model = x.shared_data
        td_model = x.trigger_data
        vs_model = x.video_store_data

        SharedData = sd_model.named_tuple

        # old_shared_data = r"IIIIQIiiiiii????IIQQQ256s256s"
        # I = uint32 - i = int32 == 4 bytes ; int
        # Q = uint64 - q = int64 == 8 bytes ; long long int
        # L = uint64 - l = int64 == 8 bytes ; long int
        # d = double == 8 bytes ; float
        # s = char[] == n bytes ; string
        # B = uint8 - b = int8 == 1 byte; char

        # shared data bytes is now aligned at 776 as of commit 590697b (1.37.19) -> SEE
        # https://github.com/ZoneMinder/zoneminder/commit/590697bd807ab9a74d605122ef0be4a094db9605
        # Before it was 776 for 64bit and 772 for 32 bit

        TriggerData = td_model.named_tuple

        VideoStoreData = vs_model.named_tuple

        s = SharedData._make(
            struct.unpack(sd_model.struct_str, self.mem_handle.read(sd_model.bytes))
        )
        t = TriggerData._make(
            struct.unpack(td_model.struct_str, self.mem_handle.read(td_model.bytes))
        )
        v = VideoStoreData._make(
            struct.unpack(vs_model.struct_str, self.mem_handle.read(vs_model.bytes))
        )

        written_images = s.last_write_index + 1
        timestamp_offset = s.size + t.size + v.size
        images_offset = timestamp_offset + g.mon_image_buffer_count * TIMEVAL_SIZE
        # align to nearest 64 bytes
        images_offset = images_offset + 64 - (images_offset % 64)
        # images_offset = images_offset + s.imagesize * s.last_write_index
        # Read timestamp data
        ts_str = " "
        if s.last_write_index <= g.mon_image_buffer_count >= 0:
            for loop in range(written_images):
                ts_str = f"image{loop + 1}_ts"
        self.mem_handle.seek(timestamp_offset)
        TimeStampData = namedtuple(
            "TimeStampData",
            ts_str,
        )
        ts = TimeStampData._make(
            struct.unpack(
                struct_time_stamp,
                self.mem_handle.read(g.mon_image_buffer_count * TIMEVAL_SIZE),
            )
        )
        logger.debug(f"{written_images = } - {images_offset = }")
        logger.debug(f"\nSharedData = {s}\n")
        self.mem_handle.seek(images_offset)
        # only need 1 image, not the recent buffers worth
        image_buffer = self.mem_handle.read(int(s.imagesize))
        # image_buffer = self.mem_handle.read(written_images * s.imagesize)
        import cv2
        import numpy as np

        img: Optional[np.ndarray] = None
        logger.debug(f"Converting images into ndarray")
        # grab total image buffer
        # convert bytes to numpy array to cv2 images
        # for i in range(written_images):
        # img = np.frombuffer(image_buffer[i * s.imagesize : (i + 1) * s.imagesize], dtype=np.uint8)
        img = np.frombuffer(image_buffer, dtype=np.uint8)
        if img.size == s.imagesize:
            logger.debug(f"Image is of the correct size, reshaping and converting")
            img = img.reshape((g.mon_height, g.mon_width, g.mon_colorspace))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            logger.debug(f"Invalid image size: {img.size}")
        # show images
        self.sd = s._asdict()
        self.td = t._asdict()
        self.vsd = v._asdict()
        self.tsd = ts._asdict()

        self.sd["video_fifo_path"] = (
            self.sd["video_fifo_path"].decode("utf-8").strip("\x00")
        )
        self.sd["audio_fifo_path"] = (
            self.sd["audio_fifo_path"].decode("utf-8").strip("\x00")
        )
        self.sd["janus_pin"] = self.sd["janus_pin"].decode("utf-8").strip("\x00")
        self.vsd["event_file"] = self.vsd["event_file"].decode("utf-8").strip("\x00")
        self.td["trigger_text"] = self.td["trigger_text"].decode("utf-8").strip("\x00")
        self.td["trigger_showtext"] = (
            self.td["trigger_showtext"].decode("utf-8").strip("\x00")
        )
        self.td["trigger_cause"] = (
            self.td["trigger_cause"].decode("utf-8").strip("\x00")
        )
        self.sd["alarm_cause"] = self.sd["alarm_cause"].decode("utf-8").strip("\x00")
        self.sd["control_state"] = (
            self.sd["control_state"].decode("utf-8").strip("\x00")
        )

        return {
            "shared_data": self.sd,
            "trigger_data": self.td,
            "video_store_data": self.vsd,
            "time_stamp_data": self.tsd,
            "image": img,
        }

    def close(self):
        """Closes the handle"""
        try:
            if self.mem_handle:
                self.mem_handle.close()
            if self.file_handle:
                self.file_handle.close()
        except Exception:
            pass
