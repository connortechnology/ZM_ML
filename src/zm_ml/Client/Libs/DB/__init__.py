from __future__ import annotations

import glob
import logging
from configparser import ConfigParser
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional, Union, Tuple, TYPE_CHECKING

from sqlalchemy import MetaData, create_engine, select
from sqlalchemy.engine import Engine, Connection, CursorResult
from sqlalchemy.exc import SQLAlchemyError

from ...Log import CLIENT_LOGGER_NAME
from ...Models.config import ZMDBSettings
from ...main import get_global_config

if TYPE_CHECKING:
    from ....Shared.configs import GlobalConfig

logger = logging.getLogger(CLIENT_LOGGER_NAME)
LP = "zmdb::"
g: Optional[GlobalConfig] = None


class ZMDB:
    engine: Optional[Engine]
    connection: Optional[Connection]
    meta: Optional[MetaData]
    connection_str: str
    config: ZMDBSettings

    def __init__(self):
        global g
        g = get_global_config()
        self.engine = None
        self.connection = None
        self.meta = None
        self.config = self.init_config()
        self._db_create()

    def set_config(self, config: ZMDBSettings):
        self.config = config
        self.reset_db()
        self._db_create()

    @staticmethod
    def _rel_path(eid: int, mid: int, scheme: str, dt: Optional[datetime] = None) -> str:
        ret_val: str = ""
        lp: str = f"{LP}relative path::"
        if scheme == "Deep":
            if dt:
                ret_val = f"{mid}/{dt.strftime('%y/%m/%d/%H/%M/%S')}"
            else:
                logger.error(f"{lp} no datetime for deep scheme path!")
        elif scheme == "Medium":
            ret_val = f"{mid}/{dt.strftime('%Y-%m-%d')}/{eid}"
        elif scheme == "Shallow":
            ret_val = f"{mid}/{eid}"
        else:
            logger.error(f"{lp} unknown scheme {scheme}")
        return ret_val

    @staticmethod
    def read_zm_configs():
        files = []
        conf_path = g.Environment.zm_conf_dir
        if conf_path.is_dir():
            for fi in glob.glob(f"{conf_path}/conf.d/*.conf"):
                files.append(fi)
            files.sort()
            files.insert(0, f"{conf_path}/zm.conf")
            config_file = ConfigParser(interpolation=None, inline_comment_prefixes="#")
            try:
                for f in files:
                    with open(f, "r") as zm_conf_file:
                        # This adds [zm_root] section to the head of each zm .conf.d config file,
                        # not physically only in memory
                        _data = zm_conf_file.read()
                        config_file.read_string(f"[zm_root]\n{_data}")
            except Exception as exc:
                logger.error(f"{LP} error opening ZoneMinder .conf files! -> {files}")
            else:
                logger.debug(f"{LP} ZoneMinder .conf files -> {files}")
                # for section in config_file.sections():
                # for key, value in config_file.items(section):
                #     logger.debug(f"{section} >>> {key} = {value}")
                conf_data = config_file["zm_root"]
                return conf_data

    def init_config(self):
        defaults = {
            'host': 'localhost',
            'port': 3306,
            'user': 'zmuser',
            'password': 'zmpass',
            'name': 'zm',
            'driver': 'mysql+pymysql'
        }
        conf_data = self.read_zm_configs()
        c = g.Environment.db
        for _attr in dir(c):
            if _attr.startswith("_"):
                continue
            elif _attr in ['construct', 'copy', 'dict', 'from_orm', 'json', 'p', 'parse_file', 'parse_obj', 'parse_raw', 'schema', 'schema_json', 'update_forward_refs', 'validate']:
                continue
            elif _attr in ['host', 'port', 'user', 'password', 'name', 'driver']:
                if not getattr(c, _attr):
                    set_to = getattr(g.config.db, _attr)
                    xtra_ = ""
                    unset_ = ""
                    if not set_to:
                        set_to = conf_data[f"ZM_DB_{_attr.upper()}"]
                        xtra_ = f" (defaulting to '{set_to}' from ZM .conf files)"
                        unset_ += "ENV_VAR "

                    if not set_to:
                        unset_ += ".CONFs "
                        set_to = defaults[_attr]
                        xtra_ = f" (defaulting to '{set_to}' from internal defaults)"
                    logger.debug(f"{LP} [{unset_.rstrip()}] unset for db.{_attr}{xtra_}")
                    setattr(c, _attr, set_to)
        return c


    def _db_create(self):
        """A private function to interface with the ZoneMinder DataBase"""
        # From @pliablepixels SQLAlchemy work - all credit goes to them.
        lp: str = f"{LP}init::"
        self.connection_str = (
            f"{self.config.driver}://{self.config.user}"
            f":{self.config.password.get_secret_value()}@{self.config.host}"
            f"/{self.config.name}"
        )
        self._check_conn()

    def _check_conn(self):
        try:
            if not self.engine:
                # logger.debug(f"{LP} creating engine with {self.connection_str = } TYPE={type(self.connection_str)}")
                self.engine = create_engine(self.connection_str, pool_recycle=3600)
            if not self.connection:
                self.connection = self.engine.connect()
            if not self.meta:
                self._refresh_meta()
        except SQLAlchemyError as e:
            logger.error(f"{self.connection_str = } :: TYPE={type(self.connection_str)}")
            logger.error(f"Could not connect to DB, message was: {e}")
            raise e
        except Exception as e:
            logger.error(f"Exception while checking DB connection on _check_conn() -> {e}")
            raise e

    def _refresh_meta(self):
        self.meta = None
        self.meta = MetaData()
        self.meta.reflect(bind=self.engine, only=["Events", "Monitors", "Monitor_Status", "Storage"])

    def grab_all(self, eid: int) -> Tuple[int, str, int, int, Decimal, str, str]:
        #         return mid, mon_name, mon_post, mon_pre, mon_fps, reason, event_path
        self._check_conn()
        event_exists: bool = False
        mid: Optional[Union[str, int]] = None
        mon_name: Optional[str] = None
        mon_post: Optional[Union[str, int]] = None
        mon_pre: Optional[Union[str, int]] = None
        mon_fps: Optional[Union[float, Decimal]] = None
        reason: Optional[str] = None
        notes: Optional[str] = None
        scheme: Optional[str] = None
        storage_id: Optional[int] = None
        start_datetime: Optional[datetime] = None
        storage_path: Optional[str] = None
        event_path: Optional[Union[Path, str]] = None
        _evt_select: select = select(self.meta.tables["Events"]).where(
            self.meta.tables["Events"].c.Id == eid
        )
        event_exists: bool = self.connection.execute(_evt_select).fetchone() is not None
        logger.debug(f"{LP} event_exists = {event_exists}")
        if not event_exists:
            raise ValueError(f"Event ID {eid} does not exist in ZoneMinder DB")

        e_select: select = select(self.meta.tables["Events"].c.MonitorId).where(
            self.meta.tables["Events"].c.Id == eid
        )
        mid_result: CursorResult = self.connection.execute(e_select)

        for row in mid_result:
            mid = row[0]
        mid_result.close()
        if mid:
            mid = int(mid)
            logger.debug(f"{LP} ZoneMinder DB returned Monitor ID: {mid}")
            if g.mid and g.mid != mid:
                logger.debug(f"{LP} CLI supplied monitor ID ({g.mid}) INCORRECT! Changed to: {mid}")
            g.mid = mid
            # add extra logging data
        else:
            logger.warning(
                f"{LP} the database query did not return a monitor ID for this event?"
            )
            raise ValueError("No Monitor ID returned from DB query")

        mid_name_select: select = select(self.meta.tables["Monitors"].c.Name).where(
            self.meta.tables["Monitors"].c.Id == mid
        )
        pre_event_select: select = select(
            self.meta.tables["Monitors"].c.PreEventCount
        ).where(self.meta.tables["Monitors"].c.Id == mid)

        # Get Monitor 'Name'

        mid_name_result: CursorResult = self.connection.execute(mid_name_select)
        for mon_row in mid_name_result:
            mon_name = mon_row[0]
        mid_name_result.close()
        if mon_name:
            logger.debug(f"{LP} ZoneMinder DB returned monitor name ('{mon_name}')")
        else:
            logger.warning(
                f"{LP} the database query did not return a monitor name ('Name') for monitor ID {mid}"
            )

        # Get Monitor Pre/Post Event Count

        pre_event_result: CursorResult = self.connection.execute(pre_event_select)
        for mon_row in pre_event_result:
            mon_pre = mon_row[0]
        pre_event_result.close()
        if mon_pre:
            mon_pre = int(mon_pre)
            logger.debug(
                f"{LP} ZoneMinder DB returned monitor PreEventCount ('{mon_pre}')"
            )
        else:
            logger.warning(
                f"{LP} the database query did not return monitor pre-event count ('PreEventCount') for monitor ID {mid}"
            )
        # PostEventCount
        post_event_select: select = select(
            self.meta.tables["Monitors"].c.PostEventCount
        ).where(self.meta.tables["Monitors"].c.Id == mid)
        select_result: CursorResult = self.connection.execute(post_event_select)

        for mon_row in select_result:
            mon_post = mon_row[0]
        select_result.close()
        if mon_post:
            mon_post = int(mon_post)
            logger.debug(
                f"{LP} ZoneMinder DB returned monitor PostEventCount ('{mon_post}')"
            )
        else:
            logger.warning(
                f"{LP} the database query did not return monitor post-event count ('PostEventCount') for monitor ID {mid}"
            )
        # Get Monitor capturing FPS
        ms_select: select = select(
            self.meta.tables["Monitor_Status"].c.CaptureFPS
        ).where(self.meta.tables["Monitor_Status"].c.MonitorId == mid)
        select_result: CursorResult = self.connection.execute(ms_select)
        for mons_row in select_result:
            mon_fps = float(mons_row[0])
        select_result.close()
        if mon_fps:
            mon_fps = Decimal(mon_fps)
            logger.debug(f"{LP} ZoneMinder DB returned monitor FPS ('{mon_fps}')")
        else:
            logger.warning(
                f"{LP} the database query did not return monitor FPS ('CaptureFPS') for monitor ID {mid}"
            )

        reason_select: select = select(self.meta.tables["Events"].c.Cause).where(
            self.meta.tables["Events"].c.Id == eid
        )
        notes_select: select = select(self.meta.tables["Events"].c.Notes).where(
            self.meta.tables["Events"].c.Id == eid
        )
        scheme_select: select = select(self.meta.tables["Events"].c.Scheme).where(
            self.meta.tables["Events"].c.Id == eid
        )
        storage_id_select: select = select(
            self.meta.tables["Events"].c.StorageId
        ).where(self.meta.tables["Events"].c.Id == eid)
        start_datetime_select: select = select(
            self.meta.tables["Events"].c.StartDateTime
        ).where(self.meta.tables["Events"].c.Id == eid)
        reason_result: CursorResult = self.connection.execute(reason_select)
        notes_result: CursorResult = self.connection.execute(notes_select)
        scheme_result: CursorResult = self.connection.execute(scheme_select)
        storage_id_result: CursorResult = self.connection.execute(storage_id_select)
        start_datetime_result: CursorResult = self.connection.execute(
            start_datetime_select
        )
        for row in notes_result:
            g.notes = row[0]
        notes_result.close()
        for row in reason_result:
            reason = row[0]
        reason_result.close()
        for row in scheme_result:
            scheme = row[0]
        scheme_result.close()
        for row in storage_id_result:
            storage_id = row[0]
        storage_id_result.close()
        for row in start_datetime_result:
            start_datetime = row[0]
        start_datetime_result.close()

        if storage_id is not None:
            storage_id = 1 if storage_id == 0 else storage_id # Catch 0 and treat as 1 (zm code issue)
            storage_path_select: select = select(
                self.meta.tables["Storage"].c.Path
            ).where(self.meta.tables["Storage"].c.Id == storage_id)
            storage_path_result: CursorResult = self.connection.execute(
                storage_path_select
            )
            for row in storage_path_result:
                storage_path = row[0]
            storage_path_result.close()
        else:
            logger.debug(f"{LP} no storage ID for event {eid}")

        if start_datetime:
            if storage_path:
                event_path = Path(
                    f"{storage_path}/{self._rel_path(eid, mid, scheme, start_datetime)}"
                )
            else:
                if storage_id:
                    logger.error(
                        f"{LP} no storage path for StorageId {storage_id}, the StorageId could "
                        f"of been removed/deleted/disabled"
                    )
                else:
                    logger.error(f"{LP} no StorageId for event {eid}!")
        else:
            logger.debug(f"{LP} no StartDateTime for event {eid}")

        if event_path:
            logger.debug(
                f"{LP} storage path for event ID: {eid} has been calculated as '{event_path}'"
            )
        else:
            logger.warning(
                f"{LP} the database could not calculate the storage path for this event!"
            )

        if reason:
            logger.debug(f"{LP} ZoneMinder DB returned event cause as '{reason}'")
        else:
            logger.warning(
                f"{LP} the database query did not return a 'reason' ('Cause') for this event!"
            )

            # Get Monitor 'ImageBufferCount'
        buffer_select: select = select(self.meta.tables["Monitors"].c.ImageBufferCount).where(
            self.meta.tables["Monitors"].c.Id == g.mid)
        width_select: select = select(self.meta.tables["Monitors"].c.Width).where(
            self.meta.tables["Monitors"].c.Id == g.mid)
        height_select: select = select(self.meta.tables["Monitors"].c.Height).where(
            self.meta.tables["Monitors"].c.Id == g.mid)
        colours_select: select = select(self.meta.tables["Monitors"].c.Colours).where(
            self.meta.tables["Monitors"].c.Id == g.mid)
        buffer_result: CursorResult = self.connection.execute(buffer_select)
        for mon_row in buffer_result:
            g.mon_image_buffer_count = mon_row[0]
        buffer_result.close()

        width_result: CursorResult = self.connection.execute(width_select)
        for mon_row in width_result:
            g.mon_width = mon_row[0]
        width_result.close()
        # height
        height_result: CursorResult = self.connection.execute(height_select)
        for mon_row in height_result:
            g.mon_height = mon_row[0]
        height_result.close()
        # colours
        colours_result: CursorResult = self.connection.execute(colours_select)
        for mon_row in colours_result:
            g.mon_colorspace = mon_row[0]
        colours_result.close()

        if g.mon_image_buffer_count:
            logger.debug(f"Got ImageBufferCount for monitor {g.mid} -=> {g.mon_image_buffer_count = }")
        else:
            logger.debug(
                f"{LP} the database query did not return ImageBufferCount for monitor {g.mid}"
            )
        if g.mon_width:
            logger.debug(f"Got Width for monitor {g.mid} -=> {g.mon_width}")
        else:
            logger.debug(
                f"{LP} the database query did not return Width for monitor {g.mid}"
            )
        if g.mon_height:
            logger.debug(f"Got Height for monitor {g.mid} -=> {g.mon_height}")
        else:
            logger.debug(
                f"{LP} the database query did not return Height for monitor {g.mid}"
            )
        if g.mon_colorspace:
            logger.debug(f"Got Colours for monitor {g.mid} -=> {g.mon_colorspace}")
        else:
            logger.debug(
                f"{LP} the database query did not return Colours for monitor {g.mid}"
            )

        g.mon_name = mon_name
        g.mon_post = mon_post
        g.mon_pre = mon_pre
        g.mon_fps = mon_fps
        g.event_cause = reason
        g.event_path = event_path
        return mid, mon_name, mon_post, mon_pre, mon_fps, reason, event_path

    def reset_db(self):
        if self.engine:
            self.engine.dispose()
            self.engine = None
        if self.connection:
            self.connection.close()
            self.connection = None
        if self.meta:
            self.meta.clear()
            self.meta = None


    def clean_up(self):
        if self.connection.closed is False:
            self.connection.close()
            logger.debug(f"{LP}exit:: Closed connection to ZoneMinder database")
        else:
            logger.debug(f"{LP}exit:: ZoneMinder database connection already closed")
