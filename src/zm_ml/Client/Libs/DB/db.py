from __future__ import annotations

import glob
import logging
import time
from configparser import ConfigParser, SectionProxy
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional, Union, Tuple, TYPE_CHECKING, Any, Dict
import warnings

import pydantic.version
from pydantic import SecretStr

try:
    from sqlalchemy import MetaData, create_engine, select
    from sqlalchemy.engine import Engine, Connection, CursorResult, ResultProxy
    from sqlalchemy.exc import SQLAlchemyError
except ImportError:
    warnings.warn("SQLAlchemy not installed, ZMDB will not be available", ImportWarning)
    MetaData: Optional[MetaData] = None
    create_engine: Optional[create_engine] = None
    select: Optional[select] = None
    Engine: Optional[Engine] = None
    Connection: Optional[Connection] = None
    CursorResult: Optional[CursorResult] = None
    SQLAlchemyError: Optional[SQLAlchemyError] = None
    ResultProxy: Optional[ResultProxy] = None

from ...Log import CLIENT_LOGGER_NAME
from ...Models.config import ZMDBSettings, ClientEnvVars

if TYPE_CHECKING:
    from ....Shared.configs import GlobalConfig

    # from sqlalchemy import MetaData, create_engine, select
    # from sqlalchemy.engine import Engine, Connection, CursorResult, ResultProxy
    # from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(CLIENT_LOGGER_NAME)
LP = "zmdb::"
g: Optional[GlobalConfig] = None


class ZMDB:
    engine: Optional[Engine]
    connection: Optional[Connection]
    meta: Optional[MetaData]
    connection_str: str
    config: ZMDBSettings
    conf_file_data: Optional[SectionProxy]
    env: Optional[ClientEnvVars]
    cgi_path: str

    def __init__(self, env: Optional[ClientEnvVars] = None):
        global g
        from ...main import get_global_config

        g = get_global_config()

        # TODO: integrate better
        if env:
            self.env = env
        else:
            self.env = g.Environment

        logger.debug(f"{LP} ENV VARS = {self.env}")

        self.engine = None
        self.connection = None
        self.meta = None
        g.db = self
        self.config = self.init_config()
        self._db_create()

    def set_config(self, config: ZMDBSettings):
        self.config = config
        self.reset_db()
        self._db_create()

    @staticmethod
    def _rel_path(
        eid: int, mid: int, scheme: str, dt: Optional[datetime] = None
    ) -> str:
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

    def read_zm_configs(self):
        files = []
        conf_path = self.env.zm_conf_dir
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
                logger.error(f"{LP} error opening ZoneMinder .conf files: {exc}")
            else:
                logger.debug(f"{LP} ZoneMinder .conf files -> {files}")
                # for section in config_file.sections():
                #     for key, value in config_file.items(section):
                #         logger.debug(f"{section} >>> {key} = {value}")
                return config_file["zm_root"]

    def init_config(self):
        """Initialize ZMDBSettings using ENV, zm .conf files and finally internal defaults"""
        defaults = {
            "host": "localhost",
            "port": 3306,
            "user": "zmuser",
            "password": "zmpass",
            "name": "zm",
            "driver": "mysql+pymysql",
        }
        # todo: get db type from zm .conf files for driver string?

        self.conf_file_data = self.read_zm_configs()
        self.cgi_path = self.conf_file_data.get("zm_path_cgi", "COULDNT GET: ZM_PATH_CGI")
        # ZM_PATH_CGI=/usr/lib/zoneminder/cgi-bin
        _pydantic_attrs = [
            "construct",
            "copy",
            "dict",
            "from_orm",
            "json",
            "p",
            "parse_file",
            "parse_obj",
            "parse_raw",
            "schema",
            "schema_json",
            "update_forward_refs",
            "validate",
        ]
        _pydantic_v2_attrs = [
            "model_computed_fields",
            "model_config",
            "model_construct",
            "model_copy",
            "model_dump",
            "model_dump_json",
            "model_extra",
            "model_fields",
            "model_fields_set",
            "model_json_schema",
            "model_parametrized_name",
            "model_post_init",
            "model_rebuild",
            "model_validate",
            "model_validate_json",
            "settings_customise_sources",
        ]
        _pydantic_attrs.extend(_pydantic_v2_attrs)
        db_config_with_env = self.env.db
        logger.debug(f"{LP} ENV VARS = {db_config_with_env}")
        for _attr in dir(db_config_with_env):
            if _attr.startswith("_"):
                continue
            elif _attr in _pydantic_attrs:
                continue
            elif _attr in ["host", "port", "user", "password", "name", "driver"]:
                if not (set_to := getattr(db_config_with_env, _attr)):
                    # env var is not set, try to get them from ZM .conf files
                    xtra_ = ""
                    unset_ = ""
                    conf_val = f"ZM_DB_{_attr.upper()}"
                    if _attr == "password":
                        conf_val = f"ZM_DB_PASS"
                    if not set_to:
                        unset_ += "ENV "
                        if conf_val in self.conf_file_data:
                            set_to = (
                                self.conf_file_data[conf_val]
                                if _attr != "password"
                                else SecretStr(self.conf_file_data[conf_val])
                            )
                            xtra_ = f" (defaulting to '{set_to}' from ZM .conf files)"

                if g and g.config:
                    if g.config.zoneminder.db:
                        cfg_file_db = getattr(g.config.zoneminder.db, _attr)
                        if cfg_file_db:
                            # todo: what if we want it to be an empty string? or int(0)
                            # There is an entry in the config file, use it even if ENV or .conf files set it
                            set_to = cfg_file_db
                            xtra_ = f" (OVERRIDING to '{set_to}' from config file)"

                if not set_to:
                    # not in env or .conf files try internal defaults
                    unset_ += "CFG .CONFs "
                    set_to = defaults[_attr]
                    xtra_ = f" (defaulting to '{set_to}' from internal defaults)"
                logger.debug(f"{LP} [{unset_.rstrip()}] unset for db. {_attr}{xtra_}")
                setattr(db_config_with_env, _attr, set_to)
        return db_config_with_env

    def _db_create(self):
        """A private function to interface with the ZoneMinder DataBase"""
        # From @pliablepixels SQLAlchemy work - all credit goes to them.
        lp: str = f"{LP}init::"
        _pw = (
            self.config.password.get_secret_value()
            if isinstance(self.config.password, SecretStr)
            else self.config.password
        )
        self.connection_str = (
            f"{self.config.driver}://{self.config.user}"
            f":{_pw}@{self.config.host}"
            f"/{self.config.name}"
        )
        self._check_conn()

    def _check_conn(self):
        """A private function to create the DB engine, connection and metadata if not already created"""
        try:
            if not self.engine:
                # logger.debug(f"{LP} creating engine with {self.connection_str = } TYPE={type(self.connection_str)}")
                self.engine = create_engine(self.connection_str, pool_recycle=3600)
            if not self.connection:
                self.connection = self.engine.connect()
            if not self.meta:
                self._refresh_meta()
        except SQLAlchemyError as e:
            logger.error(
                f"{self.connection_str = } :: TYPE={type(self.connection_str)}"
            )
            logger.error(f"Could not connect to DB, message was: {e}")
            raise e
        except Exception as e:
            logger.error(
                f"Exception while checking DB connection on _check_conn() -> {e}"
            )
            raise e

    def _refresh_meta(self):
        """A private function to refresh the DB metadata"""
        del self.meta
        self.meta = None
        self.meta = MetaData()
        self.meta.reflect(
            bind=self.engine, only=["Events", "Monitors", "Monitor_Status", "Storage"]
        )

    def run_select(self, select_stmt: select) -> ResultProxy:
        """A function to run a select statement"""
        self._check_conn()
        try:
            result = self.connection.execute(select_stmt)
        except SQLAlchemyError as e:
            logger.error(f"Could not read from DB, message was: {e}")
        else:
            return result

    def _mid_from_eid(self, eid: int) -> int:
        """A function to get the Monitor ID from the Event ID"""
        mid: int = 0
        e_select: select = select(self.meta.tables["Events"].c.MonitorId).where(
            self.meta.tables["Events"].c.Id == eid
        )
        mid_result: CursorResult = self.run_select(e_select)
        for row in mid_result:
            mid = row[0]
        mid_result.close()
        return int(mid)

    def _mon_name_from_mid(self, mid: int) -> str:
        # Get Monitor 'Name'
        mon_name = None
        mid_name_select: select = select(self.meta.tables["Monitors"].c.Name).where(
            self.meta.tables["Monitors"].c.Id == mid
        )
        mid_name_result: CursorResult = self.run_select(mid_name_select)
        for mon_row in mid_name_result:
            mon_name = mon_row[0]
        mid_name_result.close()
        return mon_name

    def _mon_preBuffer_from_mid(self, mid: int) -> int:
        mon_pre: int = 0
        pre_event_select: select = select(
            self.meta.tables["Monitors"].c.PreEventCount
        ).where(self.meta.tables["Monitors"].c.Id == mid)
        result: CursorResult = self.connection.execute(pre_event_select)
        for mon_row in result:
            mon_pre = mon_row[0]
        result.close()
        return int(mon_pre)

    def _mon_postBuffer_from_mid(self, mid: int) -> int:
        mon_post: int = 0
        post_event_select: select = select(
            self.meta.tables["Monitors"].c.PostEventCount
        ).where(self.meta.tables["Monitors"].c.Id == mid)
        select_result: CursorResult = self.connection.execute(post_event_select)

        for mon_row in select_result:
            mon_post = mon_row[0]
        select_result.close()
        return int(mon_post)

    def _mon_fps_from_mid(self, mid: int) -> Decimal:
        mon_fps: Decimal = Decimal(0)
        # Get Monitor capturing FPS
        ms_select: select = select(
            self.meta.tables["Monitor_Status"].c.CaptureFPS
        ).where(self.meta.tables["Monitor_Status"].c.MonitorId == mid)
        select_result: CursorResult = self.connection.execute(ms_select)
        for mons_row in select_result:
            mon_fps = float(mons_row[0])
        select_result.close()
        return Decimal(mon_fps)

    def _reason_from_eid(self, eid: int) -> str:
        reason: str = ""
        reason_select: select = select(self.meta.tables["Events"].c.Cause).where(
            self.meta.tables["Events"].c.Id == eid
        )
        reason_result: CursorResult = self.connection.execute(reason_select)
        for row in reason_result:
            reason = row[0]
        reason_result.close()
        return reason

    def _notes_from_eid(self, eid: int) -> str:
        notes: str = ""
        notes_select: select = select(self.meta.tables["Events"].c.Notes).where(
            self.meta.tables["Events"].c.Id == eid
        )
        notes_result: CursorResult = self.connection.execute(notes_select)
        for row in notes_result:
            notes = row[0]
        notes_result.close()
        return notes

    def _scheme_from_eid(self, eid: int):
        scheme = None
        scheme_select: select = select(self.meta.tables["Events"].c.Scheme).where(
            self.meta.tables["Events"].c.Id == eid
        )
        scheme_result: CursorResult = self.connection.execute(scheme_select)
        for row in scheme_result:
            scheme = row[0]
        scheme_result.close()
        return scheme

    def _storage_id_from_eid(self, eid: int) -> int:
        storage_id: Optional[int] = None
        storage_id_select: select = select(
            self.meta.tables["Events"].c.StorageId
        ).where(self.meta.tables["Events"].c.Id == eid)
        storage_id_result: CursorResult = self.run_select(storage_id_select)
        for row in storage_id_result:
            storage_id = row[0]
        storage_id_result.close()
        if storage_id is not None:
            storage_id = (
                1 if storage_id == 0 else storage_id
            )  # Catch 0 and treat as 1 (zm code issue)
        return storage_id

    def _start_datetime_from_eid(self, eid: int) -> datetime:
        start_datetime: Optional[datetime] = None
        start_datetime_select: select = select(
            self.meta.tables["Events"].c.StartDateTime
        ).where(self.meta.tables["Events"].c.Id == eid)
        start_datetime_result: CursorResult = self.connection.execute(
            start_datetime_select
        )
        for row in start_datetime_result:
            start_datetime = row[0]
        start_datetime_result.close()
        return start_datetime

    def _storage_path_from_storage_id(self, storage_id: int) -> str:
        storage_path: str = ""
        storage_path_select: select = select(self.meta.tables["Storage"].c.Path).where(
            self.meta.tables["Storage"].c.Id == storage_id
        )
        storage_path_result: CursorResult = self.connection.execute(storage_path_select)
        for row in storage_path_result:
            storage_path = row[0]
        storage_path_result.close()
        return storage_path

    def _get_mon_shape_from_mid(self, mid: int) -> Tuple[int, int, int]:
        """Get the monitor shape from the DB. (W, H, C)"""
        width: int = 0
        height: int = 0
        color: int = 0
        width_select: select = select(self.meta.tables["Monitors"].c.Width).where(
            self.meta.tables["Monitors"].c.Id == mid
        )
        height_select: select = select(self.meta.tables["Monitors"].c.Height).where(
            self.meta.tables["Monitors"].c.Id == mid
        )
        colours_select: select = select(self.meta.tables["Monitors"].c.Colours).where(
            self.meta.tables["Monitors"].c.Id == mid
        )

        width_result: CursorResult = self.run_select(width_select)
        for mon_row in width_result:
            width = mon_row[0]
        width_result.close()
        # height
        height_result: CursorResult = self.run_select(height_select)
        for mon_row in height_result:
            height = mon_row[0]
            g.mon_height = height
        height_result.close()
        # colours
        colours_result: CursorResult = self.run_select(colours_select)
        for mon_row in colours_result:
            color = mon_row[0]
        colours_result.close()
        return width, height, color

    def _get_image_buffer_from_mid(self, mid: int) -> int:
        """Get the monitor ImageBufferCount from the DB.

        Key in DB: 'ImageBufferCount'"""
        buffer: Optional[int] = None
        buffer_select: select = select(
            self.meta.tables["Monitors"].c.ImageBufferCount
        ).where(self.meta.tables["Monitors"].c.Id == mid)
        buffer_result: CursorResult = self.connection.execute(buffer_select)
        for mon_row in buffer_result:
            buffer = mon_row[0]
        buffer_result.close()
        return buffer

    def cause_from_eid(self, eid: int) -> str:
        """Get the cause of the event from the DB.

        Key in DB: 'Cause'"""
        return self._reason_from_eid(eid)

    def eid_exists(self, eid: int) -> bool:
        """Check if an event ID exists in the DB

        Key in DB: 'Id'"""
        event_exists: bool = False
        event_exists = (
            self.run_select(
                select(self.meta.tables["Events"]).where(
                    self.meta.tables["Events"].c.Id == eid
                )
            ).fetchone()
            is not None
        )
        return event_exists

    def grab_all(self, eid: int) -> Tuple[int, str, int, int, Decimal, str, str]:
        """FIX ME!!!! A hammer to grab all the data from the DB for a given event ID"""
        _start = time.perf_counter()
        event_exists: bool = self.eid_exists(eid)
        if not event_exists:
            raise ValueError(f"Event ID {eid} does not exist in ZoneMinder DB")

        storage_path: Optional[str] = None
        event_path: Optional[Union[Path, str]] = None
        mid: Optional[Union[str, int]] = self._mid_from_eid(eid)
        mon_name: Optional[str] = self._mon_name_from_mid(mid)
        mon_post: Optional[Union[str, int]] = self._mon_postBuffer_from_mid(mid)
        mon_pre: Optional[Union[str, int]] = self._mon_preBuffer_from_mid(mid)
        mon_fps: Optional[Union[float, Decimal]] = self._mon_fps_from_mid(mid)
        reason: Optional[str] = self._reason_from_eid(eid)
        notes: Optional[str] = self._notes_from_eid(eid)
        scheme: Optional[str] = self._scheme_from_eid(eid)
        storage_id: Optional[int] = self._storage_id_from_eid(eid)
        start_datetime: Optional[datetime] = self._start_datetime_from_eid(eid)
        height, width, color = self._get_mon_shape_from_mid(mid)
        ring_buffer: Optional[int] = self._get_image_buffer_from_mid(mid)
        if storage_id:
            storage_path = self._storage_path_from_storage_id(storage_id)

        final_str: str = ""
        if mid:
            final_str += f"Monitor ID: {mid} "
        else:
            raise ValueError("No Monitor ID returned from DB query")
        if mon_name:
            final_str += f"Monitor Name: {mon_name} "
        else:
            logger.warning(
                f"{LP} the database query did not return a monitor name ('Name') for monitor ID {mid}"
            )
        if mon_pre:
            final_str += f"Monitor PreEventCount: {mon_pre} "
        else:
            logger.warning(
                f"{LP} the database query did not return monitor pre-event count ('PreEventCount') for monitor ID {mid}"
            )
        if mon_post:
            final_str += f"Monitor PostEventCount: {mon_post} "
        else:
            logger.warning(
                f"{LP} the database query did not return monitor post-event count ('PostEventCount') for monitor ID {mid}"
            )
        if mon_fps:
            final_str += f"Monitor FPS: {mon_fps} "
        else:
            logger.warning(
                f"{LP} the database query did not return monitor FPS ('CaptureFPS') for monitor ID {mid}"
            )
        if reason:
            final_str += f"Event Cause: {reason} "
        else:
            logger.warning(
                f"{LP} the database query did not return a 'reason' ('Cause') for this event!"
            )
        if notes:
            final_str += f"Event Notes: {notes} "
        else:
            logger.warning(
                f"{LP} the database query did not return any notes ('Notes') for this event!"
            )
        if scheme:
            final_str += f"Event Storage Scheme: {scheme} "
        else:
            logger.warning(
                f"{LP} the database query did not return any scheme ('Scheme') for this event!"
            )
        if storage_id:
            final_str += f"Event Storage ID: {storage_id} "
        else:
            logger.warning(
                f"{LP} the database query did not return any storage ID ('StorageId') for this event!"
            )
        if start_datetime:
            final_str += f"Event StartDateTime: {start_datetime} "
        else:
            logger.warning(
                f"{LP} the database query did not return any start datetime ('StartDateTime') for this event!"
            )
        if storage_path:
            final_str += f"Event Storage Path: {storage_path} "
        else:
            logger.warning(
                f"{LP} the database query did not return a storage path ('Path') for this event!"
            )
        if width:
            final_str += f"Monitor Width: {width} "
        else:
            logger.warning(
                f"{LP} the database query did not return a monitor width ('Width') for this event!"
            )
        if height:
            final_str += f"Monitor Height: {height} "
        else:
            logger.warning(
                f"{LP} the database query did not return a monitor height ('Height') for this event!"
            )
        if color:
            final_str += f"Monitor Color: {color} "
        else:
            logger.warning(
                f"{LP} the database query did not return a monitor color ('Colours') for this event!"
            )
        if ring_buffer:
            final_str += f"Monitor ImageBufferCount: {ring_buffer}"
        else:
            logger.warning(
                f"{LP} the database query did not return a monitor ImageBufferCount ('ImageBufferCount') for this event!"
            )
        if storage_path:
            if eid and mid and scheme and start_datetime:
                event_path = Path(
                    f"{storage_path}/{self._rel_path(eid, mid, scheme, start_datetime)}"
                )
            else:
                logger.error(
                    f"{LP} no event ID ({eid}), monitor ID ({mid}), scheme ({scheme}) or start_datetime ({start_datetime}) to calculate the storage path!"
                )
        else:
            if storage_id:
                logger.error(
                    f"{LP} no storage path for StorageId {storage_id}, the StorageId could "
                    f"of been removed/deleted/disabled"
                )
            else:
                logger.error(
                    f"{LP} no StorageId for event {eid} to calculate the storage path!"
                )

        if event_path:
            logger.debug(
                f"{LP} storage path for event ID: {eid} has been calculated as '{event_path}'"
            )
        else:
            logger.warning(f"{LP} could not calculate the storage path for this event!")

        logger.debug(f"perf:{LP} Grabbing DB info took {time.perf_counter() - _start:.5f} s ----> {final_str.rstrip()}")
        return (
            mid,
            mon_name,
            mon_post,
            mon_pre,
            mon_fps,
            reason,
            event_path,
            notes,
            width,
            height,
            color,
            ring_buffer,
        )

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
            logger.debug(f"{LP}close:: Closed connection to ZoneMinder database")
        else:
            logger.debug(f"{LP}close:: ZoneMinder database connection already closed")
