#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
__doc__ = """
An example that uses ZM's EventStartCommand mechanism to run 
object detection on a ZM event using ZM-ML library.
"""



import logging.handlers
import sys
import time
from pathlib import Path
from typing import Optional

import uvloop
import asyncio

from zm_ml import Client
from zm_ml.Client import str_to_path, CLIENT_LOG_FORMAT
from zm_ml.Shared.configs import ClientEnvVars, GlobalConfig
from zm_ml.Client.main import parse_client_config_file, create_global_config, init_logs
from src.zm_ml.Shared.Log.handlers import BufferedLogHandler

__version__ = "0.0.0-a1"
__version_type__ = "dev"
# Setup basic console logging (hook into library logging)
logger = logging.getLogger("ZM_ML-Client")
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setFormatter(CLIENT_LOG_FORMAT)
buffered_log_handler = BufferedLogHandler()
buffered_log_handler.setFormatter(CLIENT_LOG_FORMAT)
logger.setLevel(logging.DEBUG)
logger.addHandler(console_handler)
logger.addHandler(buffered_log_handler)

def _parse_cli():
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="eventstart.py", description=__doc__)
    parser.add_argument(
        "--shm",
        "-S",
        action="store_true",
        dest="shm",
        help="Run in shared memory mode (Pulling images directly from SHM)",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument("--config", "-C", help="Config file to use", type=Path)
    parser.add_argument(
        "--event",
        "-E",
        action="store_true",
        dest="event",
        help="Run in event mode (triggered by a ZM motion alarm/event)",
    )
    parser.add_argument(
        "--eid",
        "-e",
        help="Event ID to process (Required for --event/-E)",
        type=int,
        dest="eid",
        default=0,
    )
    parser.add_argument(
        "--monitor",
        "--mid",
        "-m",
        help="Monitor ID to process (Required for --shm)",
        type=int,
        dest="mid",
        default=0,
    )
    parser.add_argument(
        "--debug", "-D", help="Enable debug logging", action="store_true", dest="debug"
    )
    parser.add_argument(
        "--live",
        "-L",
        help="This is a live event",
        action="store_true",
        dest="live",
    )

    args = parser.parse_args()
    # logger.debug(f"CLI Args: {args}")
    return args


async def main():
    global g
    lp = f"eventstart::"
    _mode = ""
    _start = time.perf_counter()
    # Do all the config stuff and setup logging
    args = _parse_cli()
    eid = args.eid
    mid = args.mid
    cfg_file: Optional[Path] = None
    if args.event:
        _mode = "event"
        logger.debug(f"{lp} Running in event mode")
        if args.eid == 0:
            logger.error(f"{lp} Event ID is required for event mode")
            sys.exit(1)
        if args.mid == 0:
            logger.warning(
                f"{lp} When monitor ID is not supplied in event mode, ZoneMinder DB is queried for it"
            )

    elif args.shm:
        _mode = "shm"
        logger.debug(f"{lp} Running in shared memory mode (Monitor ID REQUIRED)")
        if args.mid == 0:
            logger.error(f"{lp} Monitor ID is required for shared memory mode")
            sys.exit(1)
    if "config" in args and args.config:
        # logger.info(f"Configuration file supplied as: {args.config}")
        cfg_file = args.config
    else:
        logger.warning(
            f"No config file supplied, checking ENV: {g.Environment.conf_file}"
        )
        if g.Environment.conf_file:
            cfg_file = g.Environment.conf_file
    if cfg_file:
        cfg_file = str_to_path(cfg_file)
    assert cfg_file, "No config file supplied via CLI or ENV"
    g.config_file = cfg_file
    logger.info(
        f"{lp} Event: {args.event} [Event ID: {eid}] || SHM: {args.shm} ||"
        f"{' || Monitor ID: ' if mid else ''}{mid if mid else ''} || "
        f"Config File: {cfg_file}"
    )
    g.config = parse_client_config_file(cfg_file)
    logger.debug(f"{lp} INITIALIZING ZMCLIENT")
    zm_client = Client.ZMClient(global_config=g)
    _end_init = time.perf_counter()
    __event_modes = ["event", ""]
    if _mode in __event_modes:
        return await zm_client.detect(eid=eid, mid=mid)

    elif _mode == "shm":
        return await zm_client.detect(mid=mid)
    else:
        raise ValueError(f"Unknown mode: {_mode}")

if __name__ == "__main__":
    loop = uvloop.new_event_loop()
    asyncio.set_event_loop(loop)
    _start = time.perf_counter()
    ENV_VARS = ClientEnvVars()
    g: GlobalConfig = create_global_config()
    g.Environment = ENV_VARS
    logger.debug(f"About to run event loop -  g [type: {type(g)}] ")
    detections = loop.run_until_complete(main())
    # Allow 250ms for aiohttp SSl session context to close properly
    loop.run_until_complete(asyncio.sleep(0.25))
    logger.debug(f"DETECTIONS FROM uvloop: {detections}")
    logger.info(
        f"perf::FINAL:: Total: {time.perf_counter() - _start:.5f} seconds"
    )
