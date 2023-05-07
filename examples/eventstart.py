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
from zm_ml.Shared.Models.validators import str_to_path
from zm_ml.Shared.configs import ClientEnvVars, GlobalConfig
from zm_ml.Client.main import (
    parse_client_config_file,
    create_global_config,
    create_logs,
)

__version__ = "0.0.0-a1"
__version_type__ = "dev"
# Setup basic console logging (hook into library logging)
logger: logging.Logger = create_logs()


def _parse_cli():
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="eventstart.py", description=__doc__)
    parser.add_argument(
        "--shm-mode",
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
        "--event-mode",
        "-E",
        action="store_true",
        dest="event",
        help="Run in event mode (triggered by a ZM event)",
    )
    parser.add_argument(
        "--event-id",
        "--eid",
        "-e",
        help="Event ID to process (Required for --event/-E)",
        type=int,
        dest="eid",
        default=0,
    )
    parser.add_argument(
        "--monitor-id",
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
    if args.shm:
        _mode = "shm"
        logger.debug(f"{lp} Running in shared memory mode (Monitor ID REQUIRED)")
        if mid == 0:
            logger.error(f"{lp} Monitor ID is required for shared memory mode")
            sys.exit(1)
    elif args.event:
        _mode = "event"
        logger.debug(f"{lp} Running in event mode")
        if eid == 0:
            logger.error(f"{lp} Event ID is required for event mode")
            sys.exit(1)
        if mid == 0:
            logger.warning(
                f"{lp} When monitor ID is not supplied in event mode, ZoneMinder DB is queried for it"
            )

    if "config" in args and args.config:
        # logger.info(f"Configuration file supplied as: {args.config}")
        cfg_file = args.config
    else:
        logger.warning(
            f"No config file supplied, checking ENV: {g.Environment.client_conf_file}"
        )
        if g.Environment.client_conf_file:
            cfg_file = g.Environment.client_conf_file
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
    zm_client.is_live_event(args.live)
    _end_init = time.perf_counter()
    __event_modes = ["event", ""]
    if _mode in __event_modes:
        return await zm_client.detect(eid=eid, mid=g.mid)
    elif _mode == "shm":
        raise NotImplementedError("SHM mode is a work in progress")
        # return await zm_client.detect(mid=mid)
    else:
        raise ValueError(f"Unknown mode: {_mode}")


if __name__ == "__main__":
    # file name
    filename = Path(__file__).stem
    logger.debug(f"Starting {filename}...")
    loop = uvloop.new_event_loop()
    asyncio.set_event_loop(loop)
    _start = time.perf_counter()
    ENV_VARS = ClientEnvVars()
    logger.debug(f"ENV VARS: {ENV_VARS}")
    g: GlobalConfig = create_global_config()
    g.Environment = ENV_VARS
    detections = loop.run_until_complete(main())
    # Allow 250ms for aiohttp SSL session context to close properly
    loop.run_until_complete(asyncio.sleep(0.25))
    logger.debug(f"DETECTIONS FROM uvloop: {detections}")
    print(f"DETECTIONS FROM uvloop: {detections}")
    logger.info(f"perf::FINAL:: Total: {time.perf_counter() - _start:.5f} seconds")
