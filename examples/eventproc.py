#!/opt/zm_ml/data/venv/bin/python3
#!/usr/bin/env python3


from __future__ import annotations
import logging.handlers
import sys
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import uvloop
import asyncio

import zm_ml
from zm_ml.Shared.Models.validators import str2path
from zm_ml.Shared.configs import GlobalConfig
from zm_ml.Client.Models.config import ClientEnvVars
from zm_ml.Client.main import (
    parse_client_config_file,
    create_global_config,
    create_logs,
)

if TYPE_CHECKING:
    from zm_ml.Client.main import ZMClient

__doc__ = """
An example that uses ZM's EventStartCommand/EventEndCommand mechanism to run 
object detection on a ZM event using ZoMi ML library.
"""
__version__ = "0.0.1a4"
# Setup basic console logging (hook into library logging)
logger: logging.Logger = create_logs()
zm_client: Optional[ZMClient] = None


def _parse_cli():
    from argparse import ArgumentParser

    parser = ArgumentParser(prog="eventproc.py", description=__doc__)
    parser.add_argument(
        "--event-start",
        help="This is the start of an event",
        action="store_true",
        dest="event_start",
        default=True,
    )
    parser.add_argument(
        "--event-end",
        help="This is the end of an event",
        action="store_false",
        dest="event_start",
    )
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
    # python3 ./zm_ml/examples/eventproc.py -E -e 1263 -m 7 -D -C /etc/zm/client.yml
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
    global g, zm_client
    _mode = ""
    _start = time.perf_counter()
    # Do all the config stuff and setup logging
    lp = f"eventstart::" if args.event_start else f"eventend::"
    eid = args.eid
    mid = args.mid
    event_start = args.event_start
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
        cfg_file = str2path(cfg_file)
    assert cfg_file, "No config file supplied via CLI or ENV"
    g.config_file = cfg_file
    logger.info(
        f"{lp} Event: {args.event} [Event ID: {eid}] || SHM: {args.shm} ||"
        f"{' || Monitor ID: ' if mid else ''}{mid if mid else ''} || "
        f"Config File: {cfg_file}"
    )
    g.config = parse_client_config_file(cfg_file)
    zm_client = zm_ml.Client.main.ZMClient(global_config=g)
    _end_init = time.perf_counter()
    __event_modes = ["event", ""]
    if _mode in __event_modes:
        # set live or past event
        zm_client.is_live_event(args.live)
        return await zm_client.detect(eid=eid, mid=g.mid)
    elif _mode == "shm":
        raise NotImplementedError("SHM mode is a work in progress")
        # return await zm_client.detect(mid=mid)
    else:
        raise ValueError(f"Unknown mode: {_mode}")


if __name__ == "__main__":
    # file name
    filename = Path(__file__).stem
    args = _parse_cli()
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
    loop.run_until_complete(zm_client.clean_up())
    if not loop.is_closed():
        loop.close()
    logger.info(f"perf::FINAL:: Total: {time.perf_counter() - _start:.5f} seconds")
