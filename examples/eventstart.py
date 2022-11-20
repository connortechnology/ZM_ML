#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
"""An example that uses ZM's EventStartCommand mechanism to run object detection on a ZM event."""

import sys
import time
from pathlib import Path
import logging
from typing import Optional

from zm_ml.Client.main import ZMClient


logger = logging.getLogger("EventStart")
formatter = logging.Formatter(
    "%(asctime)s.%(msecs)04d %(name)s[%(process)s] %(levelname)s -> %(message)s",
    "%m/%d/%y %H:%M:%S",
)
stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)


def _parse_cli():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--config', '-C', help='Config file to use')
    parser.add_argument('--eid', '-E', help='Event ID to process', required=True, type=int)
    parser.add_argument('--mid', '-M', help='Monitor ID to process [Optional]', type=int)
    parser.add_argument('--debug', '-D', help='Enable debug logging', action='store_true')
    parser.add_argument('--live', '-L', help='Tells script this is a live event', action='store_true')
    parser.add_argument('--docker', help='script is running inside of a Docker container', action='store_true')
    args = parser.parse_args()
    logger.debug(f"CLI Args: {args}")
    return args


if __name__ == "__main__":
    args = _parse_cli()
    eid: int = args.eid
    mid: int = 0
    if 'config' in args and args.config:
        logger.info(f"Configuration file supplied as: {args.config}")
        cfg_file = Path(args.config)
    else:
        cfg_file = Path(__file__).parent / "configs/client.yml"
    if 'eid' in args and args.eid:
        logger.info(f"Event ID supplied as: {args.eid}")
        eid = args.eid
    if 'mid' in args and args.mid:
        logger.info(f"Monitor ID supplied as: {args.mid}")
        mid = args.mid

    _start = time.perf_counter()
    logger.info("Starting ZM ML Example for EventStartCommand")
    logger.info(f"Event ID: {eid = }{' || Monitor ID: ' if mid else ''}{mid if mid else ''}")
    logger.info(f"Script started with config file: {cfg_file}")
    ZM = ZMClient(cfg_file)
    _end_init = time.perf_counter()
    detections = ZM.detect(eid=eid, mid=mid)
    logger.info(f"perf::FINAL:: Total: {time.perf_counter() - _start:.5f} - "
                f"[init: {_end_init - _start:.5f}] - [detect: "
                f"{time.perf_counter() - _end_init:.5f}]")
    print(detections)
