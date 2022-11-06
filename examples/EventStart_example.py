#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import sys
import time
from pathlib import Path
import logging
from typing import Optional

from zm_ml.Client.main import ZMClient


logger = logging.getLogger("ZM ML Example")
formatter = logging.Formatter(
    "%(asctime)s.%(msecs)04d %(name)s[%(process)s] %(levelname)s -> %(message)s",
    "%m/%d/%y %H:%M:%S",
)
stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--live', action='store_true', help='Tells the script this is a live event request')
    _start = time.perf_counter()
    cfg_file = Path(__file__).parent.parent / "configs/example_detection_config.yml"
    logger.info("Starting ZM ML Example for EventStartCommand")
    # ZM EventStartScript / EventEndScript supplies EID as ARG1 and MID as ARG2
    args = sys.argv[1:]
    num_args = len(args)
    logger.debug(f"{num_args = } -- {args = }")
    mid: Optional[int] = None
    if num_args == 0:
        logger.error("Not enough arguments supplied! Minimum is 1 [Event ID]")
        sys.exit(1)
    elif num_args > 2:
        logger.error("Too many arguments supplied! Maximum is 2 [Event ID] [Monitor ID]")
        sys.exit(1)
    elif num_args == 2:
        mid = int(args[1])
    else:
        logger.debug(f"WTF -> {num_args = } -- {args = }")
    eid = int(args[0])
    logger.info(f"Event ID: {eid = }{' || Monitor ID: ' if mid else ''}{mid if mid else ''}")
    logger.info(f"Script started with config file: {cfg_file}")
    ZM = ZMClient(cfg_file)
    logger.info(f"Running detection for event {eid}")
    detections = ZM.detect(eid=eid, mid=mid)

    logger.info(f"perf:: ---- END ---- total: {time.perf_counter() - _start:.5f} second(s)")


