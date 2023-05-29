#!/usr/bin/env python3
"""
A script to be used with ZoneMinder filter system. The filtering system can pass some CLI args
to the script. This is the beggining of the exploratory work on how to integrate this.

 This has 2 modes.

 - config mode: you can use ZM ML like monitor config to only annotate certain objects
     - This allows for filtering by label/group/zone
 - all: annotate anything it finds in the event
"""
import logging
import warnings
import os
import sys

LP: str = "filter:annotate:"
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

if __name__ == "__main__":
    logger.info(f"{LP} Starting")
    logger.info(f"{LP} CLI args: { {k: v for k, v in enumerate(sys.argv)} }")

