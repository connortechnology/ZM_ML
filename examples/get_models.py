"""Use a config file to download files"""

from pathlib import Path
import logging
import sys
import argparse
from typing import List, Dict, NamedTuple

import requests
import yaml
from pydantic import BaseModel, Field, validator


class Download(BaseModel):
    name: str = Field("n/a", description="The name of this download")
    enabled: bool = Field(True, description="Whether or not to download this sections urls")
    destination: Path = Field(..., description="The destination directory to download to")
    urls: List[str] = Field(..., description="The urls to download")

    @validator("enabled", pre=True)
    def _enabled(cls, v, field, values, config):
        if v is None:
            v = True
        if isinstance(v, bool):
            return v
        v = str(v)
        v = v.strip().casefold() in ("true", "1", "yes", "y", "t", "on", "enable", "enabled")
        return v


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)04d %(name)s[%(process)s] %(levelname)s %(module)s:%(lineno)d -> %(message)s",
        "%m/%d/%y %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--config", help="Path to configuration file", required=True)
    parser.add_argument("-D", "--debug", help="Debug logging", action="store_true")
    args, _ = parser.parse_known_args()
    config_file = Path(args.config)
    if args.debug:
        logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    logger.info(f"Using config file '{config_file}'")
    cfg: Dict
    try:
        cfg = yaml.safe_load(config_file.read_text())
    except Exception as e:
        logger.error(f"Could not load config file '{config_file}': {e}")
        sys.exit(1)

    downloads: List[Download]
    _d = cfg.get("downloads", [])
    downloads = [Download(**d) for d in _d]
    if not downloads:
        raise ValueError("No downloads specified in config file")
    logger.info(f"Found {len(downloads)} downloads")
    for d in downloads:
        if not d.enabled:
            logger.info(f"Skipping disabled download '{d.name}'")
            continue
        logger.info(f"Downloading to '{d.destination}'")
        for url in d.urls:
            logger.info(f"Downloading '{url}'")
            r = requests.get(url)
            if r.status_code != 200:
                logger.error(f"Could not download '{url}': {r.status_code}")
                continue
            fname = url.split("/")[-1]
            dest = d.destination / fname
            logger.info(f"Saving to '{dest}'")
            dest.write_bytes(r.content)





