import logging
import subprocess
import time
from pathlib import Path
from typing import List, NoReturn, Union, Optional, Tuple, Dict

import numpy as np
import requests
import urllib3.exceptions
from pydantic import BaseModel, Field, AnyUrl

from ..main import get_global_config
from ...Shared.configs import GlobalConfig
from ..Models.config import MLNotificationSettings
from ..Log import CLIENT_LOGGER_NAME

logger = logging.getLogger(CLIENT_LOGGER_NAME)
g: Optional[GlobalConfig] = None
LP: str = "shell_script:"

class ShellScriptNotification:
    config: Optional[MLNotificationSettings.ShellScriptNotificationSettings]
    def __init__(self, config: Optional[MLNotificationSettings.ShellScriptNotificationSettings] = None):
        global g

        lp: str = "{LP}:init:"
        logger.debug(f"{lp} creating object")
        g = get_global_config()
        if config:
            self.config = config
        else:
            logger.debug(f"{lp} No config passed, using global config")
            self.config = g.config.notifications.shell_script


    def run(self):
        lp: str = "{LP}:run:"
        script_path = Path(self.config.script)
        if self.config.enabled:
            if not script_path.is_file():
                raise FileNotFoundError(f"Script file '{script_path.as_posix()}' not found/is not a valid file")
            if not self.config.I_AM_AWARE_OF_THE_DANGER_OF_RUNNING_SHELL_SCRIPTS != "YeS i aM awaRe!":
                raise ValueError("You must set I_AM_AWARE_OF_THE_DANGER_OF_RUNNING_SHELL_SCRIPTS to: YeS i aM awaRe!")
            cmd_array = [script_path.as_posix()]
            try:
                x = subprocess.run(cmd_array, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"{lp} Shell script failed with exit code {e.returncode}")
                logger.error(f"{lp} STDOUT: {e.stdout}")
                logger.error(f"{lp} STDERR: {e.stderr}")
                raise e
            logger.debug(f"{lp} STDOUT->{x.stdout}")
            if x.stderr:
                logger.error(f"{lp} STDERR-> {x.stderr}")
        else:
            logger.debug(f"Shell script notification disabled, skipping")

    def send(self):
        return self.run()
