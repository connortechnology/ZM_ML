#!/usr/bin/python3

"""
Call this script in a ZoneMinder filter and pass the event ID as argument.
This script gets data about an event from ZoneMinder and uploads the video file to Google Drive.

Before first usage:
- enter login credentials in this script or use ENV variables: ZM_API_USER, ZM_API_PASS, ZM_API_URL, ZM_EID
- Add pi-user to the www-group and run `sudo chown -R pi:www-data .` in this repo.
- Make sure www-data group has WRITE permissions to the folder where this script is located.
- install and configure rclone.
- Add the rclone remote name to the rclone_remote variable below.

"""
# Set to True if you are only testing this script and don't want to upload anything.
testing = False
# The name of the remote that has been configured in rclone.
rclone_remote = "GoogleDrive"
# login credentials (leave empty if you do not use authentication for ZM)
username = ""  # Can be set using environment: ZM_API_USER
password = ""  # Can be set using environment: ZM_API_PASS
# ZM api url (Example: http://localhost/zm/api or https://zm.example.com/zm/api)
zm_api_url = ""  # Can be set using environment: ZM_API_URL

import subprocess
import sys
from typing import Union, Dict, Optional

import requests
from datetime import datetime
from pathlib import Path
import logging

# logger
log_format = logging.Formatter(
    "%(asctime)s.%(msecs)04d %(name)s[%(process)s] %(levelname)s %(module)s:%(lineno)d -> %(message)s",
    "%m/%d/%y %H:%M:%S",
)
logger = logging.getLogger("google_drive_upload")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setFormatter(log_format)
logger.addHandler(console_handler)
file_handler = logging.FileHandler(Path(__file__).parent / "google_drive_upload.log")
file_handler.setFormatter(log_format)
logger.addHandler(file_handler)

# Use env vars instead of hard coding credentials (python dotenv used for .env files)
try:
    import dotenv
except ImportError:
    logger.error(
        "dotenv not installed, cannot load .env files. Please install it with `pip install python-dotenv`"
    )
else:
    logger.debug(f"Attempting to load .env file")
    dotenv.load_dotenv()
from os import environ as os_env

if "ZM_API_USER" in os_env:
    username = os_env["ZM_API_USER"]
if "ZM_API_PASS" in os_env:
    password = os_env["ZM_API_PASS"]
if "ZM_API_URL" in os_env:
    zm_api_url = os_env["ZM_API_URL"]

event_id = None
# get event id from ENV VAR, used for testing
if "ZM_EID" in os_env:
    event_id = os_env["ZM_EID"]


def zm_api_login(login_url: str, login_data: Optional[Dict[str, str]]):

    token = None
    try:
        r = requests.post(login_url, data=login_data)
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        logger.error(f"Login failed: {e}")
        raise e
    else:
        data = r.json()
        if 'version' in data:
            token = "auth_disabled"
            msg = f"ZoneMinder version: {data['version']} API version: {data['apiversion']}"
            if "access_token" in data:
                token = data["access_token"]
                logger.info(
                    f"Login successful. {msg}"
                )
            else:
                logger.warning(
                    f"It appears ZM authentication is disabled. {msg}"
                )
    return token

def check_filename_incomplete(file_path: Path, retried: bool = False):
    filename = file_path.name
    if filename.startswith("incomplete"):
        if retried:
            raise Exception(
                f"File {filename} not found in event folder even after a 10 second recheck"
            )
        logger.error(
            f"Video file is incomplete."
        )
        return False
    return True


def get_event_data(event_url: str) -> Union[dict, None]:
    """Get event data from ZoneMinder."""
    logger.info(f"Getting event info from ZM API...")
    try:
        r = requests.get(event_url)
        r.raise_for_status()
    except requests.exceptions.HTTPError as err:
        logger.error(f"HTTP error: {err}")
        raise err
    else:
        data = r.json()
        if data.get("success") is False:
            logger.error(f"API Event call to endpoint [{_event_url}] wasn't successful.")
            raise Exception("API Event call wasn't successful.")
        return data


def get_file_data(src_data: Dict):
    file_path = src_data["event"]["Event"]["FileSystemPath"]
    file_name = src_data["event"]["Event"]["DefaultVideo"]
    return file_path, file_name


if __name__ == "__main__":
    zm_api_url = zm_api_url.rstrip("/")
    logger.info(f"Logging level: {logging._levelToName[logger.getEffectiveLevel()]}")
    # given event ID
    if len(sys.argv) > 1:
        event_id = sys.argv[1]
    if event_id is None:
        logger.error("No event ID given")
        raise ValueError("No event ID given")
    # Login
    login = None
    if username and password:
        login = {"user": username, "pass": password}
    token = zm_api_login(f"{zm_api_url}/host/login.json", login)
    if token is None:
        raise ValueError(
            "Couldn't get a token for authentication. Check login data and/or API."
        )
    else:
        if token == "auth_disabled":
            token = None
        else:
            token = f"?token={token}"

        _event_url = f"{zm_api_url}/events/{event_id}.json{token}"
        data = get_event_data(_event_url)

        # Depending on storage schema (shallow, deep, etc.) the path
        # to the video file is different. One would need to call sqlalchemy
        # to interface with DB to get the storage schema and then determine relative path.

        path, filename = get_file_data(data)
        if not filename:
            if not testing:
                logger.error(f"No video file found for event {event_id}, are you passing through or encoding a video for the event?")
                raise Exception("No video file found for event")
            else:
                logger.info(f"TESTING:: No video file found for event {event_id}, are you passing through or encoding a video for the event?")

        fullpath = Path(path) / filename
        logger.debug(f"Video file name: {filename}")
        _file_complete = check_filename_incomplete(fullpath)
        if not _file_complete:
            logger.warning(f"Video file is incomplete. Retrying in 10 seconds...")
            import time

            time.sleep(10)
            logger.debug(f"10 second sleep over, retrying...")
            data = get_event_data(_event_url)
            path, filename = get_file_data(data)
            fullpath = Path(path) / filename
            _file_complete = check_filename_incomplete(fullpath, retried=True)
            if not _file_complete:
                raise Exception(
                    f"File {filename} is still incomplete even after a 10 second recheck"
                )
        logger.debug(f"Full source path of video file: {fullpath.as_posix()}")
        if (not fullpath.exists or not fullpath.is_file()) and not testing:
            logger.error("Requested file does not exist or is not a file.")
            raise Exception(f"{fullpath.as_posix()} doesn't exist on this machine!")
        elif (not fullpath.exists or not fullpath.is_file()) and testing:
            logger.info(f"TESTING:: Source file does not exist or is not a file.")

        start_time = datetime.strptime(
            data["event"]["Event"]["StartTime"], "%Y-%m-%d %H:%M:%S"
        )
        date_path = datetime.strftime(start_time, "%Y-%m-%d")
        _time = start_time.strftime("%H_%M_%S")
        year = start_time.strftime("%Y")
        month = start_time.strftime("%m")
        destination = f"ZoneMinder/{year}/{month}/{date_path}/"

        rclone_command = [
            "rclone",
            "copy",
            "--update",
            "--retries",
            "3",
            "-vv",
            f"{fullpath.as_posix()}",
            f"{rclone_remote}:{destination}",
        ]
        if testing:
            logger.info(f"TESTING:: Adding '--dry-run' to Rclone command")
            rclone_command.insert(2, "--dry-run")
        logger.debug(f"Attempting to upload file using command:: {' '.join(rclone_command)}")

        ran = subprocess.run(rclone_command, capture_output=True, text=True)
        if ran.stdout:
            logger.info(f"Rclone output:\n{ran.stdout}")
        if ran.stderr:
            logger.error(f"Rclone error:\n{ran.stderr}")

        if ran.returncode == 0:
            logger.debug("Successful upload, writing note to event.")
            note = f"Uploaded to Google Drive on {datetime.now()}"
            try:
                r = requests.put(
                    f"{zm_api_url}/events/{event_id}.json{token}",
                    data={
                        "Event[Notes]": note,
                    },
                )
                r.raise_for_status()
            except requests.exceptions.HTTPError as err:
                logger.error(f"HTTP error: {err}")
                raise err
            else:
                logger.info(f"Successfully wrote note '{note}' to event: {event_id}.")
        else:
            logger.error("Couldn't upload to Google Drive.")
            raise Exception("Couldn't upload to Google Drive.")
