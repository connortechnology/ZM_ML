import datetime
import inspect
import platform
from collections import namedtuple
import os
import logging
import sys
import argparse
import time
from typing import Optional, Tuple, Union, List, Dict
from pathlib import Path

# Change these if you want to install to a different location
DEFAULT_DATA_DIR = "/opt/zm_ml/var/lib/zm_ml"
DEFAULT_CONFIG_DIR = "/opt/zm_ml/etc/zm_ml"
DEFAULT_LOG_DIR = "/opt/zm_ml/var/logs/zm_ml"
DEFAULT_OCTAL_PERMISSONS = 0o640
# default ML models to install (SEE: available_models{})
DEFAULT_MODELS = ["yolov4", "yolov4_tiny", "yolov7", "yolov7_tiny"]

# parse .env file using pyenv
def parse_env_file(env_file: Path) -> Dict[str, str]:
    """Parse .env file using pyenv.

    :param env_file: Path to .env file.
    :type env_file: Path
    :returns: Dict of parsed .env file.
    """
    import dotenv

    envs = dotenv.dotenv_values(dotenv_path=env_file)
    return envs


# Do not change these unless you know what you are doing
available_models = {
    "yolov4": {
        "folder": "yolo/v4",
        "model": [
            "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights",
            "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4_new.weights",
        ],
        "config": [
            "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
            "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4_new.cfg",
        ],
    },
    "yolov4_tiny": {
        "folder": "yolo/v4_tiny",
        "model": [
            "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights"
        ],
        "config": [
            "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
        ],
    },
    "yolov4_p6": {
        "folder": "yolo/v4_p6",
        "model": [
            "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-p6.weights"
        ],
        "config": [
            "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-p6.cfg"
        ],
    },
    "yolov7": {
        "folder": "yolo/v7",
        "model": [
            "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov7.weights"
        ],
        "config": [
            "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov7.cfg"
        ],
    },
    "yolov7_tiny": {
        "folder": "yolo/v7_tiny",
        "model": [
            "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov7-tiny.weights"
        ],
        "config": [
            "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov7-tiny.cfg"
        ],
    },
    "yolov7x": {
        "folder": "yolo/v7x",
        "model": [
            "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov7x.weights"
        ],
        "config": [
            "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov7x.cfg"
        ],
    },
    "coral_tpu": {
        "folder": "coral_tpu",
        "model": [
            "https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
            "https://github.com/google-coral/test_data/raw/master/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite",
            "https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite",
            "https://github.com/google-coral/test_data/raw/master/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite",
            "https://github.com/google-coral/test_data/raw/master/efficientdet_lite3_512_ptq_edgetpu.tflite",
        ],
        "config": None,
    },
}
## Logging
logger = logging.getLogger("install_zm_ml")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s.%(msecs)04d %(name)s[%(process)s] %(levelname)s %(module)s:%(lineno)d -> %(message)s",
    "%m/%d/%y %H:%M:%S",
)
console = logging.StreamHandler(stream=sys.stdout)
console.setFormatter(formatter)
logger.addHandler(console)

## Misc.
__version__ = "0.0.1a1"
__dependancies__ = "psutil", "requests", "dotenv"
__doc__ = """Install ZM-ML Server / Client"""

# Logic
tst_msg_wrap = "testing!! [will not actually execute]"


def test_msg(msg):
    caller = inspect.stack()[1][3]
    lineno = inspect.stack()[1][2]
    # add lineno to msg
    msg = f"{caller}({lineno}): {msg}"
    logger.warning(f"{tst_msg_wrap.split(' ')[0]} {msg} {tst_msg_wrap.split('!! ')[1]}")


def get_distro() -> namedtuple:
    release_data = platform.freedesktop_os_release()
    nmd_tpl = namedtuple("Distro", release_data.keys())
    return nmd_tpl(**release_data)


def check_imports():
    for imp_name in __dependancies__:
        try:
            import imp_name
        except ImportError:
            logger.error(
                f"Missing dependency: {imp_name}"
                f":: Please install dependency and try again."
            )
            return False
        else:
            logger.debug(f"Found dependency: {imp_name}")
    return True


def get_web_user() -> Tuple[Optional[str], Optional[str]]:
    """Get the user that runs the web server using psutil library.

    :returns: The user that runs the web server.
    :rtype: str

    """
    import psutil
    import grp

    # get group name from gid

    www_daemon_names = ("httpd", "hiawatha", "apache", "apache2", "nginx")
    hits = []
    proc_names = []
    for proc in psutil.process_iter():
        if proc.name() in www_daemon_names and proc.name() not in proc_names:
            hits.append((proc.username(), grp.getgrgid(proc.gids().real).gr_name))
            proc_names.append(proc.name())
    ret_ = False
    if len(hits) == 1:
        ret_ = True
    elif len(hits) > 1:
        logger.warning(
            f"Multiple web server processes found ({proc_names}), using first one"
        )
        ret_ = True
    if ret_:
        return hits[0]
    return None, None


def parse_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--install-log",
        type=str,
        default=f"./zm-ml_install_{datetime.datetime.now()}.log",
        help="Log file to write to",
    )
    parser.add_argument(
        "--install-log-rotate",
        action="store_true",
        help="install logrotate file to system [if logrotate is available]",
    )

    parser.add_argument(
        "--install-type",
        choices=["server", "client", "both"],
        default="client",
        help="Install candidates",
    )
    parser.add_argument(
        "--install-secrets",
        action="store_true",
        help="install default secrets.yml [will use a numbered backup system]",
    )

    parser.add_argument(
        "--test", "-t", action="store_true", dest="test", help="Run in test mode"
    )
    parser.add_argument(
        "--web-user",
        "-W",
        help="Web server user [leave empty to auto-detect]",
        type=str,
        default="",
    )
    parser.add_argument(
        "--web-group",
        "-G",
        help="Web server group [leave empty to auto-detect]",
        type=str,
        default="",
    )
    parser.add_argument(
        "--dir-config",
        help=f"Config directory [{DEFAULT_CONFIG_DIR}]",
        default="",
        type=str,
        dest="config_dir",
    )
    parser.add_argument(
        "--dir-data",
        help=f"Variable Data directory [{DEFAULT_DATA_DIR}]",
        dest="data_dir",
        default="",
        type=str,
    )
    parser.add_argument(
        "--dir-log",
        help=f"Log directory [{DEFAULT_LOG_DIR}]",
        default="",
        dest="log_dir",
        type=str,
    )
    parser.add_argument(
        "--debug", "-D", help="Enable debug logging", action="store_true", dest="debug"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--interactive",
        "-I",
        action="store_true",
        dest="interactive",
        help="Run in interactive mode",
    )
    # get octal permissions
    parser.add_argument(
        "--create_mode",
        help=f"installation octal file permissions [{DEFAULT_OCTAL_PERMISSONS}]",
        type=lambda x: int(x, 8),
        default=DEFAULT_OCTAL_PERMISSONS,
    )
    parser.add_argument(
        "--add-model",
        action="append",
        dest="models",
        default=DEFAULT_MODELS,
        choices=available_models.keys(),
    )
    parser.add_argument(
        "--force-models",
        action="store_true",
        dest="force_models",
        help="Force model installation [Overwrite existing models]",
    )
    return parser.parse_args()


def chown(path: Path, user: Union[str, int], group: Union[str, int]):
    import pwd
    import grp

    if isinstance(user, str):
        user = pwd.getpwnam(user).pw_uid
    if isinstance(group, str):
        group = grp.getgrnam(group).gr_gid
    if not args.test:
        os.chown(path, user, group)
    else:
        test_msg(f"chown {user}:{group} {path}")


def chmod(path: Path, mode: int):
    if not args.test:
        os.chmod(path, mode)
    else:
        test_msg(f"chmod {mode:o} {path}")


def chown_mod(path: Path, user: Union[str, int], group: Union[str, int], mode: int):
    chown(path, user, group)
    chmod(path, mode)


def create_dir(path: Path, user: Union[str, int], group: Union[str, int], mode: int):
    msg = f"Created directory: {path} with user: {user} and group: {group} and mode: {mode:o}"
    if not args.test:
        path.mkdir(parents=True, exist_ok=True, mode=mode)
        chown_mod(path, user, group, mode)
        logger.info(msg)
    else:
        test_msg(msg)


def show_config(cli_args: argparse.Namespace):
    logger.info(f"Configuration:\n")
    for key, value in vars(cli_args).items():
        logger.info(f"{key}: {value}")
    if args.interactive:
        input(
            "Press Enter to continue if all looks fine, otherwise use 'Ctrl+C' to exit and edit CLI options..."
        )
    else:
        logger.info(
            "This is a non-interactive session, continuing with installation... in 5 seconds, press 'Ctrl+C' to exit"
        )
        time.sleep(5)


def do_web_user():
    global web_user, web_group
    web_user, web_group = args.web_user, args.web_group
    if not web_user or not web_group:
        if not web_user:
            if interactive:
                web_user = input("Web server username [Leave blank to try auto-find]: ")
                if not web_user:
                    logger.info(f"Web server user auto-find: {web_user}")
                    web_user = get_web_user()
            else:
                web_user, _ = get_web_user()
                logger.info(f"Web server user auto-find: {web_user}")

        if not web_group:
            if interactive:
                web_group = input("Web server group [Leave blank to try auto-find]: ")
                if not web_group:
                    _, web_group = get_web_user()
                    logger.info(f"Web server group auto-find: {web_group}")

            else:
                _, web_group = get_web_user()
                logger.info(f"Web server group auto-find: {web_group}")

    if not web_user or not web_group:
        _missing = ""
        if not web_user and web_group:
            _missing = "user"
        elif web_user and not web_group:
            _missing = "group"
        else:
            _missing = "user and group"

        msg = f"Unable to determine web server {_missing}"
        if not args.test:
            logger.error(msg)
            sys.exit(1)
        else:
            test_msg(msg)


def install_dirs(
    dest_dir: Path,
    default_dir: str,
    dir_type: str,
    sub_dirs: Optional[List[str]] = None,
):
    if sub_dirs is None:
        sub_dirs = []
    if not dest_dir:
        if interactive:
            dest_dir = input(
                f"Set {dir_type} directory [Leave blank to use default {default_dir}]: "
            )
        else:
            logger.info(f"Using default {dir_type} directory: {default_dir}")

    if not dest_dir:
        dest_dir = default_dir
    dest_dir = Path(dest_dir)
    if dir_type.casefold() == "data":
        global data_dir
        data_dir = dest_dir
        print(f"Data directory: {data_dir}")
    elif dir_type.casefold() == "log":
        global log_dir
        log_dir = dest_dir
        print(f"Log directory: {log_dir}")
    elif dir_type.casefold() == "config":
        global cfg_dir
        cfg_dir = dest_dir
        print(f"Config directory: {cfg_dir}")
    if not dest_dir.exists():
        logger.warning(f"{dir_type} directory {dest_dir} does not exist!")
        if interactive:
            x = input(f"Create {dir_type} directory [Y/n]?... 'n' will exit! ")
            if x.strip().casefold() == "n":
                logger.error(f"{dir_type} directory does not exist, exiting...")
                sys.exit(1)
            create_dir(dest_dir, web_user, web_group, create_mode)
        else:
            logger.info(f"Creating {dir_type} directory...")
            create_dir(dest_dir, web_user, web_group, create_mode)
        # create sub-folders
        if sub_dirs:
            for _sub in sub_dirs:
                _path = dest_dir / _sub
                create_dir(_path, web_user, web_group, create_mode)


def download_file(url: str, dest: Path, user: str, group: str, mode: int):
    msg = f"Downloading {url}..."
    if not args.test:
        logger.info(msg)
        import requests

        r = requests.get(url, allow_redirects=True)
        if r.status_code == 200:
            dest.write_bytes(r.content)
            chown_mod(dest, user, group, mode)
        else:
            logger.error(f"Failed to download [code: {r.status_code}] {url}!")
    else:
        test_msg(msg)


def copy_file(src: Path, dest: Path, user: str, group: str, mode: int):
    msg = f"Copying {src} to {dest}..."
    if not args.test:
        import shutil

        logger.info(msg)
        shutil.copy(src, dest)
        chown_mod(dest, user, group, mode)
    else:
        test_msg(msg)


def get_pkg_manager():
    distro = get_distro()
    binary, prefix = "apt", "install -y"
    if distro.ID in ("debian", "ubuntu", "raspbian"):
        pass
    elif distro.ID in ("centos", "fedora", "rhel"):
        binary = "yum"
    elif distro.ID == "fedora":
        binary = "dnf"
    elif distro.ID in ("arch", "manjaro"):
        binary = "pacman"
        prefix = "-Sy"
    elif distro.ID == "gentoo":
        binary = "emerge"
        prefix = "-a"
    elif distro.ID == "alpine":
        binary = "apk"
        prefix = "add -q"
    elif distro.ID == "suse":
        binary = "zypper"
        prefix = "install -y"
    elif distro.ID == "void":
        binary = "xbps-install"
        prefix = "-Sy"
    elif distro.ID == "nixos":
        binary = "nix-env"
        prefix = "-i"
    elif distro.ID == "freebsd":
        binary = "pkg"
        prefix = "-y"
    elif distro.ID == "openbsd":
        binary = "pkg_add"
        prefix = "-I"
    elif distro.ID == "netbsd":
        binary = "pkgin"
        prefix = "-y"
    elif distro.ID == "solus":
        binary = "eopkg"
        prefix = "-y"
    elif distro.ID == "windows":
        binary = "choco"
        prefix = "install -y"
    elif distro.ID == "macos":
        binary = "brew"
        prefix = "install -y"

    return binary, prefix


def install_host_dependencies(_type: str):
    if _type.strip().casefold() not in ["server", "client"]:
        logger.error(f"Invalid type '{_type}'")
    else:
        inst_binary, inst_prefix = get_pkg_manager()
        dependencies = {
            "apt": {
                "client": ["gifsicle", "libgeos-dev"],
                "server": [],
            },
            "yum": {
                "client": ["gifsicle", "geos-devel"],
                "server": [],
            },
            "pacman": {
                "client": ["gifsicle", "geos"],
                "server": [],
            },
            "zypper": {
                "client": ["gifsicle", "geos"],
                "server": [],
            },
            "dnf": {
                "client": ["gifsicle", "geos-devel"],
                "server": [],
            },
            "apk": {
                "client": ["gifsicle", "geos-devel"],
                "server": [],
            },
        }
        deps = []

        import subprocess

        if _type == "server":
            logger.info("Installing server host dependencies...")
            deps = dependencies[inst_binary]["server"]
        else:
            logger.info("Installing client host dependencies...")
            deps = dependencies[inst_binary]["client"]
        if deps:
            install_cmd = f"{inst_binary} {inst_prefix} {' '.join(deps)}"
            msg = f"Running command: {install_cmd}"
            if os.geteuid() != 0:
                logger.warning(f"You must be root to install dependencies! Please install the following "
                               f"dependencies manually: {' '.join(deps)}")
            else:
                if not args.test:
                    logger.info(msg)
                    try:
                        subprocess.run(
                            install_cmd,
                            shell=True,
                            check=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                        )
                    except subprocess.CalledProcessError as e:
                        logger.error(
                            f"Error installing host dependencies: {e.stdout.decode('utf-8')}"
                        )
                        sys.exit(1)
                    else:
                        logger.info("Host dependencies installed successfully")
                else:
                    test_msg(msg)


def main():
    if debug:
        logger.setLevel(logging.DEBUG)
        console.setLevel(logging.DEBUG)
        file_handler.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled!")
    if testing:
        logger.warning("Running in test mode")
    do_web_user()
    if not check_imports():
        msg = f"Missing dependencies, exiting..."
        if not args.test:
            logger.error(msg)
            sys.exit(1)
        else:
            test_msg(msg)
    else:
        logger.info("All dependencies found")
    install_dirs(
        data_dir, DEFAULT_DATA_DIR, "Data", sub_dirs=["models", "push", "scripts"]
    )
    global model_dir
    model_dir = data_dir / "models"
    install_dirs(cfg_dir, DEFAULT_CONFIG_DIR, "Config", sub_dirs=[])
    install_dirs(log_dir, DEFAULT_LOG_DIR, "Log", sub_dirs=[])
    # check models
    models = args.models
    if not models:
        logger.info(f"No models specified, using default models {DEFAULT_MODELS}")
        if interactive:
            x = input(f"Download default models? [Y/n]... ")
            if x.casefold() == "n":
                logger.info("Skipping model download...")
                models = []
            else:
                models = ["all"]
        else:
            logger.info("Skipping download of default models...")

    if models:
        for model in models:
            model = model.strip().casefold()
            if model not in available_models.keys():
                logger.error(
                    f"Invalid model '{model}' -  Allowed models: {', '.join(available_models.keys())}"
                )
            else:
                logger.info(f"Downloading model data: {model}")
                model_data = available_models[model]
                model_folder = model_dir / model_data["folder"]
                create_dir(model_folder, web_user, web_group, create_mode)
                _model = model_data["model"]
                _config = model_data["config"]
                if _model:
                    for model_url in model_data["model"]:
                        model_file = model_folder / Path(model_url).name
                        if model_file.exists():
                            if not force_models:
                                logger.warning(
                                    f"Model file '{model_file}' already exists, skipping..."
                                )
                                continue
                            else:
                                logger.info(
                                    f"--force-model passed via CLI - Model file '{model_file}' already exists, overwriting..."
                                )
                        download_file(
                            model_url, model_file, web_user, web_group, create_mode
                        )
                if _config:
                    for config_url in model_data["config"]:
                        config_file = model_folder / Path(config_url).name
                        if config_file.exists():
                            if not force_models:
                                logger.warning(
                                    f"Config file '{config_file}' already exists, skipping..."
                                )
                                continue
                            else:
                                logger.info(
                                    f"--force-model passed via CLI - Config file '{config_file}' already exists, overwriting..."
                                )
                        download_file(
                            config_url, config_file, web_user, web_group, create_mode
                        )

    def do_install(_inst_type: str, search_dir: Path):
        existing_secrets = False
        path_glob_out = search_dir.glob(f"{_inst_type}.*")
        for iter_item in path_glob_out:
            logger.debug(f"{_inst_type} cfg generator iterated in a for loop: {iter_item = }")
        path_glob_out = sorted(path_glob_out, reverse=True)
        if path_glob_out:
            backup_num = 1
            if path_glob_out[0].suffix == ".bak":
                backup_num = int(path_glob_out[0].stem.split(".")[-1]) + 1
            file_backup = cfg_dir / f"{_inst_type}.{backup_num}.bak"
            for _file in path_glob_out:
                if _file.name == f"{_inst_type}.yml":
                    if _inst_type == "secrets":
                        existing_secrets = True
                        if args.install_secrets:
                            existing_secrets = False
                            logger.info(
                                "--overwrite-secrets was passed on CLI, overwriting existing secrets file..."
                            )
                        if not existing_secrets:
                            # copy example config
                            copy_file(
                                install_file_dir.parent / "configs/example_secrets.yml",
                                cfg_dir / "secrets.yml",
                                web_user,
                                web_group,
                                create_mode,
                            )
                    else:
                        copy_file(_file, file_backup, web_user, web_group, create_mode)
                    break
            if _inst_type != "secrets":
                copy_file(
                    install_file_dir.parent / f"configs/example_{_inst_type}.yml",
                    cfg_dir / f"{_inst_type}.yml",
                    web_user,
                    web_group,
                    create_mode,
                )
        if _inst_type != "secrets":
            install_host_dependencies(_inst_type)
            _extra = f"[{_inst_type}]"
            _src = install_file_dir.parent
            _pip_prefix = "pip3 install"
            _pip_opts = f"--report './pip_install_report_{datetime.datetime.now()}.json'"
            import subprocess
            if not args.test:
                logger.info(f"Installing {_inst_type} pip dependencies...")
                _msg = subprocess.check_output(
                    f"{_pip_prefix} {_pip_opts} {_src}{_extra}", shell=True
                )
                for _line in _msg.decode("utf-8").splitlines():
                    logger.info(_line)
            else:
                test_msg(f"Installing {_inst_type} pip dependencies...")
                _pip_opts = f"{_pip_opts} --dry-run"
                _msg = subprocess.check_output(
                    f"{_pip_prefix} {_pip_opts} {_src}{_extra}", shell=True
                )
                for _line in _msg.decode("utf-8").splitlines():
                    logger.info(_line)

    install_server = False
    install_client = False
    _install_type = args.install_type.strip().casefold()
    if _install_type:
        if _install_type == "server":
            install_server = True
        elif _install_type == "client":
            install_client = True
        elif _install_type == "both":
            install_server = True
            install_client = True
    else:
        logger.info("No install type specified, using 'client' as default...")
        install_client = True
    do_install("secrets", cfg_dir)
    if install_server:
        do_install("server", cfg_dir)
    if install_client:
        do_install("client", cfg_dir)


if __name__ == "__main__":
    install_file_dir = Path(__file__).parent
    args = parse_cli()
    force_models: bool = args.force_models
    create_mode = args.create_mode
    interactive = args.interactive
    install_log = args.install_log
    file_handler = logging.FileHandler(install_log, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    testing = args.test
    debug = args.debug
    data_dir = args.data_dir
    cfg_dir = args.config_dir
    log_dir = args.log_dir
    models: List[str] = args.models
    model_dir: Optional[Path] = None
    web_user, web_group = "", ""
    main()
