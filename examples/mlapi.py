#!/opt/zm_ml/data/venv/bin/python3
#!/usr/bin/env python3
# The installation script will put a shebang at the top pointing to the virtualenv
# If you move stuff, you may need to update the shebang manually

"""A Script to start the MLAPI server and do user management"""
from pathlib import Path

from zm_ml.Server.app import create_logs, MLAPI

logger = create_logs()


def _parse_cli():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-C", "--config", help="Path to configuration file")
    subparsers = parser.add_subparsers(help="sub-command help", dest="command")
    sub_user = subparsers.add_parser(
        "user",
        help="User management (Will not start the server)",
    )
    user_group = sub_user.add_mutually_exclusive_group(required=True)
    user_group.add_argument(
        "--create", help="Create a new user", action="store_true", dest="user_create"
    )
    user_group.add_argument(
        "--delete",
        help="Delete an existing user",
        action="store_true",
        dest="user_delete",
    )
    user_group.add_argument(
        "--update",
        help="Update an existing user",
        action="store_true",
        dest="user_update",
    )
    user_group.add_argument(
        "--get", help="Get an existing user", action="store_true", dest="user_get"
    )
    user_group.add_argument(
        "--disable",
        help="Disable an existing user",
        action="store_true",
        dest="user_disable",
    )
    user_group.add_argument(
        "--enable",
        help="Enable an existing user",
        action="store_true",
        dest="user_enable",
    )
    user_group.add_argument(
        "--verify",
        help="Verify an existing user",
        action="store_true",
        dest="user_verify",
    )
    user_group.add_argument(
        "--list", help="List all users", action="store_true", dest="user_list"
    )

    sub_user.add_argument(
        "--disabled", help="Disable the user during create/update", action="store_true"
    )
    sub_user.add_argument("--username", help="Username for the new user", nargs="?")
    sub_user.add_argument("--password", help="Password for the new user", nargs="?")

    ret = vars(parser.parse_args())
    logger.debug(f"\n\nCLI args: {ret}\n\n")
    return ret


if __name__ == "__main__":
    import sys

    script_name = sys.argv[0].split("/")[-1]
    lp = f"{script_name}::"
    args = _parse_cli()
    config_file = args.get("config")
    # {'config': '/etc/zm/server.yml', 'command': 'user', 'create': False, 'delete': False, 'update': False, 'get': True, 'disabled': False, 'username': 'baudneo', 'password': 'admin'}
    user_mode = None
    if args.get("command") == "user":
        if args.get("user_create"):
            user_mode = "--create"
        elif args.get("user_delete"):
            user_mode = "--delete"
        elif args.get("user_update"):
            user_mode = "--update"
        elif args.get("user_get"):
            user_mode = "--get"
        elif args.get("user_disable"):
            user_mode = "--disable"
        elif args.get("user_enable"):
            user_mode = "--enable"
        elif args.get("user_verify"):
            user_mode = "--verify"
        elif args.get("user_list"):
            user_mode = "--list"

    if user_mode:
        msg = f"{lp} User management mode (server will not start): {user_mode}"
        logger.info(msg)

        from zm_ml.Server.Models.config import parse_client_config_file
        from zm_ml.Server.auth import UserDB, ZoMiUser

        # parse the config file into a pydantic model (runs substitutions and imports secrets)
        cfg = parse_client_config_file(Path(config_file))
        # Create an instance of the user database and pass it the path to the db file
        db = UserDB(cfg.server.auth.db_file)

        disabled = args.get("disabled", False)
        username = args.get("username")
        password = args.get("password")

        def _check_pw():
            if not password:
                raise ValueError(
                    f"Must specify --password when in user management mode ({user_mode})"
                )

        def _check_uname():
            if not username:
                raise ValueError(
                    f"Must specify --username when in user management mode ({user_mode})"
                )

        def _check_pw_uname():
            _check_pw()
            _check_uname()

        if user_mode == "--create":
            _check_pw_uname()
            succ = db.create_user(username, password, disabled)
            logger.info(f"\n\nCreated User ({username}): {succ}")
        elif user_mode == "--verify":
            _check_pw_uname()
            user = db.authenticate_user(username, password, cfg=cfg, bypass=True)
            if isinstance(user, ZoMiUser):
                logger.info(
                    f"\n\nVerified: user={user.username} - {password=} - {user.disabled=}"
                )
            elif isinstance(user, bool):
                logger.info(f"\n\nDID NOT verify: user={username} - {password=}")
        elif user_mode == "--delete":
            _check_uname()
            succ = db.delete_user(username)
            logger.info(f"\n\nDeleted User ({username}): {succ}")
        elif user_mode == "--update":
            _check_pw_uname()
            succ = db.update_user(username, password, disabled)
            logger.info(f"\n\nUpdated User ({username}): {succ}")
        elif user_mode == "--get":
            _check_uname()
            user = db.get_user(username)
            if user:
                logger.info(f"\n\nRetrieved User: {user}")
            else:
                logger.info(f"\n\nUser ({username}) not found")
        elif user_mode == "--disable":
            _check_uname()
            succ = db.disable_user(username)
            logger.info(f"\n\nDisabled User ({username}): {succ}")
        elif user_mode == "--enable":
            _check_uname()
            succ = db.enable_user(username)
            logger.info(f"\n\nEnabled User ({username}): {succ}")
        elif user_mode == "--list":
            users = db.list_users()
            logger.info(f"\n\nUsers: {users}")

        exit(0)
    else:
        if not config_file:
            # check ENV
            logger.info(
                f"{lp} No config file specified, checking ENV 'ML_SERVER_CONF_FILE'"
            )
            import os

            if "ML_SERVER_CONF_FILE" in os.environ:
                config_file = os.getenv("ML_SERVER_CONF_FILE")
                logger.info(f"{lp} Found ENV ML_SERVER_CONF_FILE={config_file}")
            else:
                raise FileNotFoundError("No config file specified")
        logger.info(f"{lp} Initializing MLAPI server with config file '{config_file}'")
        server: MLAPI = MLAPI(config_file)

        server.start()
