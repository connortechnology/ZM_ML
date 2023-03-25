#!/usr/bin/python3
"""A Script to start the MLAPI server."""
from zm_ml.Server.app import create_logs, MLAPI

logger = create_logs()

def _parse_cli():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-C", "--config", help="Path to configuration file")
    ret = vars(parser.parse_args())
    return ret


if __name__ == "__main__":
    # get script name
    import sys

    script_name = sys.argv[0].split("/")[-1]
    lp = f"{script_name}::"
    logger.info(f"{lp} Starting MLAPI server...")
    args = _parse_cli()
    config_file = args.get("config")
    secrets_file = args.get("secrets")
    if not config_file:
        # check ENV
        logger.info(f"{lp} No config file specified, checking ENV 'ML_SERVER_CONF_FILE'")
        import os
        if "ML_SERVER_CONF_FILE" in os.environ:
            config_file = os.getenv("ML_SERVER_CONF_FILE")
            logger.info(f"{lp} Found ENV ML_SERVER_CONF_FILE={config_file}")
        else:
            raise FileNotFoundError("No config file specified")
    logger.info(f"{lp} Creating MLAPI server with config file '{config_file}'")
    server: MLAPI = MLAPI(config_file)
    logger.info(f"{lp} Starting MLAPI server NOW!")
    server.start_server()
