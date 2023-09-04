#!/opt/zm_ml/data/venv/bin/python3
#!/usr/bin/env python3

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
    import sys

    script_name = sys.argv[0].split("/")[-1]
    lp = f"{script_name}::"
    args = _parse_cli()
    config_file = args.get("config")
    if not config_file:
        # check ENV
        logger.info(f"{lp} No config file specified, checking ENV 'ML_SERVER_CONF_FILE'")
        import os
        if "ML_SERVER_CONF_FILE" in os.environ:
            config_file = os.getenv("ML_SERVER_CONF_FILE")
            logger.info(f"{lp} Found ENV ML_SERVER_CONF_FILE={config_file}")
        else:
            raise FileNotFoundError("No config file specified")
    logger.info(f"{lp} Initializing MLAPI server with config file '{config_file}'")
    server: MLAPI = MLAPI(config_file)
    server.start()
