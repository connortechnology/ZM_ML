"""A Script to start the MLAPI server."""
import logging
import sys
from zm_ml.Server.app import SERVER_LOG_FORMAT

logger = logging.getLogger("ZM_ML-API")


if __name__ == "__main__":
    logger.info("Starting MLAPI server...")
    from argparse import ArgumentParser

    from zm_ml.Server.app import MLAPI

    parser = ArgumentParser()
    parser.add_argument("-C", "--config", help="Path to configuration file", default="/etc/zm/ml/server.yaml")
    args = vars(parser.parse_args())
    logger.info(f"Creating MLAPI server with config file '{args['config']}'")
    server: MLAPI = MLAPI(args["config"])
    server.start_server()
