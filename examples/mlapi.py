"""A Script to start the MLAPI server."""
import logging
import sys

logger = logging.getLogger("ZM_ML-Client")
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setFormatter(CLIENT_LOG_FORMAT)
buffered_log_handler = BufferedLogHandler()
buffered_log_handler.setFormatter(CLIENT_LOG_FORMAT)
logger.setLevel(logging.DEBUG)
logger.addHandler(console_handler)
logger.addHandler(buffered_log_handler)

if __name__ == "__main__":
    logger.info("Starting MLAPI server...")
    from argparse import ArgumentParser

    from zm_ml.Server.app import MLAPI

    parser = ArgumentParser()
    parser.add_argument("-C", "--config", help="Path to configuration ENV file", default="./prod.env")
    args = vars(parser.parse_args())
    logger.info(f"Creating MLAPI server with config file '{args['config']}'")
    server: MLAPI = MLAPI(args["config"])
    server.start_server()
