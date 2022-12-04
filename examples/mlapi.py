"""A Script to start the MLAPI server."""
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s.%(msecs)04d %(name)s[%(process)s] %(levelname)s %(module)s:%(lineno)d->[%(message)s]",
    "%m/%d/%y %H:%M:%S",
)
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

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
