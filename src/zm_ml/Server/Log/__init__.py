import logging


SERVER_LOGGER_NAME = "ZM MLAPI"
SERVER_LOG_FORMAT = logging.Formatter(
    "%(asctime)s.%(msecs)04d %(name)s[%(process)s] %(levelname)s %(module)s:%(lineno)d -> %(message)s",
    "%m/%d/%y %H:%M:%S",
)
SERVER_LOG_FORMAT_S6 = logging.Formatter("%(name)s[%(process)s] %(levelname)s %(module)s:%(lineno)d -> %(message)s")
