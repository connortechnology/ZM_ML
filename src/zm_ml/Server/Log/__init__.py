import logging


SERVER_LOGGER_NAME = "ZoMi:API"
SERVER_LOG_FORMAT = logging.Formatter(
    "'%(asctime)s.%(msecs)04d' %(name)s[%(process)s] %(levelname)s %(module)s:%(lineno)d -> %(message)s",
    "%m/%d/%y %H:%M:%S",
)