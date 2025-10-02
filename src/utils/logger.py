import logging
import sys


class MainLogger:
    """
    Base logger class for consistent logging across the application. All classes
    will inherit from this class to utilize the same logging configuration.
    """
    def __init__(self, logger_name = __name__):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(stream_handler)