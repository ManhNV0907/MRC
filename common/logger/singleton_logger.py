import logging
from logging import Logger, DEBUG, Formatter, StreamHandler
from threading import Lock


class SingletonLogger:
    __LOCK: Lock = Lock()
    __NAME: str = '__DEFAULT_LOGGER_NAME__'
    __LOGGER: Logger = None

    @classmethod
    def get_instance(cls) -> Logger:
        cls.__LOCK.acquire()
        if cls.__LOGGER is None:
            cls.__LOGGER = logging.getLogger(name=cls.__NAME)
            cls.__LOGGER.setLevel(level=DEBUG)

            log_format: Formatter = Formatter('%(levelname)s   %(asctime)s   %(message)s')

            log_handler: StreamHandler = StreamHandler()
            log_handler.setLevel(level=DEBUG)
            log_handler.setFormatter(log_format)
            cls.__LOGGER.addHandler(log_handler)

        cls.__LOCK.release()
        return cls.__LOGGER

    @classmethod
    def set_logger_name(cls, name: str):
        cls.__LOCK.acquire()
        if cls.__NAME == '__DEFAULT_LOGGER_NAME__':
            cls.__NAME = name
        cls.__LOCK.release()
