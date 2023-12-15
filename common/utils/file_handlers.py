from abc import ABC, abstractmethod
from pathlib import Path
from logging import Logger
from common.logger import SingletonLogger
import pickle
import joblib
import json
import time


class LocalFileHandlerUtils:
    @staticmethod
    def __extract_directory_from_file_path(file_path: str) -> Path:
        path_obj: Path = Path(file_path)
        return path_obj.parent

    @staticmethod
    def check_and_make_directory_from_file_path(file_path: str):
        directory: Path = LocalFileHandlerUtils.__extract_directory_from_file_path(file_path=file_path)
        directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def check_and_make_directory(directory: str):
        directory: Path = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)


class BaseWriteObjectToLocalPatient(ABC):
    @abstractmethod
    def _write_try_time(self, x: object, file_name: str, *args, **kwargs) -> bool:
        pass

    def write(self, x: object, file_name: str, num_tries: int = 1, wait_time: float = 0.0, *args, **kwargs) -> bool:
        for try_time in range(num_tries):
            status = self._write_try_time(x=x, file_name=file_name, *args, **kwargs)
            if status:
                return True
            if try_time + 1 < num_tries:
                time.sleep(wait_time)
        return False


class BaseReadObjectFromLocalPatient(ABC):
    @abstractmethod
    def _read_try_time(self, file_name: str, *args, **kwargs) -> object:
        pass

    def read(self, file_name: str, num_tries: int = 1, wait_time: float = 0.0, *args, **kwargs) -> object:
        for try_time in range(num_tries):
            x: object = self._read_try_time(file_name=file_name, *args, **kwargs)
            if x is not None:
                return x
            if try_time + 1 < num_tries:
                time.sleep(wait_time)
        return None


class PickleWriteObjectToLocalPatient(BaseWriteObjectToLocalPatient):
    def _write_try_time(self, x: object, file_name: str, *args, **kwargs) -> bool:
        try:
            LocalFileHandlerUtils.check_and_make_directory_from_file_path(file_path=file_name)
            with open(file_name, mode="wb") as file_obj:
                pickle.dump(x, file_obj)
            return True
        except:
            logger: Logger = SingletonLogger.get_instance()
            logger.exception("Exception while writing object to local file by Pickle")
            return False


class PickleReadObjectFromLocalPatient(BaseReadObjectFromLocalPatient):
    def _read_try_time(self, file_name: str, *args, **kwargs) -> object:
        try:
            with open(file_name, mode="rb") as file_obj:
                x: object = pickle.load(file_obj)
            return x
        except:
            logger: Logger = SingletonLogger.get_instance()
            logger.exception("Exception while reading object from local file by Pickle")
            return None


class JoblibWriteObjectToLocalPatient(BaseWriteObjectToLocalPatient):
    def _write_try_time(self, x: object, file_name: str, *args, **kwargs) -> bool:
        try:
            LocalFileHandlerUtils.check_and_make_directory_from_file_path(file_path=file_name)
            joblib.dump(x, filename=file_name)
            return True
        except:
            logger: Logger = SingletonLogger.get_instance()
            logger.exception("Exception while writing object to local file by Joblib")
            return False


class JoblibReadObjectFromLocalPatient(BaseReadObjectFromLocalPatient):
    def _read_try_time(self, file_name: str, *args, **kwargs) -> object:
        try:
            x: object = joblib.load(filename=file_name)
            return x
        except:
            logger: Logger = SingletonLogger.get_instance()
            logger.exception("Exception while reading object from local file by Pickle")
            return None


class JsonWriteObjectToLocalPatient(BaseWriteObjectToLocalPatient):
    def _write_try_time(self, x: object, file_name: str, *args, **kwargs) -> bool:
        try:
            LocalFileHandlerUtils.check_and_make_directory_from_file_path(file_path=file_name)
            with open(file_name, mode="w") as file_obj:
                json.dump(x, file_obj)
            return True
        except:
            logger: Logger = SingletonLogger.get_instance()
            logger.exception("Exception while writing object to local file by Json")
            return False


class JsonReadObjectFromLocalPatient(BaseReadObjectFromLocalPatient):
    def _read_try_time(self, file_name: str, *args, **kwargs) -> object:
        try:
            with open(file_name, mode="r") as file_obj:
                x: object = json.load(file_obj)
            return x
        except:
            logger: Logger = SingletonLogger.get_instance()
            logger.exception("Exception while reading object from local file by Json")
            return None

