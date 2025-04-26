# handlers.py

from abc import ABC, abstractmethod
from typing import Dict
from enum import Enum

class ResultHandler(ABC):
    """
    Abstract base class for all result handlers.
    Implement the handle() method to define custom handling logic.
    """
    @abstractmethod
    def handle(self, results: Dict[str, Enum]) -> None:
        """
        Process the categorization results.

        Args:
            results (Dict[str, Enum]): A dictionary mapping text (or IDs) to categories.
        """
        pass


class NoOpResultHandler(ResultHandler):
    """
    A result handler that does nothing.
    Useful as a default when no other handler is provided.
    """
    def handle(self, results: Dict[str, Enum]) -> None:
        pass


class PrintResultHandler(ResultHandler):
    """
    A simple handler that prints results to the console.
    Useful for debugging or local tests.
    """
    def handle(self, results: Dict[str, Enum]) -> None:
        for key, value in results.items():
            print(f"Text ID: {key}, Category: {value.name}")


class SaveToFileResultHandler(ResultHandler):
    """
    A handler that saves the results to a local file.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    def handle(self, results: Dict[str, Enum]) -> None:
        with open(self.file_path, "a", encoding="utf-8") as f:
            for key, value in results.items():
                f.write(f"{key},{value.name}\n")


# You can add more handlers here as needed:
# - ElasticSearchResultHandler
# - DatabaseResultHandler
# - KafkaResultHandler
# - NATSResultHandler
# etc.
