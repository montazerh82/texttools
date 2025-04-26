from abc import ABC, abstractmethod
from enum import Enum

class BaseCategorizer(ABC):
    def __init__(self, categories: Enum):
        """
        categories: An Enum class where each member represents a category.
        """
        self.categories = categories

    @abstractmethod
    def categorize(self, text: str) -> Enum:
        """
        Categorize the input text.
        Must return one of the Enum members defined in self.categories.
        """
        pass

    def preprocess(self, text: str) -> str:
        """
        Optional: Preprocess text before categorization.
        """
        return text
