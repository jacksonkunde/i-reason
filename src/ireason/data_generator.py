# data_generator.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class DataGenerator(ABC):
    """
    Abstract base class for data generators.
    """

    @abstractmethod
    def generate_data(self) -> List[Dict[str, Any]]:
        """
        Generates the dataset based on specified parameters.
        """
        pass

    @abstractmethod
    def apply_transformations(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applies any user-defined formatting or processing functions.
        """
        pass
