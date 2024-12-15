from abc import ABC, abstractmethod
from typing import Any


class DataGenerator(ABC):
    """
    Abstract base class for data generators.
    """

    @abstractmethod
    def generate_data(self, *args, **kwargs) -> Any:
        """
        Generates the dataset based on specified parameters.

        Args:
            **kwargs: Arbitrary keyword arguments to support flexible configurations
                      for various types of data generators.

        Returns:
            Any: The generated dataset. This could be a list, dictionary, or any other structure
                 depending on the subclass implementation.
        """
        pass

    @abstractmethod
    def apply_transformations(self, data: Any, **kwargs) -> Any:
        """
        Applies any user-defined transformations or processing functions to the dataset.

        Args:
            data (Any): The dataset to be transformed.
            **kwargs: Arbitrary keyword arguments to specify transformation logic.

        Returns:
            Any: The transformed dataset. This could be in the same format as the input data
                 or a different one, depending on the subclass implementation.
        """
        pass
