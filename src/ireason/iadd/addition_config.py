# addition_config.py

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class HeldOutConfig:
    """
    Configuration for held-out digits.

    Attributes:
        held_out_digits (List[int]): Digits to hold out (0-9).
        positions (List[int]): Positions (from right, starting at 1) to hold out digits from.
    """

    held_out_digits: List[int]
    positions: List[int]

    def __post_init__(self):
        for digit in self.held_out_digits:
            if digit < 0 or digit > 9:
                raise ValueError("Held-out digits must be between 0 and 9.")
        for pos in self.positions:
            if pos <= 0:
                raise ValueError("Positions must be positive integers.")


@dataclass
class DatasetConfig:
    """
    Configuration for generating a dataset.

    Attributes:
        min_digits (int): Minimum number of digits in terms.
        max_digits (int): Maximum number of digits in terms.
        min_terms (int): Minimum number of terms in the addition.
        max_terms (int): Maximum number of terms in the addition.
        num_examples (Optional[int]): Number of examples to generate (ignored if generation_type is 'generate_all').
        generation_type (str): Type of generation ('generate_all' or 'random').
        fill_zeros (bool): Whether to fill numbers with leading zeros.
    """

    min_digits: int
    max_digits: int
    min_terms: int
    max_terms: int
    generation_type: str  # 'generate_all' or 'random'
    fill_zeros: bool
    num_examples: Optional[int] = (
        None  # Optional when generation_type is 'generate_all'
    )

    def __post_init__(self):
        if self.min_digits <= 0 or self.max_digits < self.min_digits:
            raise ValueError("Invalid digit constraints.")
        if self.min_terms < 2 or self.max_terms < self.min_terms:
            raise ValueError("Invalid term constraints.")
        if self.generation_type not in ["generate_all", "random"]:
            raise ValueError("'generation_type' must be 'generate_all' or 'random'.")
        if self.generation_type == "random":
            if self.num_examples is None or self.num_examples <= 0:
                raise ValueError(
                    "'num_examples' must be positive when 'generation_type' is 'random'."
                )
        else:
            self.num_examples = (
                None  # Ensure num_examples is None when generating all combinations
            )


@dataclass
class AdditionConfig:
    """
    Main configuration for the addition data generator.

    Attributes:
        random_seed (int): Seed for random number generation.
        training_config (DatasetConfig): Configuration for training data generation.
        test_config (DatasetConfig): Configuration for test data generation.
        held_out_config (HeldOutConfig): Configuration for held-out digits.
    """

    random_seed: int
    training_config: DatasetConfig
    test_config: DatasetConfig
    held_out_config: Optional[HeldOutConfig] = None

    def __post_init__(self):
        if self.random_seed is None:
            raise ValueError("Random seed must be specified.")
