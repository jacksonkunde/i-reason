# addition_data_generator.py

from .addition_config import DatasetConfig, HeldOutConfig
from ..base.data_generator import DataGenerator

from typing import Dict, Any, Tuple, Optional
from itertools import combinations_with_replacement
import random


class AdditionDataGenerator(DataGenerator):
    """
    Generates synthetic addition data with a simple and flexible interface.
    """

    def __init__(self):
        """Simplified initialization with no required config."""
        self.random_seed = None

    def generate_data(
        self,
        dataset_type: str = "train",
        random_seed: Optional[int] = None,
        min_digits: int = 1,
        max_digits: int = 3,
        min_terms: int = 2,
        max_terms: int = 2,
        num_examples: int = 1000,
        generation_type: str = "random",
        fill_zeros: bool = False,
        held_out_digits: Optional[list[int]] = None,
        held_out_positions: Optional[list[int]] = None,
    ) -> list[Dict[str, Any]]:
        """
        Generate a dataset for training, testing, or held-out configurations.

        Args:
            dataset_type (str): Type of dataset ("train", "test").
            random_seed (Optional[int]): Seed for random number generation.
            min_digits (int): Minimum number of digits in terms.
            max_digits (int): Maximum number of digits in terms.
            min_terms (int): Minimum number of terms in the addition.
            max_terms (int): Maximum number of terms in the addition.
            num_examples (int): Number of examples to generate.
            generation_type (str): Generation type ("random" or "generate_all").
            fill_zeros (bool): Whether to fill numbers with leading zeros.
            held_out_digits (Optional[list[int]]): Digits to hold out (0-9).
            held_out_positions (Optional[list[int]]): Positions to hold out digits from.

        Returns:
            list[Dict[str, Any]]: The generated dataset.
        """
        # Set random seed if provided
        if random_seed is not None:
            self.random_seed = random_seed
            random.seed(self.random_seed)

        # Create the dataset configuration
        dataset_config = DatasetConfig(
            min_digits=min_digits,
            max_digits=max_digits,
            min_terms=min_terms,
            max_terms=max_terms,
            num_examples=num_examples,
            generation_type=generation_type,
            fill_zeros=fill_zeros,
        )

        # Create the held-out configuration if applicable
        held_out_config = None
        if held_out_digits and held_out_positions:
            held_out_config = HeldOutConfig(
                held_out_digits=held_out_digits, positions=held_out_positions
            )

        # Generate the dataset based on type
        data = self._generate_dataset(
            dataset_config,
            dataset_type=dataset_type,
            held_out_config=held_out_config,
        )

        # Filter held-out examples if applicable
        if held_out_config and dataset_type == "train":
            data, _ = self._filter_held_out_examples(data, held_out_config)

        return self.apply_transformations(data)

    def _generate_dataset(
        self,
        dataset_config,
        dataset_type: str,
        held_out_config: Optional[HeldOutConfig] = None,
    ) -> list[Dict[str, Any]]:
        if dataset_config.generation_type == "generate_all":
            data = self._generate_all(
                dataset_config,
                dataset_type,
                held_out_config=held_out_config,
            )
        elif dataset_config.generation_type == "random":
            data = self._generate_random(
                dataset_config,
                dataset_type,
                held_out_config=held_out_config,
            )
        else:
            raise ValueError("Invalid generation type.")

        return data

    def _generate_all(
        self,
        dataset_config,
        dataset_type: str,
        held_out_config: Optional[HeldOutConfig],
    ) -> list[Dict[str, Any]]:
        data = []
        min_num = 10 ** (dataset_config.min_digits - 1)
        max_num = 10**dataset_config.max_digits - 1
        print(f"min: {min_num}, max: {max_num}")
        numbers = [str(num) for num in range(min_num, max_num + 1)]
        print()

        # Generate all possible combinations of numbers for each term count
        for num_terms in range(dataset_config.min_terms, dataset_config.max_terms + 1):
            term_combinations = combinations_with_replacement(numbers, num_terms)
            for terms in term_combinations:
                example = self._create_example(
                    list(terms),
                    dataset_config.fill_zeros,
                    dataset_type,
                    held_out_config,
                )
                data.append(example)

        return data

    def _generate_random(
        self,
        dataset_config,
        dataset_type: str,
        held_out_config: Optional[HeldOutConfig] = None,
    ) -> list[Dict[str, Any]]:
        data = []
        for _ in range(dataset_config.num_examples):
            num_terms = random.randint(
                dataset_config.min_terms, dataset_config.max_terms
            )
            terms = []
            num_digits = random.randint(
                dataset_config.min_digits, dataset_config.max_digits
            )
            for _ in range(num_terms):
                term = self._generate_term(num_digits)
                terms.append(term)
            example = self._create_example(
                terms,
                dataset_config.fill_zeros,
                dataset_type,
                held_out_config=held_out_config,
            )
            data.append(example)
        return data

    def _generate_term(self, num_digits: int) -> str:
        min_value = 10 ** (num_digits - 1)
        max_value = (10**num_digits) - 1
        term = str(random.randint(min_value, max_value))
        return term

    def _create_example(
        self,
        terms: list[str],
        fill_zeros: bool,
        dataset_type: str,
        held_out_config: Optional[HeldOutConfig],
    ) -> Dict[str, Any]:
        # Apply fill zeros if specified
        fill_zeros_applied = False
        if fill_zeros:
            max_length = max(len(term) for term in terms)
            terms = [term.zfill(max_length) for term in terms]
            fill_zeros_applied = True

        # Check for held-out digits in specified positions
        if held_out_config is not None:
            contains_held_out = any(
                self._contains_held_out_digit(term, held_out_config=held_out_config)
                for term in terms
            )
        else:
            contains_held_out = False

        # Sum the terms
        total = sum(int(term) for term in terms)

        # Create question, answer, and text
        question = " + ".join(terms) + " = "
        answer = f"{total}"
        text = f"{question}{answer}"

        # Build metadata
        metadata = {
            "terms": terms,
            "sum": str(total),
            "question": question,
            "answer": answer,
            "text": text,
            "fill_zeros_applied": fill_zeros_applied,
            "contains_held_out": contains_held_out,
            "dataset_type": dataset_type,
        }

        return metadata

    def _contains_held_out_digit(
        self, term: str, held_out_config: HeldOutConfig
    ) -> bool:
        term_reversed = term[::-1]
        for pos in held_out_config.positions:
            if pos - 1 < len(term_reversed):
                digit = int(term_reversed[pos - 1])
                if digit in held_out_config.held_out_digits:
                    return True
        return False

    def _filter_held_out_examples(
        self, training_data: list[Dict[str, Any]]
    ) -> Tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
        training_data_clean = []
        moved_examples = []
        for example in training_data:
            if example["contains_held_out"]:
                # Move to test data
                example["dataset_type"] = "test"
                moved_examples.append(example)
            else:
                training_data_clean.append(example)
        return training_data_clean, moved_examples

    def apply_transformations(
        self, data: Dict[str, list[Dict[str, Any]]]
    ) -> Dict[str, list[Dict[str, Any]]]:
        return data
