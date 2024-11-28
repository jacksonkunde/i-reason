# data_generator.py

from typing import List, Dict, Any
import random

from ireason.data_generator import DataGenerator
from ireason.iadd.addition_config import AdditionConfig


class AdditionDataGenerator(DataGenerator):
    """
    Concrete implementation of DataGenerator for generating addition problems.
    """

    def __init__(self, config: AdditionConfig):
        self.config = config
        self.random_seed = config.random_seed
        random.seed(self.random_seed)

        # Data Generation Configuration
        data_gen_config = config.data_generation
        self.num_examples = data_gen_config.num_examples
        self.num_terms_config = data_gen_config.num_terms
        self.digit_constraints = data_gen_config.digit_constraints
        self.fill_zeros_config = data_gen_config.fill_zeros

        # Test Set Configuration
        self.test_set_config = config.test_set
        self.validation_split = self.test_set_config.validation_split

        # Difficult Test Set Configuration
        self.difficult_test_set = self.test_set_config.difficult_test_set
        self.greater_than_k_digits = None
        self.hold_out_digits_config = None
        self.hold_out_digits = []
        self.hold_out_positions = []
        self.apply_hold_out = False
        self.include_in_test_set = False

        if self.difficult_test_set:
            self.greater_than_k_digits = self.difficult_test_set.greater_than_k_digits
            self.hold_out_digits_config = self.difficult_test_set.hold_out_digits
            if self.hold_out_digits_config:
                self.hold_out_digits = self.hold_out_digits_config.digits
                self.hold_out_positions = self.hold_out_digits_config.positions
                self.apply_hold_out = self.hold_out_digits_config.apply_to_training_set
                self.include_in_test_set = (
                    self.hold_out_digits_config.include_in_test_set
                )

        # Export Options
        self.export_options = config.export_options

    def generate_data(self) -> List[Dict[str, Any]]:
        data = []
        for _ in range(self.num_examples):
            example = self._generate_example()
            data.append(example)
        return data

    def _generate_example(self) -> Dict[str, Any]:
        # Determine the number of terms
        num_terms = random.randint(self.num_terms_config.min, self.num_terms_config.max)
        terms = []

        for _ in range(num_terms):
            # Determine the number of digits for the term
            digits = random.randint(
                self.digit_constraints.min_digits, self.digit_constraints.max_digits
            )
            term = self._generate_term(digits)
            terms.append(term)

        # Decide whether to apply zero padding
        if self.fill_zeros_config.apply_consistently:
            fill_zeros = self.fill_zeros_config.fill
        else:
            fill_probability = self.fill_zeros_config.probability
            fill_zeros = random.random() < fill_probability

        if fill_zeros:
            max_length = max(len(term.lstrip("0")) for term in terms)
            terms = [term.zfill(max_length) for term in terms]

        # Calculate the sum of the terms
        total = sum(int(term) for term in terms)

        # Generate question, answer, and text
        question = " + ".join(terms)
        answer = {total}
        text = f"{question} = {answer}"

        # Metadata
        metadata = {
            "num_terms": num_terms,
            "terms_digits": [len(term.lstrip("0")) for term in terms],
            "fill_zeros": fill_zeros,
            "hold_out_applied": self.apply_hold_out,
            "terms": terms,
            "sum": str(total),
            "question": question,
            "answer": answer,
            "text": text,
        }

        return metadata

    def _generate_term(self, digits: int) -> str:
        min_value = 10 ** (digits - 1)
        max_value = (10**digits) - 1

        while True:
            term = str(random.randint(min_value, max_value))
            if self.apply_hold_out and self._has_hold_out_digits(term):
                continue  # Reject term if it contains held-out digits in specified positions
            else:
                break
        return term

    def _has_hold_out_digits(self, term: str) -> bool:
        """
        Checks if the term contains any of the held-out digits in the specified positions.
        Positions are counted from the right (least significant digit).
        """
        term_reversed = term[::-1]
        for pos in self.hold_out_positions:
            if pos - 1 < len(term_reversed):
                digit = int(term_reversed[pos - 1])
                if digit in self.hold_out_digits:
                    return True
        return False

    def apply_transformations(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Placeholder for user-defined transformations
        # In this implementation, we do not apply any additional transformations
        return data
