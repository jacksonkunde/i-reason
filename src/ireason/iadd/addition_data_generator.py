# addition_data_generator.py

from .addition_config import AdditionConfig
from ..base.data_generator import DataGenerator

from typing import List, Dict, Any, Tuple
from itertools import combinations_with_replacement
import random


class AdditionDataGenerator(DataGenerator):
    """
    Generates synthetic addition data based on the provided configuration.
    """

    def __init__(self, config: AdditionConfig):
        self.config = config
        self.random_seed = config.random_seed
        random.seed(self.random_seed)
        self.equal_digit_distribution = config.equal_digit_distribution
        self.training_config = config.training_config
        self.test_config = config.test_config
        self.held_out_config = config.held_out_config

    def generate_data(self) -> Dict[str, List[Dict[str, Any]]]:
        # Generate training data
        training_data = self._generate_dataset(
            self.training_config, dataset_type="train"
        )

        if self.held_out_config is not None:
            training_data_clean, moved_examples = self._filter_held_out_examples(
                training_data
            )
        else:
            training_data_clean = training_data
            moved_examples = []

        # Generate test data
        test_data = self._generate_dataset(self.test_config, dataset_type="test")

        # Combine moved examples with test data
        test_data_combined = test_data + moved_examples

        return self.apply_transformations(
            {"train_data": training_data_clean, "test_data": test_data_combined}
        )

    def _generate_dataset(
        self, dataset_config, dataset_type: str
    ) -> List[Dict[str, Any]]:
        if dataset_config.generation_type == "generate_all":
            data = self._generate_all(dataset_config, dataset_type)
        elif dataset_config.generation_type == "random":
            data = self._generate_random(dataset_config, dataset_type)
        else:
            raise ValueError("Invalid generation type.")

        return data

    def _generate_all(self, dataset_config, dataset_type: str) -> List[Dict[str, Any]]:
        data = []
        min_num = 10 ** (dataset_config.min_digits - 1)
        max_num = 10**dataset_config.max_digits - 1
        print(f"min: {min_num}, max: {max_num}")
        numbers = [str(num) for num in range(min_num, max_num + 1)]

        # Generate all possible combinations of numbers for each term count
        for num_terms in range(dataset_config.min_terms, dataset_config.max_terms + 1):
            term_combinations = combinations_with_replacement(numbers, num_terms)
            for terms in term_combinations:
                example = self._create_example(
                    list(terms), dataset_config.fill_zeros, dataset_type
                )
                data.append(example)

        return data

    # def _generate_random(
    #     self, dataset_config, dataset_type: str
    # ) -> List[Dict[str, Any]]:
    #     data = []
    #     for _ in range(dataset_config.num_examples):
    #         num_terms = random.randint(
    #             dataset_config.min_terms, dataset_config.max_terms
    #         )
    #         terms = []
    #         for _ in range(num_terms):
    #             num_digits = random.randint(
    #                 dataset_config.min_digits, dataset_config.max_digits
    #             )
    #             term = self._generate_term(num_digits)
    #             terms.append(term)
    #         example = self._create_example(
    #             terms, dataset_config.fill_zeros, dataset_type
    #         )
    #         data.append(example)
    #     return data
    def _generate_random(
        self, dataset_config, dataset_type: str
    ) -> List[Dict[str, Any]]:
        data = []
        min_digits = dataset_config.min_digits
        max_digits = dataset_config.max_digits
        digit_range = range(min_digits, max_digits + 1)
        total_digit_options = len(digit_range)
        if self.equal_digit_distribution:
            # If we want equal distribution of digit lengths:
            total_examples = dataset_config.num_examples
            examples_per_digit = total_examples // total_digit_options
            leftover = total_examples % total_digit_options

            print("examples_per_digit", examples_per_digit)

            # Distribute leftover one by one to some digit lengths
            distribution = {d: examples_per_digit for d in digit_range}
            for d in list(digit_range)[:leftover]:
                distribution[d] += 1
            print(distribution)
            # For each digit length, generate the required number of examples
            for d in digit_range:
                count_needed = distribution[d]
                # Try to generate unique terms if possible
                # Number of unique possible numbers with d digits:
                # from 10^(d-1) to 10^d - 1, total of 9 * 10^(d-1) options.
                possible_count = 9 * (10 ** (d - 1))

                if count_needed > possible_count:
                    # If we need more examples than possible unique numbers,
                    # we generate all possible numbers for that digit length.
                    all_numbers = [str(x) for x in range(10 ** (d - 1), 10**d)]
                    # Shuffle and use them as needed
                    random.shuffle(all_numbers)
                    selected_numbers = all_numbers
                else:
                    # We can just sample unique numbers
                    # Sample without replacement
                    # For large possible_count, random.sample is possible
                    # If performance is an issue, consider a different approach.
                    selected_numbers = random.sample(
                        range(10 ** (d - 1), 10**d), count_needed
                    )
                    selected_numbers = [str(num) for num in selected_numbers]

                # Now, for each selected number, we still need to decide how many terms
                # and possibly generate additional terms. The specification isn't crystal
                # clear on whether all terms must have the same digit distribution or just
                # one term. For a strict interpretation, let's assume we only enforce
                # distribution on the first term, and the other terms are random.
                # If you need all terms to follow the distribution, apply similar logic
                # to each term.

                for base_term in selected_numbers:
                    num_terms = random.randint(
                        dataset_config.min_terms, dataset_config.max_terms
                    )
                    terms = [base_term]  # first term has d digits
                    # generate remaining terms randomly (same logic as before)
                    for _ in range(num_terms - 1):
                        # Other terms can have random digit lengths - or you can fix them
                        # to the same digit length d if desired.
                        # If you want all terms to have the same digit length distribution,
                        # just use 'd' instead of random digits.
                        if self.equal_digit_distribution:
                            other_term = self._generate_term(d)
                        else:
                            rand_d = random.randint(min_digits, max_digits)
                            other_term = self._generate_term(rand_d)
                        terms.append(other_term)

                    example = self._create_example(
                        terms, dataset_config.fill_zeros, dataset_type
                    )
                    data.append(example)

        else:
            # Original random generation logic (no equal distribution)
            for _ in range(dataset_config.num_examples):
                num_terms = random.randint(
                    dataset_config.min_terms, dataset_config.max_terms
                )
                terms = []
                for _ in range(num_terms):
                    num_digits = random.randint(
                        dataset_config.min_digits, dataset_config.max_digits
                    )
                    term = self._generate_term(num_digits)
                    terms.append(term)
                example = self._create_example(
                    terms, dataset_config.fill_zeros, dataset_type
                )
                data.append(example)

        return data

    def _generate_term(self, num_digits: int) -> str:
        min_value = 10 ** (num_digits - 1)
        max_value = (10**num_digits) - 1
        term = str(random.randint(min_value, max_value))
        return term

    def _create_example(
        self, terms: List[str], fill_zeros: bool, dataset_type: str
    ) -> Dict[str, Any]:
        # Apply fill zeros if specified
        fill_zeros_applied = False
        if fill_zeros:
            max_length = max(len(term) for term in terms)
            terms = [term.zfill(max_length) for term in terms]
            fill_zeros_applied = True

        # Check for held-out digits in specified positions
        if self.held_out_config is not None:
            contains_held_out = any(
                self._contains_held_out_digit(term) for term in terms
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

    def _contains_held_out_digit(self, term: str) -> bool:
        term_reversed = term[::-1]
        for pos in self.held_out_config.positions:
            if pos - 1 < len(term_reversed):
                digit = int(term_reversed[pos - 1])
                if digit in self.held_out_config.held_out_digits:
                    return True
        return False

    def _filter_held_out_examples(
        self, training_data: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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
        self, data: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        return data
