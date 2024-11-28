# addition_config.py

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FillZerosConfig:
    """
    Configuration for zero padding in numbers.

    Attributes:
        apply_consistently (bool): Determines if zero padding is applied uniformly across the dataset.
            - If `True`, the padding behavior is consistent throughout the dataset.
            - If `False`, padding is applied based on a specified probability.

        fill (Optional[bool]): Specifies whether to apply padding when `apply_consistently` is `True`.
            - If `True`, numbers are padded with leading zeros to match the length of the longest term in each problem.
            - If `False`, padding is not applied.
            - Required if `apply_consistently` is `True`.

        probability (Optional[float]): The probability of applying padding when `apply_consistently` is `False`.
            - A float between 0 and 1 representing the chance of padding each addition problem.
            - Required if `apply_consistently` is `False`.

    Examples:
        **Consistent Padding**:

            apply_consistently: True
            fill: True

        **Random Padding**:

            apply_consistently: False
            probability: 0.5

    Notes:
        - When padding is applied, leading zeros are added so that all terms in the problem have the same length.
        - Leading zeros are not counted when determining the number of digits in a term.
    """

    apply_consistently: bool
    fill: Optional[bool] = None
    probability: Optional[float] = None

    def __post_init__(self):
        if self.apply_consistently:
            if self.fill is None:
                raise ValueError(
                    "When 'apply_consistently' is True, 'fill' must be specified."
                )
        else:
            if self.probability is None:
                raise ValueError(
                    "When 'apply_consistently' is False, 'probability' must be specified."
                )
            if not (0 <= self.probability <= 1):
                raise ValueError("'probability' must be between 0 and 1.")


@dataclass
class DigitConstraints:
    """
    Specifies the number of digits for each term in the addition problems.

    Attributes:
        min_digits (int): Minimum number of digits in a term (inclusive).
            - Must be a positive integer.

        max_digits (int): Maximum number of digits in a term (inclusive).
            - Must be greater than or equal to `min_digits`.

    Example:
        digit_constraints:
            min_digits: 1
            max_digits: 3
    """

    min_digits: int
    max_digits: int

    def __post_init__(self):
        if self.min_digits <= 0:
            raise ValueError("'min_digits' must be a positive integer.")
        if self.max_digits < self.min_digits:
            raise ValueError(
                "'max_digits' must be greater than or equal to 'min_digits'."
            )


@dataclass
class NumTerms:
    """
    Defines the number of terms (summands) in each addition problem.

    Attributes:
        min (int): Minimum number of terms (inclusive).
            - Must be an integer greater than or equal to 2.

        max (int): Maximum number of terms (inclusive).
            - Must be greater than or equal to `min`.

    Example:
        num_terms:
            min: 2
            max: 3
    """

    min: int
    max: int

    def __post_init__(self):
        if self.min < 2:
            raise ValueError("'min' must be at least 2.")
        if self.max < self.min:
            raise ValueError("'max' must be greater than or equal to 'min'.")


@dataclass
class HoldOutDigits:
    """
    Specifies digits to exclude from the training set and include in the test set at certain positions.

    Attributes:
        digits (List[int]): List of digits to hold out (integers between 0 and 9).

        positions (List[int]): List of positions where the digits are held out.
            - Positions are counted from the right (least significant digit).
            - Position `1` refers to the units place.

        apply_to_training_set (bool): Determines if the hold-out is applied to the training set.
            - If `True`, the specified digits at the specified positions are excluded from the training set.

        include_in_test_set (bool): Determines if the held-out digits are included in the test set.
            - If `True`, the test set will include problems that have the specified digits at the specified positions.

    Example:
        hold_out_digits:
            digits: [7, 8]
            positions: [1, 2]
            apply_to_training_set: True
            include_in_test_set: True

    Notes:
        - **Digits**: Must be integers between 0 and 9.
        - **Positions**: Must be positive integers.
    """

    digits: List[int]
    positions: List[int]
    apply_to_training_set: bool
    include_in_test_set: bool

    def __post_init__(self):
        for digit in self.digits:
            if not (0 <= digit <= 9):
                raise ValueError("Digits must be between 0 and 9.")
        for pos in self.positions:
            if pos <= 0:
                raise ValueError("Positions must be positive integers.")


@dataclass
class DifficultTestSet:
    """
    Options to create a test set that is more challenging than the training set.

    Attributes:
        greater_than_k_digits (int): Generates problems with terms having more than `k` digits.
            - Must be a positive integer.

        hold_out_digits (Optional[HoldOutDigits]): Configuration for holding out digits.
            - Specifies digits and positions to hold out from the training set and include in the test set.

    Example:
        difficult_test_set:
            greater_than_k_digits: 4
            hold_out_digits:
                digits: [7, 8]
                positions: [1, 2]
                apply_to_training_set: True
                include_in_test_set: True
    """

    greater_than_k_digits: int
    hold_out_digits: Optional[HoldOutDigits] = None

    def __post_init__(self):
        if self.greater_than_k_digits <= 0:
            raise ValueError("'greater_than_k_digits' must be a positive integer.")


@dataclass
class TestSetConfig:
    """
    Configuration for creating validation and difficult test sets.

    Attributes:
        validation_split (float): Fraction of the dataset to reserve as a validation set.
            - Must be a float between 0 and 1.

        difficult_test_set (Optional[DifficultTestSet]): Configuration for the more difficult test set.
            - Includes options for higher digit counts and digit hold-outs.

    Example:
        test_set:
            validation_split: 0.1
            difficult_test_set:
                greater_than_k_digits: 4
                hold_out_digits:
                    digits: [7, 8]
                    positions: [1, 2]
                    apply_to_training_set: True
                    include_in_test_set: True

    Notes:
        - A `validation_split` of 0.1 means 10% of the data is used for validation.
    """

    validation_split: float
    difficult_test_set: Optional[DifficultTestSet] = None

    def __post_init__(self):
        if not (0 <= self.validation_split <= 1):
            raise ValueError("'validation_split' must be between 0 and 1.")


@dataclass
class DataGenerationConfig:
    """
    Parameters related to the generation of addition problems.

    Attributes:
        num_examples (int): Total number of addition problems to generate.
            - Must be a positive integer.

        num_terms (NumTerms): Configuration for the number of terms in each problem.

        digit_constraints (DigitConstraints): Constraints on the number of digits in each term.

        fill_zeros (FillZerosConfig): Configuration for zero padding.

    Example:
        data_generation:
            num_examples: 10000
            num_terms:
                min: 2
                max: 3
            digit_constraints:
                min_digits: 1
                max_digits: 3
            fill_zeros:
                apply_consistently: False
                probability: 0.5
    """

    num_examples: int
    num_terms: NumTerms
    digit_constraints: DigitConstraints
    fill_zeros: FillZerosConfig

    def __post_init__(self):
        if self.num_examples <= 0:
            raise ValueError("'num_examples' must be a positive integer.")


@dataclass
class ExportOptions:
    """
    Specifies how and where to export the generated data.

    Attributes:
        format (str): The file format for exporting the data.
            - Allowed values: 'csv', 'json', 'huggingface'.

        output_directory (str): The directory where the exported data files will be saved.
            - Should be a valid directory path.

    Example:
        export_options:
            format: 'csv'
            output_directory: './output'

    Notes:
        - Ensure that the specified directory has write permissions.
    """

    format: str
    output_directory: str

    def __post_init__(self):
        allowed_formats = {"csv", "json", "huggingface"}
        if self.format not in allowed_formats:
            raise ValueError(f"'format' must be one of {allowed_formats}.")


@dataclass
class AdditionConfig:
    """
    Main configuration for the Synthetic Addition Data Generator.

    Attributes:
        random_seed (int): Sets the seed for all random number generation to ensure reproducibility.
            - If not specified, a default seed (e.g., 42) is used.

        data_generation (DataGenerationConfig): Configuration for data generation parameters.

        test_set (TestSetConfig): Configuration for test set creation.

        export_options (ExportOptions): Configuration for data export options.

    Example:
        random_seed: 42
        data_generation:
            num_examples: 10000
            num_terms:
                min: 2
                max: 3
            digit_constraints:
                min_digits: 1
                max_digits: 3
            fill_zeros:
                apply_consistently: False
                probability: 0.5
        test_set:
            validation_split: 0.1
            difficult_test_set:
                greater_than_k_digits: 4
                hold_out_digits:
                    digits: [7, 8]
                    positions: [1, 2]
                    apply_to_training_set: True
                    include_in_test_set: True
        export_options:
            format: 'csv'
            output_directory: './output'
    """

    random_seed: int
    data_generation: DataGenerationConfig
    test_set: TestSetConfig
    export_options: ExportOptions

    def __post_init__(self):
        pass  # Additional validation can be added here if needed
