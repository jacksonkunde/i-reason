# main.py
import os
import yaml
import argparse
from .addition_data_generator import AdditionDataGenerator
from .addition_config import AdditionConfig, DatasetConfig, HeldOutConfig
from ..base import CSVExporter, JSONExporter, HuggingFaceExporter

EXPORTER_CLASSES = {
    "csv": CSVExporter,
    "json": JSONExporter,
    "huggingface": HuggingFaceExporter,
}


def load_config(config_path: str) -> AdditionConfig:
    # Load configuration from YAML file
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)

    # Create the configuration dataclasses
    training_config = DatasetConfig(**config_dict["training_config"])
    test_config = DatasetConfig(**config_dict["test_config"])

    # Check if 'held_out_config' is in the configuration
    if "held_out_config" in config_dict:
        held_out_config = HeldOutConfig(**config_dict["held_out_config"])
    else:
        held_out_config = None  # Set to None if not provided

    config = AdditionConfig(
        random_seed=config_dict["random_seed"],
        training_config=training_config,
        test_config=test_config,
        held_out_config=held_out_config,
    )
    return config


def load_exporter(exporter_format: str):
    exporter_class = EXPORTER_CLASSES.get(exporter_format)
    if not exporter_class:
        raise ValueError(f"Unsupported export format: {exporter_format}")


def main(config_path: str, exporter_format: str, output_directory: str):
    config = load_config(config_path)

    data_generator = AdditionDataGenerator(config)
    data = data_generator.generate_data()
    data = data_generator.apply_transformations(data)

    exporter_class = EXPORTER_CLASSES.get(exporter_format)
    if not exporter_class:
        raise ValueError(f"Unsupported export format: {exporter_format}")

    exporter = exporter_class()
    exporter.export(data["train_data"], os.path.join(output_directory, "train"))
    exporter.export(data["test_data"], os.path.join(output_directory, "test"))

    print(f"Data exported to {output_directory}")


def run():
    parser = argparse.ArgumentParser(description="Synthetic Addition Data Generator")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--exporter",
        type=str,
        default="csv",
        help="Type of exporter -- one of [hf (HuggingFace dataset), csv, json]",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data", help="Path to output directory"
    )

    args = parser.parse_args()
    main(
        config_path=args.config,
        exporter_format=args.exporter,
        output_directory=args.output_dir,
    )
