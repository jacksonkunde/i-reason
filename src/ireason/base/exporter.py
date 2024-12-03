# exporter.py

from typing import List, Dict, Any
import csv
import json
import os

try:
    from datasets import Dataset
except ImportError:
    Dataset = None


class Exporter:
    """
    Base exporter class.
    """

    def export(self, data: List[Dict[str, Any]], output_directory: str):
        pass


class CSVExporter(Exporter):
    """
    Exports data to a CSV file.
    """

    def export(self, data: List[Dict[str, Any]], output_directory: str):
        os.makedirs(output_directory, exist_ok=True)
        output_file = os.path.join(output_directory, "data.csv")
        with open(output_file, mode="w", newline="") as csvfile:
            fieldnames = data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in data:
                writer.writerow(row)


class JSONExporter(Exporter):
    """
    Exports data to a JSON file.
    """

    def export(self, data: List[Dict[str, Any]], output_directory: str):
        os.makedirs(output_directory, exist_ok=True)
        output_file = os.path.join(output_directory, "data.json")
        with open(output_file, "w") as jsonfile:
            json.dump(data, jsonfile, indent=2)


class HuggingFaceExporter(Exporter):
    """
    Exports data as a HuggingFace Dataset.
    """

    def export(self, data: List[Dict[str, Any]], output_directory: str):
        if Dataset is None:
            raise ImportError(
                "Please install 'datasets' library to use HuggingFaceExporter."
            )
        os.makedirs(output_directory, exist_ok=True)
        dataset = Dataset.from_list(data)
        dataset.save_to_disk(output_directory)
