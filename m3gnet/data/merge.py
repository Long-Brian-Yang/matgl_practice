import json
from typing import Dict, List, Union
import numpy as np


def merge_structure_datasets(dataset1: Dict, dataset2: Dict) -> Dict:
    """
    Merge two structure datasets with the same format

    Args:
        dataset1: The first dataset dictionary
        dataset2: The second dataset dictionary

    Returns:
        The merged dataset dictionary
    """
    # Validate the format of both datasets
    required_keys = {'structures', 'labels'}
    for dataset in [dataset1, dataset2]:
        if not all(key in dataset for key in required_keys):
            raise ValueError("Dataset is missing required keys: 'structures' or 'labels'")

    # Create a new merged dataset
    merged_dataset = {
        'structures': [],
        'labels': {
            'energies': [],
            'stresses': [],
            'forces': []
        }
    }

    # Merge structures
    merged_dataset['structures'] = dataset1['structures'] + dataset2['structures']

    # Merge arrays under labels
    for key in ['energies', 'stresses', 'forces']:
        if key in dataset1['labels'] and key in dataset2['labels']:
            merged_dataset['labels'][key] = (
                dataset1['labels'][key] + dataset2['labels'][key]
            )

    # Validate data consistency
    n_structures = len(merged_dataset['structures'])
    n_energies = len(merged_dataset['labels']['energies'])
    n_stresses = len(merged_dataset['labels']['stresses'])
    n_forces = len(merged_dataset['labels']['forces'])

    if not (n_structures == n_energies == n_stresses == n_forces):
        raise ValueError(
            f"Inconsistent merged data: structures={n_structures}, "
            f"energies={n_energies}, stresses={n_stresses}, forces={n_forces}"
        )

    return merged_dataset


def load_and_merge_datasets(file1_path: str, file2_path: str, output_path: str = None):
    """
    Load two datasets from files and merge them

    Args:
        file1_path: The file path of the first dataset
        file2_path: The file path of the second dataset
        output_path: The output file path (optional)

    Returns:
        The merged dataset dictionary
    """
    # Read datasets
    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
        dataset1 = json.load(f1)
        dataset2 = json.load(f2)

    # Merge datasets
    merged_dataset = merge_structure_datasets(dataset1, dataset2)

    # If an output path is specified, save the merged dataset
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(merged_dataset, f, indent=2)

    return merged_dataset


# Example usage
if __name__ == "__main__":
    # Assume there are two dataset files: dataset1.json and dataset2.json
    merged = load_and_merge_datasets(
        'dataset.json',
        'perovskite_dataset.json',
        'merged_dataset.json'
    )

    # Print information about the merged dataset
    print(f"Number of structures in the merged dataset: {len(merged['structures'])}")
    print(f"Number of energy data points in the merged dataset: {len(merged['labels']['energies'])}")
