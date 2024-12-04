from __future__ import annotations
import os
import shutil
import warnings
import json
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd
import lightning as pl
import torch
from dgl.data.utils import split_dataset
from pymatgen.core import Structure
from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm

from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_graph
from matgl.models import M3GNet
from matgl.utils.training import ModelLightningModule

warnings.simplefilter("ignore")

def validate_structures(structures: list[Structure]) -> set:
    """Validate structures and return all unique elements."""
    all_elements = set()
    invalid_structs = []
    
    for i, struct in enumerate(structures):
        try:
            elements = set(str(site.specie) for site in struct)
            all_elements.update(elements)
        except Exception as e:
            invalid_structs.append((i, str(e)))
    
    if invalid_structs:
        print("Warning: Found invalid structures:")
        for idx, error in invalid_structs:
            print(f"Structure {idx}: {error}")
            
    return all_elements

def load_dataset(json_path: str = "dataset.json") -> tuple[list[Structure], list[str], list[float]]:
    """Load data from a custom JSON file with additional validation."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find dataset file at {json_path}")
    
    data = pd.read_json(json_path)
    
    required_columns = ["material_id", "structure", "formation_energy_per_atom"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in JSON file: {missing_columns}")
    
    structures = []
    mp_ids = []
    energies = []
    
    print("Loading structures from JSON file...")
    for mid, structure_str, energy in tqdm(zip(data["material_id"], 
                                             data["structure"], 
                                             data["formation_energy_per_atom"])):
        try:
            struct = Structure.from_str(structure_str, fmt="cif")
            structures.append(struct)
            mp_ids.append(mid)
            energies.append(energy)
        except Exception as e:
            print(f"Warning: Could not parse structure for {mid}: {e}")
            continue
            
    return structures, mp_ids, energies

def verify_element_coverage(dataset_elements: set, model_elements: tuple) -> bool:
    """Verify that all dataset elements are covered by the model."""
    model_elements_set = set(model_elements)
    uncovered_elements = dataset_elements - model_elements_set
    
    if uncovered_elements:
        print(f"ERROR: Found elements in dataset not covered by model: {uncovered_elements}")
        print(f"Model elements: {model_elements}")
        print(f"Dataset elements: {dataset_elements}")
        return False
    return True

def main(json_path: str = "extracted_materials_100.json", num_samples: int = 100, 
         max_epochs: int = 10, batch_size: int = 1):
    """Main training function with additional validation."""
    structures, mp_ids, eform_per_atom = load_dataset(json_path)
    
    if num_samples is not None:
        structures = structures[:num_samples]
        eform_per_atom = eform_per_atom[:num_samples]
    
    print(f"Training with {len(structures)} structures")
    
    # Validate structures and get all unique elements
    dataset_elements = validate_structures(structures)
    print(f"Unique elements in dataset: {sorted(dataset_elements)}")
    
    # Get element types for model initialization
    elem_list = get_element_list(structures)
    print(f"Model element types: {elem_list}")
    
    # Verify element coverage
    if not verify_element_coverage(dataset_elements, elem_list):
        raise ValueError("Element coverage verification failed")
    
    # Setup graph converter with debug output
    converter = Structure2Graph(element_types=elem_list, cutoff=4.0)
    print(f"Initialized Structure2Graph with cutoff={converter.cutoff}")
    
    # Convert structures to graphs with additional checks
    print("Converting structures to graphs...")
    mp_dataset = MGLDataset(
        threebody_cutoff=4.0,
        structures=structures,
        converter=converter,
        labels={"eform": eform_per_atom},
        include_line_graph=True,
    )
    
    # Split dataset with size verification
    dataset_size = len(mp_dataset)
    print(f"Dataset size: {dataset_size}")
    
    if dataset_size < 3:
        print("Warning: Dataset too small for splitting, using same data for train/val/test")
        train_data = val_data = test_data = mp_dataset
    else:
        train_data, val_data, test_data = split_dataset(
            mp_dataset,
            frac_list=[0.8, 0.1, 0.1],
            shuffle=True,
            random_state=42,
        )
        print(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Setup data loaders with error checking
    try:
        my_collate_fn = partial(collate_fn_graph, include_line_graph=True)
        train_loader, val_loader, test_loader = MGLDataLoader(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            collate_fn=my_collate_fn,
            batch_size=batch_size,
            num_workers=0,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create data loaders: {str(e)}")
    
    # Initialize model with verification
    print("Initializing M3GNet model...")
    model = M3GNet(
        element_types=elem_list,
        is_intensive=True,
        readout_type="set2set",
    )
    
    # Setup training module and trainer
    lit_module = ModelLightningModule(model=model, include_line_graph=True)
    logger = CSVLogger("logs", name="M3GNet_training")
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Training with error handling
    try:
        trainer.fit(
            model=lit_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
    except Exception as e:
        print(f"Training failed: {str(e)}")
        # Add stack trace for debugging
        import traceback
        traceback.print_exc()
        raise
    
    # Plot training metrics if available
    try:
        plot_training_metrics(logger.version)
    except Exception as e:
        print(f"Failed to plot metrics: {str(e)}")
    
    # Cleanup
    cleanup_files()

def plot_training_metrics(version):
    """Plot training metrics from logs."""
    metrics_path = f"logs/M3GNet_training/version_{version}/metrics.csv"
    
    metrics = pd.read_csv(metrics_path)
    
    plt.figure(figsize=(10, 6))
    metrics["train_MAE"].dropna().plot(label='Training MAE')
    metrics["val_MAE"].dropna().plot(label='Validation MAE')
    
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('training_progress.png')
    plt.close()

def cleanup_files():
    """Clean up temporary files."""
    for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", 
               "state_attr.pt", "labels.json"):
        try:
            os.remove(fn)
        except FileNotFoundError:
            pass
    
    try:
        shutil.rmtree("logs")
    except FileNotFoundError:
        pass

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train M3GNet model on custom dataset')
    parser.add_argument('json_path', type=str, nargs='?', 
                       default="extracted_materials_100.json",
                       help='Path to the JSON dataset file (default: dataset.json)')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to use (default: all)')
    parser.add_argument('--max_epochs', type=int, default=20,
                       help='Maximum number of training epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for training (default: 1)')
    
    args = parser.parse_args()
    main(args.json_path, args.num_samples, args.max_epochs, args.batch_size)