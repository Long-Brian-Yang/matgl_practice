from __future__ import annotations

import os
import shutil
import warnings

import numpy as np
import pytorch_lightning as pl
from functools import partial
from dgl.data.utils import split_dataset
#from mp_api.client import MPRester
from pytorch_lightning.loggers import CSVLogger

import matgl
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.ext.ase import PESCalculator, Atoms2Graph
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_pes
from matgl.models import M3GNet
from matgl.utils.training import PotentialLightningModule
from matgl.config import DEFAULT_ELEMENTS

from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor

from sklearn.linear_model import LinearRegression
from sklearn import metrics

import torch

from prettytable import PrettyTable

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

os.system('rm -r ~/.dgl')

# Function to parse data from XYZ files and extract structures, energies, forces, and stresses
def parse_data(filename, structure_type='ase'):
    """
    Reads a .xyz file and extracts structural, energy, force, and stress data.

    Args:
        filename (str): Path to the .xyz file.
        structure_type (str): 'ase' for ASE structures or 'pymatgen' for Pymatgen structures.

    Returns:
        structures: List of atomic structures.
        energies: List of energies per structure.
        forces: List of force arrays.
        stress: List of stress tensors.
    """
    data = read(filename, index=':')  # Read all structures from the XYZ file
    if structure_type == 'ase':
        structures = [atoms for atoms in data]
    else:
        structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in data]
    # Extracting energies, forces, and stresses (assumes these properties are included in the .xyz file)
    energies = [atoms.get_total_energy() for atoms in data]
    forces = [atoms.get_forces().tolist() for atoms in data]
    stress = [atoms.get_stress(voigt=False).tolist() for atoms in data]

    return structures, energies, forces, stress

# Function to compute elemental reference energies using linear regression
def compute_element_refs_dict(filename, elements):
    """
    Computes elemental reference energies by fitting energies to element counts using linear regression.

    Args:
        filename (str): Path to the training dataset.
        elements (list): List of elements to include.

    Returns:
        element_ref_dict (dict): Reference energy for each element.
    """
    # Parse training data
    structures, energies, forces, stress = parse_data(filename, structure_type='ase')
    element_encoder = np.zeros((len(structures), len(elements)))  # Matrix to encode element counts per structure

    # Populate the element count matrix
    for io, atoms in enumerate(structures):
        for jo, el in enumerate(elements):
            if el in atoms.get_chemical_symbols():
                element_encoder[io, jo] = len((np.array(atoms.get_chemical_symbols()) == el).nonzero()[0])

    # Perform linear regression to compute reference energies
    lin_reg = LinearRegression(fit_intercept=False)
    lin_reg.fit(element_encoder, energies)

    # Map each element to its computed reference energy
    element_refs_lin_reg = lin_reg.coef_
    element_ref_dict = dict(zip(elements, element_refs_lin_reg))

    return element_ref_dict

# Function to evaluate model performance on energy, force, and stress predictions
def eval_train(structures, energies, forces, stress):
    """
    Evaluates the trained model on energy, force, and stress predictions for a dataset.

    Args:
        structures: List of ASE structures.
        energies: List of ground-truth energies.
        forces: List of ground-truth forces.
        stress: List of ground-truth stresses.
    """
    structures_ase = structures  # Assuming input structures are already in ASE format

    # Lists to store ground-truth and model-predicted values
    energy_per_atom = []
    forces_flat = []
    stress_flat = []
    ml_energies = []
    ml_energies_ft = []
    ml_forces = []
    ml_forces_ft = []
    ml_stress = []
    ml_stress_ft = []

    # Loop through structures and compute properties
    for io, atoms in enumerate(structures_ase):
        # Compute ground-truth values
        energy_per_atom.append(energies[io] / len(atoms))
        for fat in forces[io]:
            for f in fat:
                forces_flat.append(f)
        for sat in stress[io]:
            for s in sat:
                stress_flat.append(s)

        # Predict properties using the pre-trained M3GNet model
        atoms.calc = m3gnet
        ml_energies.append(atoms.get_total_energy() / len(atoms))
        forces_atoms = atoms.get_forces().ravel()
        for f in forces_atoms:
            ml_forces.append(f)
        stress_atoms = atoms.get_stress(voigt=False).ravel()
        for s in stress_atoms:
            ml_stress.append(s)

        # Predict properties using the fine-tuned M3GNet model
        atoms.calc = m3gnet_ft
        ml_energies_ft.append(atoms.get_total_energy() / len(atoms))
        forces_atoms = atoms.get_forces().ravel()
        for f in forces_atoms:
            ml_forces_ft.append(f)
        stress_atoms = atoms.get_stress(voigt=False).ravel()
        for s in stress_atoms:
            ml_stress_ft.append(s)

    # Compute performance metrics (MAE and RMSE)
    rmse_energy = metrics.root_mean_squared_error(energy_per_atom, ml_energies)
    rmse_energy_ft = metrics.root_mean_squared_error(energy_per_atom, ml_energies_ft)

    mae_energy = metrics.mean_absolute_error(energy_per_atom, ml_energies)
    mae_energy_ft = metrics.mean_absolute_error(energy_per_atom, ml_energies_ft)

    rmse_forces = metrics.root_mean_squared_error(forces_flat, ml_forces)
    rmse_forces_ft = metrics.root_mean_squared_error(forces_flat, ml_forces_ft)

    mae_forces = metrics.mean_absolute_error(forces_flat, ml_forces)
    mae_forces_ft = metrics.mean_absolute_error(forces_flat, ml_forces_ft)

    rmse_stress = metrics.root_mean_squared_error(stress_flat, ml_stress)
    rmse_stress_ft = metrics.root_mean_squared_error(stress_flat, ml_stress_ft)

    mae_stress = metrics.mean_absolute_error(stress_flat, ml_stress)
    mae_stress_ft = metrics.mean_absolute_error(stress_flat, ml_stress_ft)

    # Create a PrettyTable to display results
    headers = ['Model', 'Energy MAE meV/atom', 'Energy RMSE meV/atom', 'Force MAE eV/Ang', 'Force RMSE eV/Ang', 'Stress MAE GPa', 'Stress RMSE GPa']
    m3gnet_metrics = ['M3GNET', mae_energy * 1000, rmse_energy * 1000, mae_forces, rmse_forces, mae_stress, rmse_stress]
    m3gnet_ft_metrics = ['M3GNET-FT', mae_energy_ft * 1000, rmse_energy_ft * 1000, mae_forces_ft, rmse_forces_ft, mae_stress_ft, rmse_stress_ft]

    model_metrics_table = [headers, m3gnet_metrics, m3gnet_ft_metrics]

    tab2 = PrettyTable(model_metrics_table[0])
    tab2.add_rows(model_metrics_table[1:])
    tab2.float_format = "7.4"
    print(tab2)

# Paths to training and test data
training_data_path = './train.xyz'
test_data_path = './test.xyz'
name = 'finetune_m3gnet'

# Define the list of elements to consider
jpca_elements = ['Cs', 'Pb', 'Rb', 'C', 'H', 'N', 'I', 'Br']

# Parse training and test data
train_structures, train_energies, train_forces, train_stress = parse_data(training_data_path)
test_structures, test_energies, test_forces, test_stress = parse_data(test_data_path)

# Create labels for training and test datasets
train_labels = {
    "energies": train_energies,
    "forces": train_forces,
    "stresses": train_stress,
}
test_labels = {
    "energies": test_energies,
    "forces": test_forces,
    "stresses": test_stress,
}

# Print information about the datasets
print(f"{len(train_structures)} training structures")
print(f"{len(test_structures)} test structures")

print('Considering following elements:')
print(jpca_elements)

# Compute elemental reference energies from the training dataset
element_ref_dict = compute_element_refs_dict(training_data_path, jpca_elements)

print('Elemental ref energies from dataset:')
print(element_ref_dict)

# Set up the graph converter for the dataset
element_types = DEFAULT_ELEMENTS  # Use default elements
converter = Structure2Graph(element_types=element_types, cutoff=5.0)

# Create training and test datasets
train_dataset = MGLDataset(
    threebody_cutoff=4.0, structures=[AseAtomsAdaptor.get_structure(atoms) for atoms in train_structures], converter=converter, labels=train_labels, include_line_graph=True, save_cache=False
)
test_dataset = MGLDataset(
    threebody_cutoff=4.0, structures=[AseAtomsAdaptor.get_structure(atoms) for atoms in test_structures], converter=converter, labels=test_labels, include_line_graph=True, save_cache=False
)

# Split training dataset into training and validation sets
train_data, val_data = split_dataset(
    train_dataset,
    frac_list=[0.9, 0.1],
    shuffle=True,
    random_state=42,
)

# Create data loaders for training, validation, and testing
my_collate_fn = partial(collate_fn_pes, include_line_graph=True)

train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_dataset,
    collate_fn=my_collate_fn,
    batch_size=16,
    num_workers=0,
)

# Load the pre-trained M3GNet model
m3gnet_nnp = matgl.load_model("M3GNet-MP-2021.2.8-DIRECT-PES")
model_pretrained = m3gnet_nnp.model

# Get the elemental reference energies and data standardization from the pre-trained model
element_refs_m3gnet = m3gnet_nnp.element_refs.property_offset
data_std = m3gnet_nnp.data_std
element_ref_dict_m3gnet = dict(zip(DEFAULT_ELEMENTS, element_refs_m3gnet))

# Create a list of element references for the fine-tuning process
element_refs = [element_ref_dict[el] if el in jpca_elements else element_ref_dict_m3gnet[el] for el in DEFAULT_ELEMENTS]

# Set up the fine-tuning module with the pre-trained model
lit_module_finetune = PotentialLightningModule(
    model=model_pretrained,
    lr=1e-3,
    include_line_graph=True,
    force_weight=1.0,
    stress_weight=0.1,
    element_refs=element_refs,
    data_std=data_std,
    decay_steps=100,
    decay_alpha=0.01
)

# If you wish to disable GPU or MPS (M1 mac) training, use the accelerator="cpu" kwarg.
logger = CSVLogger("logs", name="M3GNet_finetuning")
trainer = pl.Trainer(max_epochs=5, accelerator="cuda", logger=logger, inference_mode=False)
trainer.fit(model=lit_module_finetune, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Save the trained model
model_save_path = f"./{name}/"
lit_module_finetune.model.save(model_save_path)

# Load the fine-tuned model for evaluation
# pot_ft = matgl.load_model(model_save_path)
# m3gnet_ft = PESCalculator(pot_ft)
m3gnet_ft = PESCalculator(lit_module_finetune.model)  # Create PESCalculator using the fine-tuned model

# Load the pre-trained model for comparison
m3gnet = PESCalculator(matgl.load_model("M3GNet-MP-2021.2.8-DIRECT-PES"))

# Evaluate and print the training and test set results
print('TRAINING SET:')
eval_train(train_structures, train_energies, train_forces, train_stress)
print('TEST SET:')
eval_train(test_structures, test_energies, test_forces, test_stress)
