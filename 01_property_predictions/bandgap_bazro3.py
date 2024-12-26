from __future__ import annotations

import warnings
import torch
from pymatgen.core import Lattice, Structure
import matgl

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

# Load the MEGNet model for Band Gap prediction
model = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")

# Define the lattice parameters for BaZrO3
# Assuming a lattice constant of 4.192 Å
a = 4.192  # lattice constant in angstroms
lattice = Lattice.cubic(a)

# Define the atomic positions for BaZrO3
species = ["Ba", "Zr", "O", "O", "O"]
coords = [
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
    [0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, 0.0, 0.5],
]

# Create the Structure object
struct = Structure(lattice, species, coords)

# Iterate over the different computational methods
methods = [(0, "PBE"), (1, "GLLB-SC"), (2, "HSE"), (3, "SCAN")]

# Predict the band gap for BaZrO3 using the MEGNet model
for i, method in methods:
    graph_attrs = torch.tensor([i])
    bandgap = model.predict_structure(structure=struct, state_attr=graph_attrs)
    print(f"The predicted {method} band gap for BaZrO₃ is {float(bandgap):.3f} eV.")
