from __future__ import annotations
import os
import warnings
import numpy as np
from mp_api.client import MPRester
from functools import partial
from dgl.data.utils import split_dataset

import matgl
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.data import MGLDataset, MGLDataLoader, collate_fn_pes
from matgl.config import DEFAULT_ELEMENTS


def prepare_data(batch_size=16):
    # Clear DGL cache
    os.system('rm -r ~/.dgl')

    # To suppress warnings for clearer output
    warnings.simplefilter("ignore")

    # Setup Materials Project API key
    mpr = MPRester(api_key="kzum4sPsW7GCRwtOqgDIr3zhYrfpaguK")
    entries = mpr.get_entries_in_chemsys(["Ba", "Zr", "O"])
    structures = [e.structure for e in entries]
    energies = [e.energy for e in entries]

    # Prepare forces and stresses
    forces = [np.zeros((len(e.structure), 3), dtype=np.float32) for e in entries]
    stresses = [np.zeros((3, 3), dtype=np.float32) for _ in structures]

    labels = {
        "energies": np.array(energies, dtype=np.float32),
        "forces": forces,
        "stresses": stresses
    }

    print(f"{len(structures)} downloaded from MP.")

    # Prepare dataset
    element_types = DEFAULT_ELEMENTS
    converter = Structure2Graph(element_types=element_types, cutoff=5.0)

    dataset = MGLDataset(
        threebody_cutoff=4.0,
        structures=structures,
        converter=converter,
        labels=labels,
        include_line_graph=True,
        save_cache=False
    )

    # Split dataset
    train_data, val_data, test_data = split_dataset(
        dataset,
        frac_list=[0.8, 0.1, 0.1],
        shuffle=True,
        random_state=42,
    )

    # Create data loaders
    my_collate_fn = partial(collate_fn_pes, include_line_graph=True)
    train_loader, val_loader, test_loader = MGLDataLoader(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        collate_fn=my_collate_fn,
        batch_size=batch_size,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader


def cleanup():
    # Clean up temporary files
    for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
        try:
            os.remove(fn)
        except FileNotFoundError:
            pass
