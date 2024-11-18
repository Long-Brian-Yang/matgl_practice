from __future__ import annotations

import warnings

import torch
from pymatgen.core import Lattice, Structure

import matgl

# To suppress warnings for clearer output
warnings.simplefilter("ignore")
model = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")

# For multi-fidelity models, we need to define graph label ("0": PBE, "1": GLLB-SC, "2": HSE, "3": SCAN)
for i, method in ((0, "PBE"), (1, "GLLB-SC"), (2, "HSE"), (3, "SCAN")):
    graph_attrs = torch.tensor([i])
    bandgap = model.predict_structure(structure=struct, state_attr=graph_attrs)
    print(f"The predicted {method} band gap for CsCl is {float(bandgap):.3f} eV.")
