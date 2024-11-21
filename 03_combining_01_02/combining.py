from __future__ import annotations

import warnings

import torch
from pymatgen.core import Lattice, Structure
from pymatgen.ext.matproj import MPRester

import matgl
from matgl.ext.ase import Relaxer

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

sto = Structure.from_spacegroup(
    "Pm-3m", Lattice.cubic(4.5), ["Sr", "Ti", "O"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]
)
print(sto)

api_key = "kzum4sPsW7GCRwtOqgDIr3zhYrfpaguK"
mpr = MPRester(api_key)
doc = mpr.summary.search(material_ids=["mp-5229"])[0]
sto_dft = doc['structure']
sto_dft_bandgap = doc['band_gap']
sto_dft_forme = doc['formation_energy_per_atom']


# Relax the crystal structure
pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
relaxer = Relaxer(potential=pot)
relax_results = relaxer.relax(sto, fmax=0.01)
relaxed_sto = relax_results["final_structure"]
print(relaxed_sto)

print(sto_dft)

# Formation energy prediction
# Load the pre-trained MEGNet formation energy model.
model = matgl.load_model("M3GNet-MP-2018.6.1-Eform")
eform_sto = model.predict_structure(sto)
eform_relaxed_sto = model.predict_structure(relaxed_sto)

print(f"The predicted formation energy for the unrelaxed SrTiO3 is {float(eform_sto):.3f} eV/atom.")
print(f"The predicted formation energy for the relaxed SrTiO3 is {float(eform_relaxed_sto):.3f} eV/atom.")
print(f"The Materials Project formation energy for DFT-relaxed SrTiO3 is {sto_dft_forme:.3f} eV/atom.")

# Band gap prediction
model = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")

# For multi-fidelity models, we need to define graph label ("0": PBE, "1": GLLB-SC, "2": HSE, "3": SCAN)
for i, method in ((0, "PBE"), (1, "GLLB-SC"), (2, "HSE"), (3, "SCAN")):
    graph_attrs = torch.tensor([i])
    bandgap_sto = model.predict_structure(structure=sto, state_attr=graph_attrs)
    bandgap_relaxed_sto = model.predict_structure(structure=relaxed_sto, state_attr=graph_attrs)

    print(f"{method} band gap")
    print(f"\tUnrelaxed STO = {float(bandgap_sto):.2f} eV.")
    print(f"\tRelaxed STO = {float(bandgap_relaxed_sto):.2f} eV.")
print(f"The PBE band gap for STO from Materials Project is {sto_dft_bandgap:.2f} eV.")