from __future__ import annotations

import warnings

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

import matgl
from matgl.ext.ase import PESCalculator, MolecularDynamics, Relaxer

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

# Load the M3GNet model for potential energy surface (PES) prediction
pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")

# Structure relaxation
relaxer = Relaxer(potential=pot)
struct = Structure.from_spacegroup("Pm-3m", Lattice.cubic(4.5), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
relax_results = relaxer.relax(struct, fmax=0.01)
# extract results
final_structure = relax_results["final_structure"]
final_energy = relax_results["trajectory"].energies[-1]
# print out the final relaxed structure and energy

print(final_structure)
print(f"The final energy is {float(final_energy):.3f} eV.")