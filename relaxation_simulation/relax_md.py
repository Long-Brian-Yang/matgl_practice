from __future__ import annotations

import warnings

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

import matgl
from matgl.ext.ase import PESCalculator, MolecularDynamics, Relaxer

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

# Load the M3GNet model for potential energy surface prediction
pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")

# Sturcture relaxation
relaxer = Relaxer(potential=pot)
struct = Structure.from_spacegroup("Pm-3m", Lattice.cubic(4.5), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
relax_results = relaxer.relax(struct, fmax=0.01)
# extract results
final_structure = relax_results["final_structure"]
final_energy = relax_results["trajectory"].energies[-1]
# print out the final relaxed structure and energy

print(final_structure)
print(f"The final energy is {float(final_energy):.3f} eV.")

# Molecular Dynamics simulation
ase_adaptor = AseAtomsAdaptor()
# Create ase atom object
atoms = ase_adaptor.get_atoms(final_structure)
# Initialize the velocity according to Maxwell Boltzamnn distribution
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
# Create the MD class
driver = MolecularDynamics(atoms, potential=pot, temperature=300, logfile="md_trial.log")
# Run
driver.run(100)
print(f"The potential energy of CsCl at 300 K after 100 steps is {float(atoms.get_potential_energy()):.3f} eV.")

# Single point calculation
# define the M3GNet calculator
calc = PESCalculator(pot)
# set up the calculator for atoms object
atoms.set_calculator(calc)
print(f"The calculated potential energy is {atoms.get_potential_energy():.3f} eV.")
