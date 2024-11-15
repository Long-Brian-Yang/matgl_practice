from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
import matgl
from matgl.ext.ase import PESCalculator
import warnings
from ase import units

warnings.simplefilter("ignore")

# Make structure with pymatgen
struct = Structure.from_spacegroup("Pm-3m", Lattice.cubic(4.5), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])

# Convert to ASE atoms
atoms = AseAtomsAdaptor.get_atoms(struct)
atoms *= [3, 3, 3]

pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
atoms.calc = PESCalculator(pot)

energy = atoms.get_potential_energy()

print(f"Energy = {energy:5.3f} eV")