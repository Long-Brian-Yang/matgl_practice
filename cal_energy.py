import warnings
import matgl
from ase.build import add_adsorbate, fcc111, molecule
from matgl.ext.ase import PESCalculator

warnings.simplefilter("ignore")

# Create an FCC (111) surface model
slab = fcc111('Pt', size=(3, 3, 4), vacuum=10.0)

# Load molecule
mol = molecule("CO")

# Position the molecule above the surface
add_adsorbate(slab=slab, adsorbate=mol, height=2.5, position="fcc")

pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
slab.calc = PESCalculator(pot)

energy = slab.get_potential_energy()
print(f"Energy = {energy:5.3f} eV")