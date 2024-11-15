import warnings
import matgl
from ase import units
from ase.build import add_adsorbate, fcc111, molecule
from ase.constraints import FixAtoms
from ase.md import Langevin
from ase.visualize import view
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from matgl.ext.ase import PESCalculator

warnings.simplefilter("ignore")

# Create an FCC (111) surface model
slab = fcc111('Pt', size=(3, 3, 4), vacuum=10.0)

# Load molecule
mol = molecule("CO")

# Position the molecule above the surface
add_adsorbate(slab=slab, adsorbate=mol, height=2.5, position="fcc")

from ase.constraints import FixAtoms

# Fix the lower half of the slab
mask = [atom.tag >= 3 for atom in slab]
slab.set_constraint(FixAtoms(mask=mask))

pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
slab.calc = PESCalculator(pot)

# Define the MD simulation parameters
temperature_K = 300  # Kelvin
timestep = 1 * units.fs  # Time step in femtoseconds
friction = 0.10 / units.fs  # Friction coefficient for Langevin dynamics

MaxwellBoltzmannDistribution(slab, temperature_K=temperature_K)

# Initialize the Langevin dynamics
dyn = Langevin(slab, timestep=timestep, temperature_K=temperature_K, friction=friction, trajectory="md.traj")

# Run the MD simulation
dyn.run(500)