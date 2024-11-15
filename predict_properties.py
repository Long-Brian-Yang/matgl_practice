from pymatgen.core import Lattice, Structure
import matgl

model = matgl.load_model("MEGNet-MP-2018.6.1-Eform")

# This is the structure obtained from the Materials Project.
struct = Structure.from_spacegroup("Pm-3m", Lattice.cubic(4.1437), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
eform = model.predict_structure(struct)
print(f"The predicted formation energy for CsCl is {float(eform.numpy()):.3f} eV/atom.")