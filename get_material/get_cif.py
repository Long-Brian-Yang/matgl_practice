from pymatgen.ext.matproj import MPRester

# Initialize the MP Rester
mpr = MPRester("API_Key") # Input API key
structure = mpr.get_structure_by_material_id("mp-1960") # Input the material ID
structure.to(filename="Li2O.cif")