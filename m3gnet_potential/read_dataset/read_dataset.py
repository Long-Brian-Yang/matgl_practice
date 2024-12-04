import json
import numpy as np
from pymatgen.core import Structure
from mp_api.client import MPRester

def save_dataset_to_json(structures, energies, forces, stresses, filename="dataset.json"):
    """
    Save dataset information to JSON file
    """
    dataset = {
        "structures": [],
        "labels": {
            "energies": [],
            "forces": [],
            "stresses": []
        }
    }
    
    # Convert structures
    for i, struct in enumerate(structures):
        struct_dict = {
            "lattice": struct.lattice.matrix.tolist(),
            "species": [str(site.specie) for site in struct],
            "coords": [site.coords.tolist() for site in struct]
        }
        dataset["structures"].append(struct_dict)
        
        dataset["labels"]["energies"].append(float(energies[i]))
        dataset["labels"]["forces"].append(forces[i].tolist())
        dataset["labels"]["stresses"].append(stresses[i].tolist())
    
    # Save with indent for readability
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)

def load_dataset_from_json(filename="dataset.json"):
    """
    Load dataset from JSON file
    """
    with open(filename, 'r') as f:
        dataset = json.load(f)
    
    # Convert back to structures
    structures = []
    for struct_dict in dataset["structures"]:
        struct = Structure(
            lattice=struct_dict["lattice"],
            species=struct_dict["species"],
            coords=struct_dict["coords"]
        )
        structures.append(struct)
    
    # Convert back to numpy arrays
    energies = np.array(dataset["labels"]["energies"], dtype=np.float32)
    forces = [np.array(force, dtype=np.float32) for force in dataset["labels"]["forces"]]
    stresses = [np.array(stress, dtype=np.float32) for stress in dataset["labels"]["stresses"]]
    
    return structures, energies, forces, stresses


mpr = MPRester(api_key="kzum4sPsW7GCRwtOqgDIr3zhYrfpaguK")
entries = mpr.get_entries_in_chemsys(["Ba", "Zr", "O"])

structures = [e.structure for e in entries]
energies = np.array([e.energy for e in entries], dtype=np.float32)
forces = [np.zeros((len(e.structure), 3), dtype=np.float32) for e in entries]
stresses = [np.zeros((3, 3), dtype=np.float32) for _ in structures]

save_dataset_to_json(structures, energies, forces, stresses, "dataset.json")

loaded_structures, loaded_energies, loaded_forces, loaded_stresses = load_dataset_from_json("dataset.json")
print(f"Successfully saved and loaded {len(loaded_structures)} structures")