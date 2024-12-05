import json
import numpy as np
from pymatgen.io.vasp import Poscar, Outcar, Vasprun
import os
import sys

def parse_vasp_files(poscar_path, outcar_path, vasprun_path, output_json_path):
    """
    Parse VASP output files and save the required information as a JSON file.

    Parameters:
    - poscar_path (str): Path to the POSCAR or CONTCAR file.
    - outcar_path (str): Path to the OUTCAR file.
    - vasprun_path (str): Path to the vasprun.xml file.
    - output_json_path (str): Path to the output JSON file.
    """
    
    # Check if files exist
    for file_path in [poscar_path, outcar_path, vasprun_path]:
        if not os.path.isfile(file_path):
            print(f"Error: File does not exist - {file_path}")
            sys.exit(1)
    
    # Read POSCAR or CONTCAR file
    try:
        poscar = Poscar.from_file(poscar_path)
        structure = poscar.structure
        lattice = structure.lattice.matrix.tolist()  # Convert lattice matrix to list
        species = [str(sp) for sp in structure.species]
        coords = structure.frac_coords.tolist()  # Fractional coordinates
    except Exception as e:
        print(f"Error: Unable to read POSCAR file - {e}")
        sys.exit(1)
    
    # Initialize data structure
    data = {
        "structures": [
            {
                "lattice": lattice,
                "species": species,
                "coords": coords
            }
        ],
        "labels": {
            "energies": [],
            "stresses": [],
            "forces": []
        }
    }
    
    # Read OUTCAR file to extract force information
    forces_extracted = False
    try:
        outcar = Outcar(outcar_path)
        if hasattr(outcar, 'ionic_steps') and outcar.ionic_steps:
            # Extract forces from the last ionic step
            last_step_outcar = outcar.ionic_steps[-1]
            if 'forces' in last_step_outcar and last_step_outcar['forces'] is not None:
                forces = last_step_outcar['forces']
                # Ensure forces are NumPy arrays or lists, and convert to list
                if isinstance(forces, np.ndarray):
                    forces = forces.tolist()
                elif isinstance(forces, list):
                    pass  # Already a list
                else:
                    print("Error: Incorrect format for forces.")
                    sys.exit(1)
                data["labels"]["forces"].append(forces)
                forces_extracted = True
            else:
                print("Warning: No force information found in OUTCAR file.")
        else:
            print("Warning: No ionic step information found in OUTCAR file. Attempting to extract from vasprun.xml.")
    except Exception as e:
        print(f"Error: Unable to read OUTCAR file - {e}")
        sys.exit(1)
    
    # Read vasprun.xml file to extract energy, stress, and (if needed) force information
    try:
        vasprun = Vasprun(vasprun_path, parse_potcar_file=False, parse_dos=False, parse_eigen=False)
        energy = vasprun.final_energy  # Final energy (eV)
        data["labels"]["energies"].append(energy)
        
        ionic_steps = vasprun.ionic_steps
        if ionic_steps:
            last_step_vasprun = ionic_steps[-1]
            
            # Extract stress tensor
            stress = last_step_vasprun["stress"]  # Stress tensor
            print(f"Stress tensor before conversion: {stress}")
            
            # Check the structure of the stress tensor and convert
            if isinstance(stress, list):
                try:
                    # Convert stress tensor to NumPy array for processing
                    stress_array = np.array(stress, dtype=float)
                    stress_gpa = stress_array * 0.1  # Convert kBar to GPa
                    stress_flat = stress_gpa.flatten().tolist()
                    data["labels"]["stresses"].append(stress_flat)
                except Exception as e:
                    print(f"Error: Issue processing stress tensor - {e}")
                    print(f"Stress tensor content: {stress}")
                    sys.exit(1)
            else:
                print("Error: Incorrect format for stress tensor.")
                sys.exit(1)
            
            # If forces were not extracted from OUTCAR, attempt to extract from vasprun.xml
            if not forces_extracted:
                if "forces" in last_step_vasprun and last_step_vasprun["forces"] is not None:
                    forces = last_step_vasprun["forces"]
                    # Ensure forces are NumPy arrays or lists, and convert to list
                    if isinstance(forces, np.ndarray):
                        forces = forces.tolist()
                    elif isinstance(forces, list):
                        pass  # Already a list
                    else:
                        print("Error: Incorrect format for forces.")
                        sys.exit(1)
                    data["labels"]["forces"].append(forces)
                    print(f"Success: Force information extracted from vasprun.xml.")
                else:
                    print("Warning: No force information found in vasprun.xml file.")
        else:
            print("Warning: No ionic step information found in vasprun.xml file.")
        
    except Exception as e:
        print(f"Error: Unable to read vasprun.xml file - {e}")
        sys.exit(1)
    
    # Save as JSON file
    try:
        with open(output_json_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Success: Data saved to {output_json_path}")
    except Exception as e:
        print(f"Error: Unable to write JSON file - {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Define file paths
    poscar_file = "POSCAR"          # or "CONTCAR"
    outcar_file = "OUTCAR"
    vasprun_file = "vasprun.xml"
    output_json = "test_dataset.json"

    # Call the parsing function
    parse_vasp_files(poscar_file, outcar_file, vasprun_file, output_json)