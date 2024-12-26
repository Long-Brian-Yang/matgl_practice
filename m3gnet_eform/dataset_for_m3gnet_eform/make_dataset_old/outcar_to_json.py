import json
import numpy as np
from pymatgen.io.vasp import Outcar
from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.cif import CifWriter


def structure_to_cif_string(structure):
    """
    Convert pymatgen Structure to CIF format string.

    Args:
        structure (Structure): Pymatgen Structure object

    Returns:
        str: Structure in CIF format
    """
    cif = CifWriter(structure)
    cif_string = str(cif)
    return cif_string


def outcar_to_dataset_format(outcar_path='OUTCAR', vasprun_path='vasprun.xml', material_id=None):
    """
    Convert VASP OUTCAR file to dataset format.

    Args:
        outcar_path (str): Path to OUTCAR file
        vasprun_path (str): Path to vasprun.xml file
        material_id (str): Optional material ID

    Returns:
        dict: Material data in dataset format
    """
    # Read OUTCAR and vasprun.xml
    outcar = Outcar(outcar_path)
    vasprun = Vasprun(vasprun_path)

    # Get final structure
    final_structure = vasprun.final_structure

    # Get formation energy and band gap
    energy_per_atom = vasprun.final_energy / len(final_structure)
    band_gap = vasprun.eigenvalue_band_properties[0]

    # Get composition and formula
    composition = final_structure.composition
    formula = composition.reduced_formula

    # Get lattice parameters
    lattice = final_structure.lattice

    # Convert structure to CIF format
    structure_cif = structure_to_cif_string(final_structure)

    # Create dataset format
    material_data = {
        "material_id": material_id if material_id else "user_material",
        "chemical_formula": formula,
        "formation_energy_per_atom": float(energy_per_atom),
        "band_gap": float(band_gap),
        "lattice_parameters": {
            "a": float(lattice.a),
            "b": float(lattice.b),
            "c": float(lattice.c),
            "alpha": float(lattice.alpha),
            "beta": float(lattice.beta),
            "gamma": float(lattice.gamma),
            "volume": float(lattice.volume)
        },
        "structure": structure_cif  # Store structure as CIF string
    }

    return material_data


def save_to_json(material_data, output_file='dataset.json'):
    """
    Save material data to JSON file.

    Args:
        material_data (dict): Material data in dataset format
        output_file (str): Output JSON file path
    """
    with open(output_file, 'w') as f:
        json.dump([material_data], f, indent=2)


def main():
    try:
        # Convert OUTCAR to dataset format
        material_data = outcar_to_dataset_format(
            outcar_path='OUTCAR',
            vasprun_path='vasprun.xml',
            material_id='BaZrO3'
        )

        # Save to JSON file
        save_to_json(material_data)

        # Print summary
        print("\nConversion successful! Summary of converted data:")
        print(f"Material ID: {material_data['material_id']}")
        print(f"Formula: {material_data['chemical_formula']}")
        print(f"Formation Energy per Atom: {material_data['formation_energy_per_atom']:.4f} eV")
        print(f"Band Gap: {material_data['band_gap']:.4f} eV")
        print("\nLattice Parameters:")
        for param, value in material_data['lattice_parameters'].items():
            print(f"  {param}: {value:.4f}")
        print("\nStructure stored in CIF format")

    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        raise


if __name__ == "__main__":
    main()
