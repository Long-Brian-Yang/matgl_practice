# check_abo3.py
import json
from pymatgen.core import Structure

def is_abo3_type(composition_dict):
    """
    Check if a composition fits ABO3 stoichiometry.
    
    Args:
        composition_dict (dict): Dictionary of element counts
        
    Returns:
        bool, tuple: (is_abo3, (A_element, B_element)) if true, (False, None) if not
    """
    # Convert all values to integers for reliable comparison
    total_atoms = sum(composition_dict.values())
    unit_formula = {k: v * 5 / total_atoms for k, v in composition_dict.items()}
    
    # Count number of each type of element
    counts = {}
    for element, count in unit_formula.items():
        rounded_count = round(count)
        if rounded_count not in counts:
            counts[rounded_count] = []
        counts[rounded_count].append(element)
    
    # Check for ABO3 pattern (1:1:3 ratio)
    if 1 in counts and 3 in counts:
        if len(counts[1]) == 2 and len(counts[3]) == 1:  # Two elements with count 1, one element with count 3
            oxygen = counts[3][0]
            if oxygen == 'O':  # Verify that the element with count 3 is oxygen
                A_element = counts[1][0]
                B_element = counts[1][1]
                return True, (A_element, B_element)
    
    return False, None

def find_abo3_materials(input_file):
    """
    Find all ABO3-type materials in the JSON file.
    
    Args:
        input_file (str): Path to input JSON file
        
    Returns:
        list: List of ABO3 materials information
    """
    print(f"Reading data from {input_file}")
    # Read the JSON file
    with open(input_file, 'r') as f:
        materials = json.load(f)
    
    abo3_materials = []
    print(f"Processing {len(materials)} materials...")
    
    for material in materials:
        try:
            # Convert structure string to Structure object
            structure = Structure.from_str(material["structure"], fmt="cif")
            composition = structure.composition
            composition_dict = composition.as_dict()
            
            is_abo3, elements = is_abo3_type(composition_dict)
            
            if is_abo3:
                material_info = {
                    'material_id': material['material_id'],
                    'formula': composition.reduced_formula,
                    'A_element': elements[0],
                    'B_element': elements[1],
                    'formation_energy': material['formation_energy_per_atom'],
                    'band_gap': material['band_gap'],
                    'lattice_parameters': {
                        'a': structure.lattice.a,
                        'b': structure.lattice.b,
                        'c': structure.lattice.c,
                        'alpha': structure.lattice.alpha,
                        'beta': structure.lattice.beta,
                        'gamma': structure.lattice.gamma,
                        'volume': structure.lattice.volume
                    }
                }
                abo3_materials.append(material_info)
                
        except Exception as e:
            print(f"Error processing material {material['material_id']}: {str(e)}")
            continue
    
    return abo3_materials

def main():
    # Find ABO3 materials
    abo3_materials = find_abo3_materials("mp.2018.6.1.json")
    
    # Print results
    if abo3_materials:
        print(f"\nFound {len(abo3_materials)} ABO3-type materials:")
        for material in abo3_materials[:5]:  # Show first 5 as example
            print(f"\nMaterial ID: {material['material_id']}")
            print(f"Formula: {material['formula']}")
            print(f"A-site element: {material['A_element']}")
            print(f"B-site element: {material['B_element']}")
            print(f"Formation Energy: {material['formation_energy']:.4f} eV/atom")
            print(f"Band Gap: {material['band_gap']:.4f} eV")
            print("Lattice Parameters:")
            for param, value in material['lattice_parameters'].items():
                print(f"  {param}: {value:.4f}")
        if len(abo3_materials) > 5:
            print("\n... and more")
    else:
        print("No ABO3-type materials found in the dataset.")
    
    # Save results to a new JSON file
    output_file = "abo3_materials.json"
    with open(output_file, 'w') as f:
        json.dump(abo3_materials, f, indent=2)
    
    if abo3_materials:
        print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()