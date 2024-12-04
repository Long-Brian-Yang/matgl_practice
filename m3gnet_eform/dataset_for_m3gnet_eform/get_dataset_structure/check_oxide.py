# check_oxide.py
import json
import re
from pymatgen.core import Structure

def is_oxide_ending_with_o(composition_dict, formula):
    """
    Check if a material is an oxide with oxygen at the end of its formula.
    
    Args:
        composition_dict (dict): Dictionary of element counts
        formula (str): Chemical formula of the material
        
    Returns:
        bool: True if the material is an oxide ending with O
    """
    # Check if oxygen is present
    if 'O' not in composition_dict:
        return False
    
    # Check if formula ends with O (ignoring numbers)
    formula_clean = re.sub(r'\d+$', '', formula)  # Remove trailing numbers
    return formula_clean.endswith('O')

def find_o_ending_oxides(input_file):
    """
    Find all oxide materials with oxygen at the end of their formula.
    
    Args:
        input_file (str): Path to input JSON file
        
    Returns:
        list: List of oxide materials information
    """
    print(f"Reading data from {input_file}")
    # Read the JSON file
    with open(input_file, 'r') as f:
        materials = json.load(f)
    
    oxide_materials = []
    print(f"Processing {len(materials)} materials...")
    
    for material in materials:
        try:
            # Convert structure string to Structure object
            structure = Structure.from_str(material["structure"], fmt="cif")
            composition = structure.composition
            composition_dict = composition.as_dict()
            formula = composition.reduced_formula
            
            if is_oxide_ending_with_o(composition_dict, formula):
                material_info = {
                    'material_id': material['material_id'],
                    'formula': formula,
                    'elements': [str(el) for el in composition.elements],
                    'element_counts': composition_dict,
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
                oxide_materials.append(material_info)
                
        except Exception as e:
            print(f"Error processing material {material['material_id']}: {str(e)}")
            continue
    
    return oxide_materials

def main():
    # Find oxide materials
    oxide_materials = find_o_ending_oxides("mp.2018.6.1.json")
    
    # Print results
    if oxide_materials:
        print(f"\nFound {len(oxide_materials)} oxide materials ending with O:")
        for material in oxide_materials[:5]:  # Show first 5 as example
            print(f"\nMaterial ID: {material['material_id']}")
            print(f"Formula: {material['formula']}")
            print(f"Elements: {', '.join(material['elements'])}")
            print(f"Composition: {material['element_counts']}")
            print(f"Formation Energy: {material['formation_energy']:.4f} eV/atom")
            print(f"Band Gap: {material['band_gap']:.4f} eV")
            print("Lattice Parameters:")
            for param, value in material['lattice_parameters'].items():
                print(f"  {param}: {value:.4f}")
        if len(oxide_materials) > 5:
            print("\n... and more")
    else:
        print("No oxide materials ending with O found in the dataset.")
    
    # Save results to a new JSON file
    output_file = "o_ending_oxides.json"
    with open(output_file, 'w') as f:
        json.dump(oxide_materials, f, indent=2)
    
    if oxide_materials:
        print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()