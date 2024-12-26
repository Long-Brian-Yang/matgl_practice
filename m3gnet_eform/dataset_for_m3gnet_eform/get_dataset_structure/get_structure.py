import json
from pymatgen.core import Structure


def extract_and_save_materials(input_json="mp.2018.6.1.json", output_json="extracted_materials_100.json", n_materials=100):
    """
    Extract material information and save to a new JSON file.

    Args:
        input_json (str): Input JSON file path
        output_json (str): Output JSON file path
        n_materials (int): Number of materials to extract
    """
    # Load the dataset
    with open(input_json, 'r') as f:
        data = json.load(f)

    # Select the first n materials
    selected_materials = data[:n_materials]

    # Initialize list for materials
    materials_data = []

    for material in selected_materials:
        try:
            # Get the structure string directly without parsing
            structure_str = material["structure"]

            # Create material entry with the original structure string
            material_entry = {
                "material_id": material["material_id"],
                "structure": structure_str,  # Keep the original CIF string
                "formation_energy_per_atom": material["formation_energy_per_atom"],
                "band_gap": material["band_gap"]
            }

            materials_data.append(material_entry)

        except Exception as e:
            print(f"Error processing material {material['material_id']}: {str(e)}")
            continue

    # Save to JSON file
    with open(output_json, 'w') as f:
        json.dump(materials_data, f, indent=2)

    print(f"Extracted {len(materials_data)} materials and saved to {output_json}")
    return materials_data


if __name__ == "__main__":
    # Extract materials and save to JSON
    materials = extract_and_save_materials()

    # Print summary
    print("\nExtracted Materials Summary:")
    for material in materials:
        print(f"\nMaterial ID: {material['material_id']}")
        print(f"Formation Energy: {material['formation_energy_per_atom']:.4f} eV/atom")
        print(f"Band Gap: {material['band_gap']:.4f} eV")
