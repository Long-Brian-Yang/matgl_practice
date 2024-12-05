import json
import copy

def generate_additional_data(initial_data, num_additional=9):
    structures = initial_data['structures']
    labels = initial_data['labels']
    
    for i in range(num_additional):
        new_struct = copy.deepcopy(structures[0])
        
        # Adjust lattice parameters
        for j in range(3):
            new_struct['lattice'][j][j] += 0.001 * (i+1)  # Increase by 0.001, 0.002, ..., 0.009
        
        # Adjust coordinates
        new_coords = []
        for coord in new_struct['coords']:
            new_coord = [c + 0.001 * (i+1) for c in coord]
            new_coords.append(new_coord)
        new_struct['coords'] = new_coords
        
        structures.append(new_struct)
        
        # Adjust energy
        new_energy = labels['energies'][0] - 0.005 * (i+1)  # Decrease energy value
        labels['energies'].append(new_energy)
        
        # Adjust stress
        new_stress = copy.deepcopy(labels['stresses'][0])
        for k in range(len(new_stress)):
            new_stress[k] += 0.00001 * (i+1)
        labels['stresses'].append(new_stress)
        
        # Adjust force vectors
        new_force = copy.deepcopy(labels['forces'][0])
        for atom_force in new_force:
            for l in range(len(atom_force)):
                atom_force[l] += 0.001 * (i+1)
        labels['forces'].append(new_force)
    
    return initial_data

if __name__ == "__main__":
    # Read initial JSON file
    with open('test_dataset.json', 'r') as f:
        initial_data = json.load(f)
    
    # Generate additional data points
    expanded_data = generate_additional_data(initial_data, num_additional=9)
    
    # Save as a new JSON file
    with open('expanded_dataset.json', 'w') as f:
        json.dump(expanded_data, f, indent=2)
    
    print("Successfully generated a JSON file with 10 data points: expanded_dataset.json")