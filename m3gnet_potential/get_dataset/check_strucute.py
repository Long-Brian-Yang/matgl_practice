import json

def verify_dataset(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    num_structures = len(data['structures'])
    num_energies = len(data['labels']['energies'])
    num_stresses = len(data['labels']['stresses'])
    num_forces = len(data['labels']['forces'])

    print(f"Number of structures: {num_structures}")
    print(f"Number of energies: {num_energies}")
    print(f"Number of stresses: {num_stresses}")
    print(f"Number of forces: {num_forces}")

    assert num_structures == num_energies == num_stresses == num_forces, "The number of labels does not match the number of structures!"

    for i, force in enumerate(data['labels']['forces']):
        assert len(force) == 5 and all(len(atom_force) == 3 for atom_force in force), f"The format of force entry {i} is incorrect!"

    for i, stress in enumerate(data['labels']['stresses']):
        assert len(stress) == 9, f"The format of stress entry {i} is incorrect!"

    print("Dataset verification passed!")

verify_dataset('expanded_dataset.json')