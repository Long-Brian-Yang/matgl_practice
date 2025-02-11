import json
import random


def generate_fabricated_data(num_samples=10):
    data = {
        "structures": [],
        "labels": {
            "energies": [],
            "stresses": [],
            "forces": []
        }
    }

    for _ in range(num_samples):
        # Generate lattice parameters
        lattice = [
            [4.192 + random.uniform(-0.005, 0.005), 0.0, 0.0],
            [0.0, 4.192 + random.uniform(-0.005, 0.005), 0.0],
            [0.0, 0.0, 4.192 + random.uniform(-0.005, 0.005)]
        ]

        # Slight variations in species and coordinates
        species = ["Ba", "Zr", "O", "O", "O"]
        coords = [
            [0.0 + random.uniform(-0.005, 0.005), 0.0 + random.uniform(-0.005, 0.005),
             0.0 + random.uniform(-0.005, 0.005)],
            [0.5 + random.uniform(-0.01, 0.01), 0.5 + random.uniform(-0.01, 0.01), 0.5 + random.uniform(-0.01, 0.01)],
            [0.5 + random.uniform(-0.01, 0.01), 0.5 + random.uniform(-0.01, 0.01), 0.0 + random.uniform(-0.01, 0.01)],
            [0.5 + random.uniform(-0.01, 0.01), 0.0 + random.uniform(-0.01, 0.01), 0.5 + random.uniform(-0.01, 0.01)],
            [0.0 + random.uniform(-0.01, 0.01), 0.5 + random.uniform(-0.01, 0.01), 0.5 + random.uniform(-0.01, 0.01)]
        ]

        # Generate energy
        energy = -41.7 + random.uniform(-0.001, 0.001)

        # Generate stress tensor (flattened to a one-dimensional list)
        stress = [
            -0.001828773 + random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            -0.001828773 + random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            random.uniform(-0.00001, 0.00001),
            -0.001828773 + random.uniform(-0.00001, 0.00001)
        ]

        # Generate force vectors
        forces = []
        for _ in range(5):
            force = [
                round(random.uniform(-0.05, 0.05), 8),
                round(random.uniform(-0.05, 0.05), 8),
                round(random.uniform(-0.05, 0.05), 8)
            ]
            forces.append(force)

        # Add to data structure
        structure = {
            "lattice": lattice,
            "species": species,
            "coords": coords
        }
        data["structures"].append(structure)
        data["labels"]["energies"].append(round(energy, 8))
        data["labels"]["stresses"].append(stress)
        data["labels"]["forces"].append(forces)

    return data


if __name__ == "__main__":
    num_samples = 10  # Adjust the number of samples as needed
    fabricated_data = generate_fabricated_data(num_samples=num_samples)

    # Save as JSON file
    with open("fabricated_test_dataset.json", "w") as f:
        json.dump(fabricated_data, f, indent=4)

    print(f"Successfully generated {num_samples} fabricated data samples, saved as 'fabricated_test_dataset.json'")
