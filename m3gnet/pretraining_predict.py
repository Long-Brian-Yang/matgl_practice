import os
import json
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from ase.io import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Lattice, Structure
from matgl.ext.ase import PESCalculator, MolecularDynamics, Relaxer
from matgl.utils.training import PotentialLightningModule
from dataset import prepare_data
import matgl


def setup_logging(log_dir: str = "logs") -> None:
    """
    Set up logging configuration
    Args:
        log_dir: Directory for log files
    """
    Path(log_dir).mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/predict.log"),
            logging.StreamHandler()
        ]
    )


def load_dataset(dataset_path: str) -> tuple:
    """
    Load dataset and DFT energy values
    Args:
        dataset_path: Path to the dataset file
    Returns:
        structures: List of structures and corresponding DFT energy values
    """
    with open(dataset_path, "r") as f:
        data = json.load(f)

    structures = []
    dft_energies = data["labels"]["energies"]  # Read DFT energy values

    for idx, item in enumerate(data["structures"]):
        structures.append({
            'structure': Structure(
                lattice=item["lattice"],
                species=item["species"],
                coords=item["coords"]
            ),
            'dft_energy': dft_energies[idx]
        })
    return structures


def predict_properties(model_path: str, dataset_path: str, output_dir: str) -> str:
    """
    Predict properties using the model and compare with DFT results
    Args:
        model_path: Path to the model
        dataset_path: Path to the dataset
        output_dir: Output directory
    Returns:
        Path to the output file
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Loading dataset from {dataset_path}...")
        structures = load_dataset(dataset_path)
        logger.info(f"Loaded {len(structures)} structures from dataset.")

        logger.info("Loading pre-trained M3GNet model...")
        pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")

        logger.info("Preparing PES calculator...")
        calculator = PESCalculator(potential=pot)

        predictions = []
        for struct_data in structures:
            atoms = AseAtomsAdaptor().get_atoms(struct_data['structure'])
            atoms.calc = calculator
            predicted_energy = atoms.get_potential_energy()
            predicted_forces = atoms.get_forces()

            predictions.append({
                "dft_energy": struct_data['dft_energy'],
                "predicted_energy": float(predicted_energy),
                "predicted_forces": predicted_forces.tolist()
            })

        output_file = Path(output_dir) / "predictions.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(predictions, f, indent=4)

        logger.info(f"Predictions saved to {output_file}")
        return str(output_file)

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    Calculate R² and MAE
    Args:
        y_true: True values
        y_pred: Predicted values
    Returns:
        r2: R² value
        mae: Mean absolute error
    """
    mae = np.mean(np.abs(y_true - y_pred))
    mean_y = np.mean(y_true)
    ss_tot = np.sum((y_true - mean_y) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2, mae


def plot_predictions(predictions_file: str, output_dir: str) -> None:
    """
    Plot the comparison between predicted values and DFT results
    Args:
        predictions_file: Path to the predictions file
        output_dir: Output directory
    """
    with open(predictions_file, "r") as f:
        predictions = json.load(f)

    # Extract DFT and predicted energy values
    dft_energies = np.array([p["dft_energy"] for p in predictions])
    predicted_energies = np.array([p["predicted_energy"] for p in predictions])

    # Calculate R² and MAE
    r2, mae = calculate_metrics(dft_energies, predicted_energies)

    # Create the plot
    plt.figure(figsize=(10, 8))

    plt.scatter(dft_energies, predicted_energies, color=(150/255, 155/255, 199/255),
                alpha=0.5, s=50)

    # Add diagonal line
    min_val = min(dft_energies.min(), predicted_energies.min())
    max_val = max(dft_energies.max(), predicted_energies.max())
    plt.plot([min_val, max_val], [min_val, max_val],
             color='grey', linestyle='--', label='Perfect prediction')

    plt.xlabel('DFT Energy (eV)')
    plt.ylabel('Predicted Energy (eV)')
    plt.title('Pre-training M3GNet Prediction vs DFT')

    # Add R² and MAE information
    plt.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.3f} eV',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top')

    plt.tight_layout()

    output_plot_path = Path(output_dir) / "pretrained_prediction.png"
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Print statistics
    print("\nPrediction Statistics:")
    print(f"Energy: R² = {r2:.3f}, MAE = {mae:.3f} eV")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict properties with pre-trained M3GNet model")
    parser.add_argument("--dataset-path", type=str, default="./data/perovskite_dataset.json",
                        help="Path to dataset JSON file")
    parser.add_argument("--output-dir", type=str, default="./predictions_output",
                        help="Directory to save predictions and plots")

    args = parser.parse_args()

    try:
        predictions_file = predict_properties(None, args.dataset_path, args.output_dir)
        plot_predictions(predictions_file, args.output_dir)
    except Exception as e:
        logging.error(f"Program failed: {str(e)}")