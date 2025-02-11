# import os
# import json
# import logging
# import matplotlib.pyplot as plt
# from pathlib import Path
# import numpy as np
# from ase.io import Trajectory
# from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
# from pymatgen.io.ase import AseAtomsAdaptor
# from pymatgen.core import Lattice, Structure
# from matgl.ext.ase import PESCalculator, MolecularDynamics, Relaxer
# from matgl.utils.training import PotentialLightningModule
# from dataset import prepare_data
# import matgl


# def setup_logging(log_dir: str = "logs") -> None:
#     """
#     Set up logging configuration
#     Args:
#         log_dir: Directory for log files
#     """
#     Path(log_dir).mkdir(exist_ok=True)
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(f"{log_dir}/predict.log"),
#             logging.StreamHandler()
#         ]
#     )


# def load_dataset(dataset_path: str) -> tuple:
#     """
#     Load dataset and DFT energy values
#     Args:
#         dataset_path: Path to the dataset file
#     Returns:
#         structures: List of structures and corresponding DFT energy values
#     """
#     with open(dataset_path, "r") as f:
#         data = json.load(f)

#     structures = []
#     dft_energies = data["labels"]["energies"]  # Read DFT energy values

#     for idx, item in enumerate(data["structures"]):
#         structures.append({
#             'structure': Structure(
#                 lattice=item["lattice"],
#                 species=item["species"],
#                 coords=item["coords"]
#             ),
#             'dft_energy': dft_energies[idx]
#         })
#     return structures


# def predict_properties(model_path: str, dataset_path: str, output_dir: str) -> str:
#     """
#     Predict properties using the model and compare with DFT results
#     Args:
#         model_path: Path to the model
#         dataset_path: Path to the dataset
#         output_dir: Output directory
#     Returns:
#         Path to the output file
#     """
#     setup_logging()
#     logger = logging.getLogger(__name__)

#     try:
#         logger.info(f"Loading dataset from {dataset_path}...")
#         structures = load_dataset(dataset_path)
#         logger.info(f"Loaded {len(structures)} structures from dataset.")

#         logger.info("Loading pre-trained M3GNet model...")
#         pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")

#         logger.info("Preparing PES calculator...")
#         calculator = PESCalculator(potential=pot)

#         predictions = []
#         for struct_data in structures:
#             atoms = AseAtomsAdaptor().get_atoms(struct_data['structure'])
#             atoms.calc = calculator
#             predicted_energy = atoms.get_potential_energy()
#             predicted_forces = atoms.get_forces()

#             predictions.append({
#                 "dft_energy": struct_data['dft_energy'],
#                 "predicted_energy": float(predicted_energy),
#                 "predicted_forces": predicted_forces.tolist()
#             })

#         output_file = Path(output_dir) / "predictions.json"
#         output_file.parent.mkdir(parents=True, exist_ok=True)
#         with open(output_file, "w") as f:
#             json.dump(predictions, f, indent=4)

#         logger.info(f"Predictions saved to {output_file}")
#         return str(output_file)

#     except Exception as e:
#         logger.error(f"Prediction failed: {str(e)}")
#         raise


# def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
#     """
#     Calculate R² and MAE
#     Args:
#         y_true: True values
#         y_pred: Predicted values
#     Returns:
#         r2: R² value
#         mae: Mean absolute error
#     """
#     mae = np.mean(np.abs(y_true - y_pred))
#     mean_y = np.mean(y_true)
#     ss_tot = np.sum((y_true - mean_y) ** 2)
#     ss_res = np.sum((y_true - y_pred) ** 2)
#     r2 = 1 - (ss_res / ss_tot)
#     return r2, mae


# def plot_predictions(predictions_file: str, output_dir: str) -> None:
#     """
#     Plot the comparison between predicted values and DFT results
#     Args:
#         predictions_file: Path to the predictions file
#         output_dir: Output directory
#     """
#     with open(predictions_file, "r") as f:
#         predictions = json.load(f)

#     # Extract DFT and predicted energy values
#     dft_energies = np.array([p["dft_energy"] for p in predictions])
#     predicted_energies = np.array([p["predicted_energy"] for p in predictions])

#     # Calculate R² and MAE
#     r2, mae = calculate_metrics(dft_energies, predicted_energies)

#     # Create the plot
#     plt.figure(figsize=(10, 8))

#     plt.scatter(dft_energies, predicted_energies, color=(150/255, 155/255, 199/255),
#                 alpha=0.5, s=50)

#     # Add diagonal line
#     min_val = min(dft_energies.min(), predicted_energies.min())
#     max_val = max(dft_energies.max(), predicted_energies.max())
#     plt.plot([min_val, max_val], [min_val, max_val],
#              color='grey', linestyle='--', label='Perfect prediction')

#     plt.xlabel('DFT Energy (eV)')
#     plt.ylabel('Predicted Energy (eV)')
#     plt.title('Pre-training M3GNet Prediction vs DFT')

#     # Add R² and MAE information
#     plt.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.3f} eV',
#              transform=plt.gca().transAxes,
#              bbox=dict(facecolor='white', alpha=0.8),
#              verticalalignment='top')

#     plt.tight_layout()

#     output_plot_path = Path(output_dir) / "pretrained_prediction.png"
#     plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
#     plt.show()

#     # Print statistics
#     print("\nPrediction Statistics:")
#     print(f"Energy: R² = {r2:.3f}, MAE = {mae:.3f} eV")


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Predict properties with pre-trained M3GNet model")
#     parser.add_argument("--dataset-path", type=str, default="./data/perovskite_dataset.json",
#                         help="Path to dataset JSON file")
#     parser.add_argument("--output-dir", type=str, default="./predictions_output",
#                         help="Directory to save predictions and plots")

#     args = parser.parse_args()

#     try:
#         predictions_file = predict_properties(None, args.dataset_path, args.output_dir)
#         plot_predictions(predictions_file, args.output_dir)
#     except Exception as e:
#         logging.error(f"Program failed: {str(e)}")
import os
import json
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from pymatgen.core import Structure
from matgl.ext.ase import PESCalculator
from pymatgen.io.ase import AseAtomsAdaptor
from scipy import stats
import matgl

# Set environment variable
os.environ['DGLBACKEND'] = 'pytorch'

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate various statistical metrics for model evaluation
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        dict: Dictionary containing various metrics
    """
    # Calculate basic metrics
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    
    # Calculate R² score
    mean_y = np.mean(y_true)
    ss_tot = np.sum((y_true - mean_y)**2)
    ss_res = np.sum((y_true - y_pred)**2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Calculate adjusted R² (taking into account the number of samples)
    n = len(y_true)
    p = 1  # number of predictors (in this case just one)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    # Calculate Pearson correlation coefficient
    pearson_r, p_value = stats.pearsonr(y_true, y_pred)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'adj_r2': adj_r2,
        'pearson_r': pearson_r,
        'p_value': p_value
    }

def predict_properties(dataset_path: str = "./data/perovskite_dataset.json", 
                      output_dir: str = "./predictions_output",
                      min_energy: float = -45,
                      max_energy: float = 0) -> str:
    """
    Predict properties using default paths
    
    Args:
        dataset_path: Path to dataset JSON file
        output_dir: Directory to save outputs
        min_energy: Minimum energy filter
        max_energy: Maximum energy filter
        
    Returns:
        str: Path to output predictions file
    """
    logger = logging.getLogger(__name__)
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Load dataset
        logger.info(f"Loading dataset from {dataset_path}...")
        with open(dataset_path, "r") as f:
            data = json.load(f)

        structures = []
        dft_energies = data["labels"]["energies"]

        # Filter structures by energy range
        for idx, (item, energy) in enumerate(zip(data["structures"], dft_energies)):
            if min_energy <= energy <= max_energy:
                structures.append({
                    'structure': Structure(
                        lattice=item["lattice"],
                        species=item["species"],
                        coords=item["coords"]
                    ),
                    'dft_energy': energy
                })

        logger.info(f"Loaded {len(structures)} structures within energy range [{min_energy}, {max_energy}] eV")

        # Load pretrained model
        logger.info("Loading pretrained M3GNet model...")
        pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        calculator = PESCalculator(potential=pot)

        # Make predictions
        predictions = []
        for i, struct_data in enumerate(structures, 1):
            logger.info(f"Processing structure {i}/{len(structures)}")
            atoms = AseAtomsAdaptor().get_atoms(struct_data['structure'])
            atoms.calc = calculator
            predicted_energy = atoms.get_potential_energy()
            predicted_forces = atoms.get_forces()

            predictions.append({
                "dft_energy": struct_data['dft_energy'],
                "predicted_energy": float(predicted_energy),
                "predicted_forces": predicted_forces.tolist()
            })

        # Save predictions
        output_file = Path(output_dir) / "predictions.json"
        with open(output_file, "w") as f:
            json.dump(predictions, f, indent=4)

        logger.info(f"Predictions saved to {output_file}")
        return str(output_file)

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise

def plot_predictions(predictions_file: str, output_dir: str) -> None:
    """
    Plot comparison between predicted values and DFT results
    
    Args:
        predictions_file: Path to predictions JSON file
        output_dir: Directory to save output plot
    """
    with open(predictions_file, "r") as f:
        predictions = json.load(f)

    dft_energies = np.array([p["dft_energy"] for p in predictions])
    predicted_energies = np.array([p["predicted_energy"] for p in predictions])

    # Calculate comprehensive metrics
    metrics = calculate_metrics(dft_energies, predicted_energies)

    # Create plot
    plt.figure(figsize=(10, 8))
    plt.scatter(dft_energies, predicted_energies, 
                color='#4A90E2', alpha=0.6, s=60, label='Predictions')

    # Add perfect prediction line
    min_val = min(dft_energies.min(), predicted_energies.min())
    max_val = max(dft_energies.max(), predicted_energies.max())
    plt.plot([min_val, max_val], [min_val, max_val],
             color='#FF6B6B', linestyle='--', label='Perfect prediction')

    plt.xlabel('DFT Energy (eV)', fontsize=12)
    plt.ylabel('Predicted Energy (eV)', fontsize=12)
    plt.title('M3GNet Predictions vs DFT', fontsize=14)
    plt.legend()

    # Add comprehensive statistics
    stats_text = (
        f'R² = {metrics["r2"]:.3f}\n'
        f'Adj. R² = {metrics["adj_r2"]:.3f}\n'
        f'MAE = {metrics["mae"]:.3f} eV\n'
        f'RMSE = {metrics["rmse"]:.3f} eV\n'
        f'Pearson r = {metrics["pearson_r"]:.3f}\n'
        f'Samples = {len(predictions)}'
    )
    plt.text(0.05, 0.95, stats_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
             verticalalignment='top',
             fontsize=10)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    output_plot_path = Path(output_dir) / "prediction_comparison.png"
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Print detailed statistics
    print("\nPrediction Statistics:")
    print(f"R² Score: {metrics['r2']:.3f}")
    print(f"Adjusted R²: {metrics['adj_r2']:.3f}")
    print(f"Mean Absolute Error: {metrics['mae']:.3f} eV")
    print(f"Root Mean Square Error: {metrics['rmse']:.3f} eV")
    print(f"Pearson Correlation: {metrics['pearson_r']:.3f}")
    print(f"P-value: {metrics['p_value']:.3e}")
    print(f"Number of samples: {len(predictions)}")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Use default paths
        predictions_file = predict_properties()
        plot_predictions(predictions_file, "./predictions_output")
    except Exception as e:
        logging.error(f"Program execution failed: {str(e)}")