import warnings
import numpy as np
import logging
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.api as sm
from ase import Atom, Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory
from pymatgen.io.ase import AseAtomsAdaptor
from mp_api.client import MPRester
from pymatgen.io.vasp import Poscar
from pymatgen.core import Structure
import matgl
from matgl.ext.ase import PESCalculator, MolecularDynamics

warnings.filterwarnings("ignore")


def setup_logging(log_dir: str = "logs") -> None:
    """
    Setup logging system

    Args:
        log_dir (str): log directory

    Returns:
        None
    """
    Path(log_dir).mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/finetuning_md_simulation.log"),
            logging.StreamHandler()
        ]
    )


def parse_args():
    """
    Parse command line arguments

    Args:
        None

    Returns:
        args (argparse.Namespace): parsed arguments
    """
    parser = argparse.ArgumentParser(description='MD Simulation for BaZrO3 with protons')
    parser.add_argument('--temperatures', type=float, nargs='+', default=[900],
                        help='Temperatures for MD simulation (K), e.g. 800 900 1000')
    parser.add_argument('--timestep', type=float, default=1.0,
                        help='Timestep for MD simulation (fs)')
    parser.add_argument('--friction', type=float, default=0.002,
                        help='Friction coefficient for MD')
    parser.add_argument('--n-steps', type=int, default=2000,
                        help='Number of MD steps')
    parser.add_argument('--n-protons', type=int, default=1,
                        help='Number of protons to add')
    parser.add_argument('--output-dir', type=str, default='./finetuning_md_results',
                        help='Directory to save outputs')
    parser.add_argument('--model-path', type=str,
                        default="../vasp_json_training_finetuning_m3gnet_potential/finetuned_model/final_model",
                        help='Path to fine-tuned model')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for computation (cpu/cuda)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')

    return parser.parse_args()


def add_protons(atoms: Atoms, n_protons: int) -> Atoms:
    """
    Add protons to the structure based on theoretical understanding.

    Args:
        atoms (Atoms): Initial structure
        n_protons (int): Number of protons to add

    Returns:
        Atoms: Structure with protons added
    """
    logger = logging.getLogger(__name__)
    OH_BOND_LENGTH = 0.98  # Å
    MAX_NEIGHBOR_DIST = 3.0  # Å

    o_indices = [i for i, symbol in enumerate(atoms.get_chemical_symbols()) if symbol == 'O']

    if len(o_indices) < n_protons:
        logger.warning(
            f"Number of protons ({n_protons}) exceeds number of O atoms ({len(o_indices)})")
        n_protons = len(o_indices)

    used_oxygens = []
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()

    for i in range(n_protons):
        available_oxygens = [idx for idx in o_indices if idx not in used_oxygens]
        if not available_oxygens:
            logger.warning(
                "No more available oxygen atoms for proton incorporation")
            break

        o_idx = available_oxygens[0]
        used_oxygens.append(o_idx)
        o_pos = atoms.positions[o_idx]

        neighbors = []
        for other_idx in o_indices:
            if other_idx != o_idx:
                dist = atoms.get_distance(o_idx, other_idx, mic=True)
                if dist < MAX_NEIGHBOR_DIST:
                    vec = atoms.get_distance(
                        o_idx, other_idx, vector=True, mic=True)
                    neighbors.append({
                        'idx': other_idx,
                        'dist': dist,
                        'vec': vec
                    })

        direction = np.zeros(3)
        if neighbors:
            for n in sorted(neighbors, key=lambda x: x['dist'])[:3]:
                weight = 1.0 / max(n['dist'], 0.1)
                direction -= n['vec'] * weight

            if np.linalg.norm(direction) > 1e-6:
                direction = direction / np.linalg.norm(direction)
            else:
                direction = np.array([0, 0, 1])
        else:
            direction = np.array([0, 0, 1])

        h_pos = o_pos + direction * OH_BOND_LENGTH

        is_valid = True
        min_allowed_dist = 0.8  # Å
        for pos in atoms.positions:
            dist = np.linalg.norm(h_pos - pos)
            if dist < min_allowed_dist:
                is_valid = False
                break

        if any(pbc):
            scaled_pos = np.linalg.solve(cell.T, h_pos.T).T
            scaled_pos = scaled_pos % 1.0
            h_pos = cell.T @ scaled_pos

        if not is_valid:
            logger.warning(
                f"Invalid proton position near O atom {o_idx}, trying different direction")
            continue

        atoms.append(Atom('H', position=h_pos))

        oh_dist = atoms.get_distance(-1, o_idx, mic=True)
        logger.info(f"Added proton {i+1}/{n_protons}:")
        logger.info(f"  Near O atom: {o_idx}")
        logger.info(f"  Position: {h_pos}")
        logger.info(f"  OH distance: {oh_dist:.3f} Å")

    logger.info(f"Successfully added {n_protons} protons")
    logger.info(f"Final composition: {atoms.get_chemical_formula()}")

    return atoms


def get_bazro3_structure():
    """Get structure with mp-3834 ID from Materials Project database"""
    mpr = MPRester(api_key="kzum4sPsW7GCRwtOqgDIr3zhYrfpaguK")
    structure = mpr.get_structure_by_material_id("mp-3834")

    if not structure:
        raise ValueError("Structure mp-3834 not found in the database")

    return structure


def calculate_msd(trajectory, atom_index, timestep=1.0):
    """Calculate Mean Square Displacement for a specific atom"""
    positions = []

    for atoms in trajectory:
        positions.append(atoms.positions[atom_index])

    positions = np.array(positions)
    initial_pos = positions[0]
    displacements = positions - initial_pos
    msd = np.sum(displacements**2, axis=1)
    time = np.arange(len(msd)) * timestep / 1000  # Convert fs to ps

    return time, msd


def analyze_msd(trajectories: list, proton_index: int, temperatures: list,
                timestep: float, output_dir: Path, logger: logging.Logger) -> None:
    """
    Analyze MSD data and create plots for all temperatures

    Args:
        trajectories: List of trajectory file paths
        proton_index: Index of the proton to track
        temperatures: List of temperatures
        timestep: MD timestep
        output_dir: Output directory
        logger: Logger instance
    """
    plt.figure(figsize=(12, 8))
    fontsize = 24

    for traj_file, temp in zip(trajectories, temperatures):
        logger.info(f"Analyzing trajectory for {temp}K...")
        trajectory = Trajectory(str(traj_file), 'r')
        time, msd = calculate_msd(trajectory, proton_index, timestep=timestep)

        model = sm.OLS(msd, time)
        result = model.fit()
        slope = result.params[0]
        D = slope / 6

        plt.plot(time, msd, label=f"{temp}K")
        plt.plot(time, time * slope, '--', alpha=0.5)

        logger.info(f"Results for {temp}K:")
        logger.info(f"  Diffusion coefficient: {D*1e-16*1e12:6.4e} [cm^2/s]")
        logger.info(f"  Maximum MSD: {np.max(msd):.2f} Å²")
        logger.info(f"  Average MSD: {np.mean(msd):.2f} Å²")

    plt.title("M3GNet fine-tuned by VASP", fontsize=fontsize)
    plt.xlabel("Time (ps)", fontsize=fontsize)
    plt.ylabel("MSD (Å²)", fontsize=fontsize)
    plt.tick_params(labelsize=fontsize - 4)
    plt.legend(fontsize=fontsize - 4)
    plt.tight_layout()

    plt.savefig(output_dir / 'finetuning_msd.png', dpi=300, bbox_inches='tight')
    plt.close()


def run_md_simulation(args) -> None:
    """
    Run molecular dynamics simulation at multiple temperatures

    Args:
        args (argparse.Namespace): command line arguments
    """
    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        setup_logging(str(output_dir / "logs"))
        logger = logging.getLogger(__name__)

        if args.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")

        logger.info("Loading BaZrO3 structure...")
        test_structure = get_bazro3_structure()
        logger.info("Successfully loaded BaZrO3 structure")

        logger.info("Loading potential model...")
        pot = matgl.load_model(args.model_path)
        logger.info(f"Loaded fine-tuned model from {args.model_path}")

        ase_adaptor = AseAtomsAdaptor()
        atoms = ase_adaptor.get_atoms(test_structure)
        atoms = add_protons(atoms, args.n_protons)
        proton_index = len(atoms) - 1

        pmg_structure = AseAtomsAdaptor().get_structure(atoms)
        poscar = Poscar(pmg_structure)
        poscar.write_file(output_dir / "POSCAR_with_H")

        trajectory_files = []

        for temp in args.temperatures:
            logger.info(f"\nStarting simulation at {temp}K...")

            temp_dir = output_dir / f"T_{temp}K"
            temp_dir.mkdir(exist_ok=True)

            current_atoms = atoms.copy()
            current_atoms.calc = PESCalculator(potential=pot)

            MaxwellBoltzmannDistribution(current_atoms, temperature_K=temp)

            traj_file = temp_dir / f"md_bazro3h_{temp}K.traj"
            trajectory_files.append(traj_file)
            traj = Trajectory(str(traj_file), 'w', current_atoms)

            driver = MolecularDynamics(
                current_atoms,
                potential=pot,
                temperature=temp,
                timestep=args.timestep,
                friction=args.friction,
                trajectory=traj
            )

            logger.info(f"Running MD at {temp}K...")
            for step in range(args.n_steps):
                driver.run(1)
                if step % 100 == 0:
                    logger.info(f"Temperature {temp}K - Step {step}/{args.n_steps}")

            traj.close()

        analyze_msd(trajectory_files, proton_index, args.temperatures,
                    args.timestep, output_dir, logger)

        logger.info("\nMD simulations completed successfully")

    except Exception as e:
        logger.error(f"MD simulation failed: {str(e)}")
        raise


if __name__ == "__main__":
    args = parse_args()
    try:
        run_md_simulation(args)
    except Exception as e:
        logging.error(f"Program failed: {str(e)}")
