# md_simulation.py
import warnings
import numpy as np
import logging
import argparse
from pathlib import Path
from ase import Atom, Atoms
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Poscar
from pymatgen.core import Structure
import matgl
from matgl.ext.ase import PESCalculator, MolecularDynamics

warnings.filterwarnings("ignore")


def setup_logging(log_dir: str = "logs") -> None:
    """
    Setup logging system

    Args:
        log_dir (str): Log directory path

    Returns:
        None
    """
    Path(log_dir).mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/md_simulation.log"),
            logging.StreamHandler()
        ]
    )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='MD Simulation with protons')
    parser.add_argument('--cif-file', type=str, default='./vasp/BaZrO3_333.cif',
                        help='Path to input CIF file')
    parser.add_argument('--temperatures', type=float, nargs='+',
                        default=[300, 600, 900],
                        help='Temperatures for MD simulation (K)')
    parser.add_argument('--timestep', type=float, default=0.5,
                        help='Timestep for MD simulation (fs)')
    parser.add_argument('--friction', type=float, default=0.01,
                        help='Friction coefficient for MD')
    parser.add_argument('--n-steps', type=int, default=2000,
                        help='Number of MD steps')
    parser.add_argument('--n-protons', type=int, default=1,
                        help='Number of protons to add')
    parser.add_argument('--output-dir', type=str, default='md_results',
                        help='Directory to save outputs')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to fine-tuned model (default: use pretrained)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for computation (cpu/cuda)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    return parser.parse_args()


def add_protons(atoms: Atoms, n_protons: int, pot=None) -> Atoms:
    """
    Add protons to the structure near oxygen atoms

    Args:
        atoms (Atoms): Initial structure
        n_protons (int): Number of protons to add
        pot (matgl.Potential): Potential model for energy minimization

    Returns:
        Atoms: Structure with protons added
    """
    logger = logging.getLogger(__name__)
    OH_BOND_LENGTH = 0.98  # Å
    MAX_NEIGHBOR_DIST = 3.0  # Å

    o_indices = [i for i, symbol in enumerate(atoms.get_chemical_symbols())
                 if symbol == 'O']

    if len(o_indices) < n_protons:
        logger.warning(f"Number of protons ({n_protons}) exceeds number of O atoms ({len(o_indices)})")
        n_protons = len(o_indices)

    used_oxygens = []
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()

    for i in range(n_protons):
        available_oxygens = [idx for idx in o_indices if idx not in used_oxygens]
        if not available_oxygens:
            logger.warning("No more available oxygen atoms")
            break

        o_idx = available_oxygens[0]
        used_oxygens.append(o_idx)
        o_pos = atoms.positions[o_idx]

        # Find neighboring oxygen atoms
        neighbors = []
        for other_idx in o_indices:
            if other_idx != o_idx:
                dist = atoms.get_distance(o_idx, other_idx, mic=True)
                if dist < MAX_NEIGHBOR_DIST:
                    vec = atoms.get_distance(o_idx, other_idx, vector=True, mic=True)
                    neighbors.append({'idx': other_idx, 'dist': dist, 'vec': vec})

        # Calculate proton position
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

        # Check position validity
        is_valid = True
        min_allowed_dist = 0.8  # Å
        for pos in atoms.positions:
            dist = np.linalg.norm(h_pos - pos)
            if dist < min_allowed_dist:
                is_valid = False
                break

        # Apply periodic boundary conditions
        if any(pbc):
            scaled_pos = np.linalg.solve(cell.T, h_pos.T).T
            scaled_pos = scaled_pos % 1.0
            h_pos = cell.T @ scaled_pos

        if not is_valid:
            logger.warning(f"Invalid proton position near O atom {o_idx}")
            continue

        atoms.append(Atom('H', position=h_pos))
        oh_dist = atoms.get_distance(-1, o_idx, mic=True)

        logger.info(f"Added proton {i+1}/{n_protons}:")
        logger.info(f"  Near O atom: {o_idx}")
        logger.info(f"  Position: {h_pos}")
        logger.info(f"  OH distance: {oh_dist:.3f} Å")

    # Energy minimization
    if pot is not None:
        logger.info("Starting energy minimization...")
        atoms.calc = PESCalculator(potential=pot)
        optimizer = BFGS(atoms)
        try:
            optimizer.run(fmax=0.05, steps=200)
            logger.info(f"Energy minimization completed in {optimizer.get_number_of_steps()} steps")
        except Exception as e:
            logger.warning(f"Energy minimization failed: {str(e)}")

    return atoms


def run_md_simulation(args) -> None:
    """Run molecular dynamics simulation"""
    try:
        # Initialize
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(str(output_dir / "logs"))
        logger = logging.getLogger(__name__)

        if args.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")

        # Load structure and model
        logger.info(f"Loading structure from: {args.cif_file}")
        atoms = read(args.cif_file)

        logger.info("Loading potential model...")
        if args.model_path and Path(args.model_path).exists():
            pot = matgl.load_model(args.model_path)
            logger.info(f"Loaded fine-tuned model from {args.model_path}")
        else:
            pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
            logger.info("Loaded pretrained M3GNet model")

        # Add protons
        atoms = add_protons(atoms, args.n_protons, pot)
        proton_index = len(atoms) - 1

        # Save structure with protons
        structure = AseAtomsAdaptor().get_structure(atoms)
        Poscar(structure).write_file(output_dir / "POSCAR_with_H")

        # Run MD simulations
        trajectory_files = []
        for temp in args.temperatures:
            logger.info(f"\nStarting simulation at {temp}K...")

            temp_dir = output_dir / f"T_{temp}K"
            temp_dir.mkdir(exist_ok=True)

            current_atoms = atoms.copy()
            current_atoms.calc = PESCalculator(potential=pot)
            MaxwellBoltzmannDistribution(current_atoms, temperature_K=temp)

            traj_file = temp_dir / f"md_{temp}K.traj"
            trajectory_files.append(traj_file)
            traj = Trajectory(str(traj_file), 'w', current_atoms)

            driver = MolecularDynamics(
                current_atoms,
                potential=pot,
                temperature=temp,
                timestep=args.timestep,
                friction=args.friction
                # trajectory=traj
            )

            logger.info("Running MD simulation...")
            for step in range(args.n_steps):
                driver.run(1)
                current_atoms.set_velocities(driver.atoms.get_velocities())
                traj.write(current_atoms)
                if step % 100 == 0:
                    logger.info(f"Temperature {temp}K - Step {step}/{args.n_steps}")

            traj.close()

        # Run analysis
        from diffusion_analysis import run_all_analysis
        logger.info("\nStarting analysis...")
        try:
            run_all_analysis(
                trajectories=trajectory_files,
                temperatures=args.temperatures,
                proton_index=proton_index,
                timestep=args.timestep,
                output_dir=output_dir,
                logger=logger
            )
            logger.info("Analysis completed!")
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"MD simulation failed: {str(e)}")
        raise


if __name__ == "__main__":
    args = parse_args()
    try:
        run_md_simulation(args)
    except Exception as e:
        logging.error(f"Program failed: {str(e)}")
