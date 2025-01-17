import warnings
import numpy as np
import logging
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.signal import savgol_filter
from ase import Atom, Atoms
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Poscar

from pymatgen.core import Structure
import matgl
import torch
from matgl.ext.ase import PESCalculator, MolecularDynamics

warnings.filterwarnings("ignore")
torch.set_default_device("cuda")


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
            logging.FileHandler(f"{log_dir}/pretraining_md_simulation.log"),
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
    parser = argparse.ArgumentParser(description='MD Simulation with protons')
    parser.add_argument('--cif-file', type=str, default='./structures/BaZrO3_125.cif',
                        help='Path to input CIF file')
    parser.add_argument('--temperatures', type=float, nargs='+', default=[1000],
                        help='Temperatures for MD simulation (K), e.g. 800 900 1000')
    parser.add_argument('--timestep', type=float, default=0.5,
                        help='Timestep for MD simulation (fs)')
    parser.add_argument('--friction', type=float, default=0.01,
                        help='Friction coefficient for MD')
    parser.add_argument('--n-steps', type=int, default=20000,
                        help='Number of MD steps')
    parser.add_argument('--n-protons', type=int, default=2,
                        help='Number of protons to add')
    parser.add_argument('--output-dir', type=str, default='./pretraining_md_results',
                        help='Directory to save outputs')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to fine-tuned model (default: use pretrained)')
    parser.add_argument('--window-size', type=int, default=None,
                        help='Window size for MSD calculation')
    parser.add_argument('--loginterval', type=int, default=1,
                        help='Logging interval for MD simulation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for computation (cpu/cuda)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')

    return parser.parse_args()


# def add_protons(atoms: Atoms, n_protons: int, pot=None) -> Atoms:
#     """
#     Add protons to the structure near oxygen atoms

#     Args:
#         atoms (Atoms): input structure
#         n_protons (int): number of protons to add
#         pot (matgl.Potential): potential model for energy minimization

#     Returns:
#         atoms (Atoms): structure with protons added
#     """
#     logger = logging.getLogger(__name__)
#     OH_BOND_LENGTH = 0.98  # Å
#     MAX_NEIGHBOR_DIST = 3.0  # Å

#     o_indices = [i for i, symbol in enumerate(atoms.get_chemical_symbols()) if symbol == 'O']

#     if len(o_indices) < n_protons:
#         logger.warning(f"Number of protons ({n_protons}) exceeds number of O atoms ({len(o_indices)})")
#         n_protons = len(o_indices)

#     used_oxygens = []
#     cell = atoms.get_cell()
#     pbc = atoms.get_pbc()

#     for i in range(n_protons):
#         available_oxygens = [idx for idx in o_indices if idx not in used_oxygens]
#         if not available_oxygens:
#             logger.warning("No more available oxygen atoms for proton incorporation")
#             break

#         o_idx = available_oxygens[0]
#         used_oxygens.append(o_idx)
#         o_pos = atoms.positions[o_idx]

#         # Find neighboring oxygen atoms
#         neighbors = []
#         for other_idx in o_indices:
#             if other_idx != o_idx:
#                 dist = atoms.get_distance(o_idx, other_idx, mic=True)
#                 if dist < MAX_NEIGHBOR_DIST:
#                     vec = atoms.get_distance(o_idx, other_idx, vector=True, mic=True)
#                     neighbors.append({'idx': other_idx, 'dist': dist, 'vec': vec})

#         # Calculate proton position direction
#         direction = np.zeros(3)
#         if neighbors:
#             for n in sorted(neighbors, key=lambda x: x['dist'])[:3]:
#                 weight = 1.0 / max(n['dist'], 0.1)
#                 direction -= n['vec'] * weight

#             norm = np.linalg.norm(direction)
#             if norm > 1e-6:
#                 direction = direction / norm
#             else:
#                 direction = np.array([0, 0, 1])
#         else:
#             direction = np.array([0, 0, 1])

#         h_pos = o_pos + direction * OH_BOND_LENGTH

#         # Check position validity
#         is_valid = True
#         min_allowed_dist = 0.8  # Å
#         for pos in atoms.positions:
#             dist = np.linalg.norm(h_pos - pos)
#             if dist < min_allowed_dist:
#                 is_valid = False
#                 break

#         # Apply periodic boundary conditions if needed
#         if any(pbc):
#             scaled_pos = np.linalg.solve(cell.T, h_pos.T).T
#             scaled_pos = scaled_pos % 1.0
#             h_pos = cell.T @ scaled_pos

#         if not is_valid:
#             logger.warning(f"Invalid proton position near O atom {o_idx}, trying different direction")
#             continue

#         # Add proton
#         atoms.append(Atom('H', position=h_pos))
#         oh_dist = atoms.get_distance(-1, o_idx, mic=True)

#         logger.info(f"Added proton {i+1}/{n_protons}:")
#         logger.info(f"  Near O atom: {o_idx}")
#         logger.info(f"  Position: {h_pos}")
#         logger.info(f"  OH distance: {oh_dist:.3f} Å")

#     logger.info(f"Successfully added {n_protons} protons")
#     logger.info(f"Final composition: {atoms.get_chemical_formula()}")

#     # Perform energy minimization if potential is provided
#     if pot is not None:
#         logger.info("Starting energy minimization...")
#         atoms.calc = PESCalculator(potential=pot)
#         optimizer = BFGS(atoms)
#         try:
#             optimizer.run(fmax=0.05, steps=200)
#             logger.info(f"Energy minimization completed in {optimizer.get_number_of_steps()} steps")
#         except Exception as e:
#             logger.warning(f"Energy minimization failed: {str(e)}")

#     return atoms

# def add_protons(atoms: Atoms, n_protons: int) -> Atoms:
#     logger = logging.getLogger(__name__)
#     OH_BOND_LENGTH = 0.98  # Å
#     MAX_NEIGHBOR_DIST = 3.0  # Å

#     # 1. 找出所有氧原子
#     o_indices = [i for i, symbol in enumerate(atoms.get_chemical_symbols()) if symbol == 'O']

#     # 2. 计算晶胞中心
#     cell_center = np.mean(atoms.get_cell(), axis=0) / 2

#     # 3. 计算并排序氧原子到中心的距离
#     o_distances = []
#     for idx in o_indices:
#         pos = atoms.positions[idx]
#         # 使用 mic=True 来考虑周期性边界条件
#         dist = atoms.get_distance(idx, -1, mic=True, vector=True)
#         dist_to_center = np.linalg.norm(dist)
#         o_distances.append((dist_to_center, idx))

#     # 4. 选择最靠近中心的氧原子
#     o_distances.sort()  # 按距离排序
#     candidate_o_indices = [idx for _, idx in o_distances[:n_protons*2]]  # 选择更多候选点

#     used_oxygens = []
#     cell = atoms.get_cell()
#     pbc = atoms.get_pbc()

#     for i in range(n_protons):
#         # 从候选氧原子中选择未使用的
#         available_oxygens = [idx for idx in candidate_o_indices if idx not in used_oxygens]
#         if not available_oxygens:
#             break

#         o_idx = available_oxygens[0]
#         used_oxygens.append(o_idx)
#         o_pos = atoms.positions[o_idx]

#         # 寻找邻近氧原子
#         neighbors = []
#         for other_idx in o_indices:
#             if other_idx != o_idx:
#                 dist = atoms.get_distance(o_idx, other_idx, mic=True)
#                 if dist < MAX_NEIGHBOR_DIST:
#                     vec = atoms.get_distance(o_idx, other_idx, vector=True, mic=True)
#                     neighbors.append({'idx': other_idx, 'dist': dist, 'vec': vec})

#         # 计算质子位置，偏向体相内部
#         direction = np.zeros(3)
#         if neighbors:
#             # 结合邻近氧原子方向和朝向中心的方向
#             center_dir = cell_center - o_pos
#             center_dir = center_dir / np.linalg.norm(center_dir)

#             for n in sorted(neighbors, key=lambda x: x['dist'])[:3]:
#                 weight = 1.0 / max(n['dist'], 0.1)
#                 direction -= n['vec'] * weight

#             # 添加朝向中心的分量
#             direction += center_dir * 0.5
#             direction = direction / np.linalg.norm(direction)
#         else:
#             # 如果没有邻近氧原子，使用朝向中心的方向
#             direction = cell_center - o_pos
#             direction = direction / np.linalg.norm(direction)

#         h_pos = o_pos + direction * OH_BOND_LENGTH

#         # 检查位置有效性
#         is_valid = True
#         min_allowed_dist = 0.8  # Å
#         for pos in atoms.positions:
#             dist = np.linalg.norm(h_pos - pos)
#             if dist < min_allowed_dist:
#                 is_valid = False
#                 break

#         if any(pbc):
#             scaled_pos = np.linalg.solve(cell.T, h_pos.T).T
#             scaled_pos = scaled_pos % 1.0
#             h_pos = cell.T @ scaled_pos

#         if not is_valid:
#             logger.warning(f"Invalid proton position near O atom {o_idx}, trying different direction")
#             continue

#         atoms.append(Atom('H', position=h_pos))
#         oh_dist = atoms.get_distance(-1, o_idx, mic=True)

#         logger.info(f"Added proton {i+1}/{n_protons}:")
#         logger.info(f"  Near O atom: {o_idx}")
#         logger.info(f"  Position: {h_pos}")
#         logger.info(f"  OH distance: {oh_dist:.3f} Å")
#         logger.info(f"  Distance to center: {np.linalg.norm(h_pos - cell_center):.3f} Å")

#     return atoms
def add_protons(atoms: Atoms, n_protons: int, pot=None) -> Atoms:
    """
    Add protons to the structure near oxygen atoms with potential-guided relaxation

    Args:
        atoms (Atoms): Input structure
        n_protons (int): Number of protons to add
        pot (matgl.Potential): Potential for energy minimization

    Returns:
        atoms (Atoms): Structure with protons added
    """
    logger = logging.getLogger(__name__)
    OH_BOND_LENGTH = 0.98  # Initial O-H bond length in Å

    # Find all oxygen atoms
    o_indices = [i for i, symbol in enumerate(atoms.get_chemical_symbols()) if symbol == 'O']

    if len(o_indices) < n_protons:
        logger.warning(f"Number of protons ({n_protons}) exceeds number of O atoms ({len(o_indices)})")
        n_protons = len(o_indices)

    # Randomly select oxygen atoms to attach protons
    selected_o_indices = np.random.choice(o_indices, n_protons, replace=False)

    cell = atoms.get_cell()
    pbc = atoms.get_pbc()

    for i, o_idx in enumerate(selected_o_indices):
        o_pos = atoms.positions[o_idx]

        # Use random direction for initial proton placement
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)

        # Place proton at initial position
        h_pos = o_pos + direction * OH_BOND_LENGTH

        # Apply periodic boundary conditions if needed
        if any(pbc):
            scaled_pos = np.linalg.solve(cell.T, h_pos.T).T
            scaled_pos = scaled_pos % 1.0
            h_pos = cell.T @ scaled_pos

        # Add proton
        atoms.append(Atom('H', position=h_pos))
        oh_dist = atoms.get_distance(-1, o_idx, mic=True)

        logger.info(f"Added proton {i+1}/{n_protons}:")
        logger.info(f"  Near O atom: {o_idx}")
        logger.info(f"  OH distance: {oh_dist:.3f} Å")

    logger.info(f"Successfully added {n_protons} protons")
    logger.info(f"Final composition: {atoms.get_chemical_formula()}")

    # Perform very gentle energy minimization if potential is provided
    if pot is not None:
        logger.info("Starting gentle energy minimization...")
        atoms.calc = PESCalculator(potential=pot)
        optimizer = BFGS(atoms)
        try:
            # Use very loose convergence criteria
            optimizer.run(fmax=0.5, steps=100)  # 更宽松的力收敛标准，更少的步数
            logger.info(f"Energy minimization completed in {optimizer.get_number_of_steps()} steps")
        except Exception as e:
            logger.warning(f"Energy minimization failed: {str(e)}")

    return atoms


def get_structure_from_cif(cif_file: str) -> Structure:
    """
    Read structure from CIF file

    Args:
        cif_file (str): Path to CIF file

    Returns:
        Structure: Pymatgen Structure object
    """
    logger = logging.getLogger(__name__)
    try:
        # Check if directory exists
        cif_path = Path(cif_file)
        cif_path.parent.mkdir(parents=True, exist_ok=True)

        if not cif_path.exists():
            logger.error(f"CIF file not found: {cif_file}")
            raise FileNotFoundError(f"CIF file not found: {cif_file}")

        # Read structure using ASE
        atoms = read(str(cif_file))
        logger.info(f"Successfully read structure from {cif_file}")
        logger.info(f"Initial composition: {atoms.get_chemical_formula()}")

        # Convert to pymatgen Structure
        structure = AseAtomsAdaptor().get_structure(atoms)
        return structure

    except Exception as e:
        logger.error(f"Failed to read CIF file: {str(e)}")
        raise


# def calculate_msd_sliding_window(trajectory: Trajectory, atom_indices: list,
#                                  timestep: float = 1.0, window_size: int = None):
#     """
#     Calculate MSD using sliding window method for both directional and total MSD.

#     Args:
#         trajectory (Trajectory): ASE trajectory object
#         atom_indices (list): indices of atoms to calculate MSD
#         timestep (float): timestep in fs
#         window_size (int): window size for MSD calculation

#     Returns:
#         time (np.ndarray): time array in ps
#         msd_x (np.ndarray): MSD in x-direction
#         msd_y (np.ndarray): MSD in y-direction
#         msd_z (np.ndarray): MSD in z-direction
#         msd_total (np.ndarray): total MSD
#         D_x (float): diffusion coefficient in x-direction
#         D_y (float): diffusion coefficient in y-direction
#         D_z (float): diffusion coefficient in z-direction
#         D_total (float): total diffusion coefficient
#     """
#     positions_all = np.array([atoms.get_positions() for atoms in trajectory])
#     positions = positions_all[:, atom_indices]

#     n_frames = len(positions)
#     if window_size is None:
#         window_size = n_frames // 4

#     shift_t = window_size // 2  # Shift window by half its size

#     # Initialize arrays for accumulating MSD values
#     msd_x = np.zeros(window_size)
#     msd_y = np.zeros(window_size)
#     msd_z = np.zeros(window_size)
#     msd_total = np.zeros(window_size)
#     counts = np.zeros(window_size)

#     # Calculate MSD using sliding windows
#     n_windows = n_frames - window_size + 1
#     for start in range(0, n_frames - window_size, shift_t):
#         window = slice(start, start + window_size)
#         ref_pos = positions[start]

#         # Calculate displacements
#         disp = positions[window] - ref_pos

#         # Calculate MSD components
#         msd_x += np.mean(disp[..., 0]**2, axis=1)
#         msd_y += np.mean(disp[..., 1]**2, axis=1)
#         msd_z += np.mean(disp[..., 2]**2, axis=1)
#         msd_total += np.mean(np.sum(disp**2, axis=2), axis=1)
#         counts += 1

#     # Average MSDs
#     msd_x /= counts
#     msd_y /= counts
#     msd_z /= counts
#     msd_total /= counts

#     # Calculate time array in picoseconds
#     time = np.arange(window_size) * timestep / 1000

#     # Calculate diffusion coefficients using statsmodels OLS
#     model_x = sm.OLS(msd_x, sm.add_constant(time))
#     D_x = model_x.fit().params[1] / 2  # For 1D

#     model_y = sm.OLS(msd_y, sm.add_constant(time))
#     D_y = model_y.fit().params[1] / 2

#     model_z = sm.OLS(msd_z, sm.add_constant(time))
#     D_z = model_z.fit().params[1] / 2

#     model_total = sm.OLS(msd_total, sm.add_constant(time))
#     D_total = model_total.fit().params[1] / 6  # For 3D

#     return time, msd_x, msd_y, msd_z, msd_total, D_x, D_y, D_z, D_total


# def analyze_msd(trajectories: list, proton_index: int, temperatures: list,
#                 timestep: float, output_dir: Path, logger: logging.Logger,
#                 window_size: int = None) -> None:
#     """
#     Analyze MSD for all components and plot results

#     Args:
#         trajectories (list): List of trajectory files
#         proton_index (int): Index of the proton atom
#         temperatures (list): List of temperatures
#         timestep (float): Timestep for MD simulation
#         output_dir (Path): Output directory
#         logger (logging.Logger): Logger object
#         window_size (int): Window size for MSD calculation

#     Returns:
#         None
#     """
#     # First create subplot figure with all components
#     fontsize = 24
#     components = ['x', 'y', 'z', 'total']
#     fig, axes = plt.subplots(2, 2, figsize=(24, 20))
#     axes = axes.flatten()

#     for traj_file, temp in zip(trajectories, temperatures):
#         logger.info(f"Analyzing trajectory for {temp}K...")
#         trajectory = Trajectory(str(traj_file), 'r')

#         # Calculate MSD with directional components
#         time, msd_x, msd_y, msd_z, msd_total, D_x, D_y, D_z, D_total = calculate_msd_sliding_window(
#             trajectory, [proton_index], timestep=timestep, window_size=window_size
#         )

#         # Convert diffusion coefficients to cm²/s
#         D_x_cm2s = D_x * 1e-16 * 1e12
#         D_y_cm2s = D_y * 1e-16 * 1e12
#         D_z_cm2s = D_z * 1e-16 * 1e12
#         D_total_cm2s = D_total * 1e-16 * 1e12

#         # Plot each component
#         msds = [msd_x, msd_y, msd_z, msd_total]
#         Ds = [D_x_cm2s, D_y_cm2s, D_z_cm2s, D_total_cm2s]

#         for ax, msd, D, component in zip(axes, msds, Ds, components):
#             # Plot MSD
#             ax.plot(time, msd, label=f"{temp}K (D={D:.2e} cm²/s)")

#             # Linear fit using np.polyfit
#             slope = np.polyfit(time, msd, 1)[0]

#             # Plot fit line
#             ax.plot(time, time * slope, '--', alpha=0.5)

#             # Customize plot
#             ax.set_title(f"{component.upper()}-direction MSD" if component != 'total' else "Total MSD",
#                          fontsize=fontsize)
#             ax.set_xlabel("Time (ps)", fontsize=fontsize-4)
#             ax.set_ylabel("MSD (Å²)", fontsize=fontsize-4)
#             ax.tick_params(labelsize=fontsize-6)
#             ax.legend(fontsize=fontsize-6)
#             ax.grid(True, alpha=0.3)

#             # Log results
#             logger.info(f"Results for {temp}K ({component}-direction):")
#             logger.info(f"  Diffusion coefficient: {D:6.4e} [cm²/s]")
#             logger.info(f"  Maximum MSD: {np.max(msd):.2f} Å²")
#             logger.info(f"  Average MSD: {np.mean(msd):.2f} Å²")

#     plt.tight_layout()
#     plt.savefig(output_dir / 'msd_analysis_all_components.png', dpi=300, bbox_inches='tight')
#     plt.close()

#     # Create additional single plot for total MSD with style matching reference
#     plt.figure(figsize=(12, 8))

#     for traj_file, temp in zip(trajectories, temperatures):
#         trajectory = Trajectory(str(traj_file), 'r')
#         time, _, _, _, msd_total, _, _, _, D_total = calculate_msd_sliding_window(
#             trajectory, [proton_index], timestep=timestep, window_size=window_size
#         )

#         # Convert D to cm²/s
#         D_cm2s = D_total * 1e-16 * 1e12

#         # Plot MSD with diffusion coefficient in label
#         plt.plot(time, msd_total, label=f"{temp}K (D={D_cm2s:.2e} cm²/s)")

#         # Linear fit using np.polyfit
#         slope = np.polyfit(time, msd_total, 1)[0]

#         # Plot fit line
#         plt.plot(time, time * slope, '--', alpha=0.5)

#         logger.info(f"Total Results for {temp}K:")
#         logger.info(f"  Diffusion coefficient: {D_cm2s:6.4e} [cm²/s]")

#     plt.title("M3GNet pre-training by VASP", fontsize=fontsize)
#     plt.xlabel("Time (ps)", fontsize=fontsize)
#     plt.ylabel("MSD (Å²)", fontsize=fontsize)
#     plt.tick_params(labelsize=fontsize-4)
#     plt.legend(fontsize=fontsize-4)
#     plt.tight_layout()

#     plt.savefig(output_dir / 'msd_total.png', dpi=300, bbox_inches='tight')
#     plt.close()
def calculate_msd_sliding_window(trajectory: Trajectory, proton_indices: list,
                                 timestep: float = 0.5,  window_size: int = None,
                                 loginterval: int = 10):
    """
    Calculate mean squared displacement (MSD) using a sliding window approach

    Args:
        trajectory (Trajectory): ASE Trajectory object
        proton_indices (list): List of proton indices to calculate MSD
        timestep (float): MD timestep (fs)
        loginterval (int): Logging interval for MD simulation
        window_size (int): Size of sliding window for MSD calculation

    Returns:
        tuple: time, msd_x, msd_y, msd_z, msd_total, D_x, D_y, D_z, D_total
    """
    logging.debug(f"Calculating MSD for protons: {proton_indices}")

    positions_all = np.array([atoms.get_positions() for atoms in trajectory])
    positions = positions_all[:, proton_indices]
    n_frames = len(positions)

    if window_size is None:
        window_size = min(n_frames // 2, 2000)
    shift_t = window_size // 6
    # if window_size is None:
    #     window_size = min(n_frames // 3, 8000)
    # shift_t = window_size // 8

    # Initialize arrays
    msd_x = np.zeros(window_size)
    msd_y = np.zeros(window_size)
    msd_z = np.zeros(window_size)
    msd_total = np.zeros(window_size)
    counts = np.zeros(window_size)

    # Calculate box size for periodic boundary conditions
    box_size = trajectory[0].get_cell().diagonal()

    # MSD calculation
    for start in range(0, n_frames - window_size, shift_t):
        window = slice(start, start + window_size)
        ref_pos = positions[start]

        # Calculate displacements
        disp = positions[window] - ref_pos[None, :, :]

        # Apply periodic boundary conditions for each dimension
        for i in range(3):
            mask = np.abs(disp[..., i]) > box_size[i]/2
            disp[..., i][mask] -= np.sign(disp[..., i][mask]) * box_size[i]

        # Calculate MSD components - average over protons
        msd_x += np.mean(disp[..., 0]**2, axis=1)
        msd_y += np.mean(disp[..., 1]**2, axis=1)
        msd_z += np.mean(disp[..., 2]**2, axis=1)
        msd_total += np.mean(np.sum(disp**2, axis=2), axis=1)
        counts += 1

    # Average MSD
    msd_x /= counts
    msd_y /= counts
    msd_z /= counts
    msd_total /= counts

    # Minimal smoothing to handle obvious noise
    window_length = min(11, window_size // 40)
    # window_length = min(21, window_size // 30)
    if window_length > 3 and window_length % 2 == 0:
        window_length += 1
        # Only slightly smooth the total MSD
        msd_total = savgol_filter(msd_total, window_length, 2)

    time_per_frame = (timestep * loginterval) / 1000.0  # ps
    time = np.arange(window_size) * time_per_frame

    # Fit process
    # Use the middle 80% of data
    start_fit = int(window_size * 0.1)
    end_fit = int(window_size * 0.95)
    X = sm.add_constant(time[start_fit:end_fit])

    # Use ordinary least squares
    model_x = sm.OLS(msd_x[start_fit:end_fit], X).fit()
    D_x = model_x.params[1] / 2

    model_y = sm.OLS(msd_y[start_fit:end_fit], X).fit()
    D_y = model_y.params[1] / 2

    model_z = sm.OLS(msd_z[start_fit:end_fit], X).fit()
    D_z = model_z.params[1] / 2

    model_total = sm.OLS(msd_total[start_fit:end_fit], X).fit()
    D_total = model_total.params[1] / 6

    return time, msd_x, msd_y, msd_z, msd_total, D_x, D_y, D_z, D_total


def analyze_msd(trajectories: list, proton_indices: list, temperatures: list,
                timestep: float, output_dir: Path, logger: logging.Logger,
                window_size: int = None, loginterval: int = 10) -> None:
    """
    Analyze trajectories by calculating mean squared displacement (MSD) for each component

    Args:
        trajectories (list): List of trajectory files
        proton_indices (list): List of proton indices to calculate MSD
        temperatures (list): List of temperatures
        timestep (float): MD timestep (fs)
        output_dir (Path): Output directory
        logger (logging.Logger): Logger object
        window_size (int): Size of sliding window for MSD calculation
        loginterval (int): Logging interval for MD simulation

    Returns:
        None
    """
    logger.info(f"Starting MSD analysis for {len(proton_indices)} protons")
    logger.info(f"Proton indices: {proton_indices}")

    # Create subplot figure with all components
    fontsize = 24
    components = ['x', 'y', 'z', 'total']
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    axes = axes.flatten()

    for traj_file, temp in zip(trajectories, temperatures):
        logger.info(f"Analyzing trajectory for {temp}K...")
        trajectory = Trajectory(str(traj_file), 'r')

        # Calculate MSD with directional components
        time, msd_x, msd_y, msd_z, msd_total, D_x, D_y, D_z, D_total = calculate_msd_sliding_window(
            trajectory, proton_indices, timestep=timestep,
            window_size=window_size, loginterval=loginterval
        )

        # Convert diffusion coefficients to cm²/s
        D_x_cm2s = D_x * 1e-16 / 1e-12
        D_y_cm2s = D_y * 1e-16 / 1e-12
        D_z_cm2s = D_z * 1e-16 / 1e-12
        D_total_cm2s = D_total * 1e-16 / 1e-12

        # Plot each component
        msds = [msd_x, msd_y, msd_z, msd_total]
        Ds = [D_x_cm2s, D_y_cm2s, D_z_cm2s, D_total_cm2s]

        for ax, msd, D, component in zip(axes, msds, Ds, components):
            # Plot MSD
            ax.plot(time, msd, label=f"{temp}K (D={D:.2e} cm²/s)")

            # Linear fit using np.polyfit
            slope = np.polyfit(time, msd, 1)[0]

            # Plot fit line
            ax.plot(time, time * slope, '--', alpha=0.5)

            # Customize plot
            ax.set_title(f"{component.upper()}-direction MSD" if component != 'total' else "Total MSD",
                         fontsize=fontsize)
            ax.set_xlabel("Time (ps)", fontsize=fontsize-4)
            ax.set_ylabel("MSD (Å²)", fontsize=fontsize-4)
            ax.tick_params(labelsize=fontsize-6)
            ax.legend(fontsize=fontsize-6)
            ax.grid(True, alpha=0.3)

            # Log results
            logger.info(f"Results for {temp}K ({component}-direction):")
            logger.info(f"  Diffusion coefficient: {D:6.4e} [cm²/s]")
            logger.info(f"  Maximum MSD: {np.max(msd):.2f} Å²")
            logger.info(f"  Average MSD: {np.mean(msd):.2f} Å²")

    plt.tight_layout()
    plt.savefig(output_dir / 'msd_analysis_all_components.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create additional single plot for total MSD with style matching reference
    plt.figure(figsize=(12, 8))

    for traj_file, temp in zip(trajectories, temperatures):
        trajectory = Trajectory(str(traj_file), 'r')
        time, _, _, _, msd_total, _, _, _, D_total = calculate_msd_sliding_window(
            trajectory, proton_indices, timestep=timestep,
            window_size=window_size, loginterval=loginterval
        )

        # Convert D to cm²/s
        D_cm2s = D_total * 1e-16 / 1e-12

        # Plot MSD with diffusion coefficient in label
        plt.plot(time, msd_total, label=f"{temp}K (D={D_cm2s:.2e} cm²/s)")

        # Linear fit using np.polyfit
        slope = np.polyfit(time, msd_total, 1)[0]

        # Plot fit line
        plt.plot(time, time * slope, '--', alpha=0.5)

        logger.info(f"Total Results for {temp}K:")
        logger.info(f"  Diffusion coefficient: {D_cm2s:6.4e} [cm²/s]")

    plt.title("M3GNet pre-training by VASP", fontsize=fontsize)
    plt.xlabel("Time (ps)", fontsize=fontsize)
    plt.ylabel("MSD (Å²)", fontsize=fontsize)
    plt.tick_params(labelsize=fontsize-4)
    plt.legend(fontsize=fontsize-4)
    plt.tight_layout()

    plt.savefig(output_dir / 'msd_total.png', dpi=300, bbox_inches='tight')
    plt.close()


def run_md_simulation(args) -> None:
    """
    Run molecular dynamics simulation at multiple temperatures

    Args:
        args (argparse.Namespace): command line arguments

    Returns:
        None
    """
    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        setup_logging(str(output_dir / "logs"))
        logger = logging.getLogger(__name__)

        if args.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")

        # Load structure from CIF file
        logger.info(f"Loading structure from CIF file: {args.cif_file}...")
        structure = get_structure_from_cif(args.cif_file)
        logger.info("Successfully loaded structure")

        logger.info("Loading potential model...")
        if args.model_path and Path(args.model_path).exists():
            pot = matgl.load_model(args.model_path)
            logger.info(f"Loaded fine-tuned model from {args.model_path}")
        else:
            pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
            logger.info("Loaded pretrained M3GNet model")

        ase_adaptor = AseAtomsAdaptor()
        atoms = ase_adaptor.get_atoms(structure)
        initial_atom_count = len(atoms)
        atoms = add_protons(atoms, args.n_protons, pot)
        proton_indices = list(range(initial_atom_count, len(atoms)))
        logger.info(f"Added {len(proton_indices)} protons at indices: {proton_indices}")

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

            # Use structure name in trajectory file
            structure_name = Path(args.cif_file).stem
            traj_file = temp_dir / f"md_{structure_name}_{temp}K.traj"
            trajectory_files.append(traj_file)
            traj = Trajectory(str(traj_file), 'w', current_atoms)

            # driver = MolecularDynamics(
            #     current_atoms,
            #     potential=pot,
            #     temperature=temp,
            #     timestep=args.timestep,
            #     friction=args.friction,
            #     trajectory=traj
            # )

            # logger.info(f"Running MD at {temp}K...")
            # for step in range(args.n_steps):
            #     driver.run(1)
            #     if step % 100 == 0:
            #         logger.info(f"Temperature {temp}K - Step {step}/{args.n_steps}")

            # traj.close()

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
                if step % args.loginterval == 0:
                    current_atoms.set_velocities(driver.atoms.get_velocities())
                    traj.write(current_atoms)
                if step % 100 == 0:
                    logger.info(f"Temperature {temp}K - Step {step}/{args.n_steps}")

            traj.close()

        analyze_msd(trajectory_files, proton_indices, args.temperatures,
                    args.timestep, output_dir, logger, args.window_size,
                    loginterval=args.loginterval)

        logger.info("\nMD simulations completed for all temperatures")

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"MD simulation failed: {str(e)}")
        raise


if __name__ == "__main__":
    args = parse_args()
    try:
        run_md_simulation(args)
    except Exception as e:
        logging.error(f"Program failed: {str(e)}")
