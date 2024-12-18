# proton_analysis.py
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import curve_fit
from ase.io.trajectory import Trajectory
from mpl_toolkits.mplot3d import Axes3D

def calculate_msd_sliding_window(trajectory: Trajectory, atom_indices: list,
                                 timestep: float = 1.0, window_size: int = None):
    """
    Calculate MSD using sliding window method for both directional and total MSD.
    
    Args:
        trajectory (Trajectory): ASE trajectory object
        atom_indices (list): indices of atoms to calculate MSD
        timestep (float): timestep in fs
        window_size (int): window size for MSD calculation
        
    Returns:
        time (np.ndarray): time array in ps
        msd_x (np.ndarray): MSD in x-direction
        msd_y (np.ndarray): MSD in y-direction
        msd_z (np.ndarray): MSD in z-direction
        msd_total (np.ndarray): total MSD
        D_x (float): diffusion coefficient in x-direction
        D_y (float): diffusion coefficient in y-direction
        D_z (float): diffusion coefficient in z-direction
        D_total (float): total diffusion coefficient
    """
    positions_all = np.array([atoms.get_positions() for atoms in trajectory])
    positions = positions_all[:, atom_indices]

    n_frames = len(positions)
    if window_size is None:
        window_size = n_frames // 4

    shift_t = window_size // 2  # Shift window by half its size

    # Initialize arrays for accumulating MSD values
    msd_x = np.zeros(window_size)
    msd_y = np.zeros(window_size)
    msd_z = np.zeros(window_size)
    msd_total = np.zeros(window_size)
    counts = np.zeros(window_size)

    # Calculate MSD using sliding windows
    n_windows = n_frames - window_size + 1
    for start in range(0, n_frames - window_size, shift_t):
        window = slice(start, start + window_size)
        ref_pos = positions[start]

        # Calculate displacements
        disp = positions[window] - ref_pos

        # Calculate MSD components
        msd_x += np.mean(disp[..., 0]**2, axis=1)
        msd_y += np.mean(disp[..., 1]**2, axis=1)
        msd_z += np.mean(disp[..., 2]**2, axis=1)
        msd_total += np.mean(np.sum(disp**2, axis=2), axis=1)
        counts += 1

    # Average MSDs
    msd_x /= counts
    msd_y /= counts
    msd_z /= counts
    msd_total /= counts

    # Calculate time array in picoseconds
    time = np.arange(window_size) * timestep / 1000

    # Calculate diffusion coefficients using statsmodels OLS
    model_x = sm.OLS(msd_x, sm.add_constant(time))
    D_x = model_x.fit().params[1] / 2  # For 1D

    model_y = sm.OLS(msd_y, sm.add_constant(time))
    D_y = model_y.fit().params[1] / 2

    model_z = sm.OLS(msd_z, sm.add_constant(time))
    D_z = model_z.fit().params[1] / 2

    model_total = sm.OLS(msd_total, sm.add_constant(time))
    D_total = model_total.fit().params[1] / 6  # For 3D

    return time, msd_x, msd_y, msd_z, msd_total, D_x, D_y, D_z, D_total


def analyze_msd(trajectories: list, proton_index: int, temperatures: list,
                timestep: float, output_dir: Path, logger: logging.Logger,
                window_size: int = None) -> None:
    """
    Analyze MSD data and create separate plots for x, y, z directions and total MSD
    """
    # First create subplot figure with all components
    fontsize = 24
    components = ['x', 'y', 'z', 'total']
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    axes = axes.flatten()

    for traj_file, temp in zip(trajectories, temperatures):
        logger.info(f"Analyzing trajectory for {temp}K...")
        trajectory = Trajectory(str(traj_file), 'r')

        # Calculate MSD with directional components
        time, msd_x, msd_y, msd_z, msd_total, D_x, D_y, D_z, D_total = calculate_msd_sliding_window(
            trajectory, [proton_index], timestep=timestep, window_size=window_size
        )

        # Convert diffusion coefficients to cm²/s
        D_x_cm2s = D_x * 1e-16 * 1e12
        D_y_cm2s = D_y * 1e-16 * 1e12
        D_z_cm2s = D_z * 1e-16 * 1e12
        D_total_cm2s = D_total * 1e-16 * 1e12

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
            trajectory, [proton_index], timestep=timestep, window_size=window_size
        )

        # Convert D to cm²/s
        D_cm2s = D_total * 1e-16 * 1e12

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

def plot_proton_pathway_3d(trajectory_file: str, proton_index: int, 
                          temperature: float, output_dir: Path) -> None:
    """
    Plot 3D proton diffusion pathway

    Args:
        trajectory_file (str): Path to trajectory file
        proton_index (int): Index of proton atom
        temperature (float): Temperature (K)
        output_dir (Path): Output directory
    """
    traj = Trajectory(trajectory_file, 'r')
    positions = np.array([atoms[proton_index].position for atoms in traj])
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            'b-', alpha=0.6, label='Proton path')
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
              c='g', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
              c='r', s=100, label='End')
    
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(f'Proton Diffusion Pathway at {temperature}K')
    ax.legend()
    
    plt.savefig(output_dir / f'proton_pathway_3d_{temperature}K.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def calculate_activation_energy(temperatures: list, diffusion_coefficients: list, 
                              output_dir: Path, logger: logging.Logger) -> tuple:
    """
    Calculate activation energy using Arrhenius equation

    Args:
        temperatures (list): List of temperatures (K)
        diffusion_coefficients (list): List of diffusion coefficients (cm²/s)
        output_dir (Path): Output directory
        logger (logging.Logger): Logger instance

    Returns:
        tuple: (activation_energy, pre_exponential_factor)
    """
    temps = np.array(temperatures)
    diff_coeffs = np.array(diffusion_coefficients)
    
    x_data = 1 / temps
    y_data = np.log(diff_coeffs)
    
    # Fit Arrhenius equation
    kB = 8.617333262145e-5  # Boltzmann constant (eV/K)
    popt, _ = curve_fit(lambda x, Ea, lnA: lnA - Ea/(kB*1/x), x_data, y_data)
    Ea, lnA = popt
    A = np.exp(lnA)
    
    plt.figure(figsize=(10, 8))
    plt.plot(x_data, y_data, 'bo', label='Data')
    plt.plot(x_data, lnA - Ea/(kB*1/x_data), 'r-', 
             label=f'Fit: Ea = {Ea:.3f} eV')
    
    plt.xlabel('1/T (K⁻¹)')
    plt.ylabel('ln(D) (ln(cm²/s))')
    plt.title('Arrhenius Plot of Proton Diffusion')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(output_dir / 'arrhenius_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Activation energy: {Ea:.3f} eV")
    logger.info(f"Pre-exponential factor: {A:.2e} cm²/s")
    
    return Ea, A

def calculate_vacf(trajectory_file: str, atom_index: int, timestep: float, 
                  max_dt: int = None) -> tuple:
    """
    Calculate velocity autocorrelation function
    """
    from ase.io import read
    atoms_list = read(trajectory_file, index=':', format='traj')
    
    try:
        velocities = []
        for atoms in atoms_list:
            if hasattr(atoms, 'get_velocities'):
                velocities.append(atoms.get_velocities()[atom_index])
            else:
                raise AttributeError("No velocity information in trajectory")
                
        velocities = np.array(velocities)
        
    except (AttributeError, KeyError):
        print("No velocities found in trajectory, calculating from positions...")
        positions = np.array([atoms[atom_index].position for atoms in atoms_list])
        velocities = np.zeros((len(positions)-2, 3))
        dt = timestep * 1e-15  # Convert fs to s
        for i in range(1, len(positions)-1):
            velocities[i-1] = (positions[i+1] - positions[i-1]) / (2 * dt)
    
    if len(velocities) == 0:
        return None, None
        
    if max_dt is None:
        max_dt = len(velocities) // 2
    
    vacf = np.zeros(max_dt)
    for dt in range(max_dt):
        vacf[dt] = np.mean([np.dot(velocities[t], velocities[t + dt]) 
                           for t in range(len(velocities) - dt)])
    
    vacf = vacf / vacf[0]  # 归一化
    return np.arange(max_dt), vacf

def plot_vacf(trajectories: list, temperatures: list, proton_index: int,
              timestep: float, output_dir: Path, logger: logging.Logger) -> None:
    """
    Plot velocity autocorrelation function for different temperatures
    """
    plt.figure(figsize=(12, 8))
    
    for traj_file, temp in zip(trajectories, temperatures):
        logger.info(f"Calculating VACF for {temp}K...")
        time, vacf = calculate_vacf(traj_file, proton_index, timestep)
        if time is not None and vacf is not None:
            time = time * timestep / 1000  # Convert to ps
        plt.plot(time, vacf, label=f'{temp}K')
    
    plt.xlabel('Time (ps)')
    plt.ylabel('VACF')
    plt.title('Velocity Autocorrelation Function')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(output_dir / 'vacf.png', dpi=300, bbox_inches='tight')
    plt.close()

def run_all_analysis(trajectories: list, temperatures: list, proton_index: int,
                    timestep: float, output_dir: Path, logger: logging.Logger,
                    window_size: int = None) -> None:
    """
    Run all analysis functions

    Args:
        trajectories (list): List of trajectory files
        temperatures (list): List of temperatures (K)
        proton_index (int): Proton index
        timestep (float): Timestep (fs)
        output_dir (Path): Output directory
        logger (logging.Logger): Logger instance
        window_size (int): Window size for MSD calculation
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. MSD analysis
        logger.info("Starting MSD analysis...")
        analyze_msd(trajectories, proton_index, temperatures,
                   timestep, output_dir, logger, window_size)

        # 2. 3D diffusion pathways
        logger.info("Plotting proton pathways...")
        for traj_file, temp in zip(trajectories, temperatures):
            plot_proton_pathway_3d(traj_file, proton_index, temp, output_dir)

        # 3. Calculate diffusion coefficients and activation energy
        logger.info("Calculating diffusion coefficients...")
        diffusion_coefficients = []
        for traj_file in trajectories:
            trajectory = Trajectory(str(traj_file), 'r')
            _, _, _, _, _, _, _, _, D_total = calculate_msd_sliding_window(
                trajectory, [proton_index], timestep=timestep, window_size=window_size
            )
            diffusion_coefficients.append(D_total * 1e-16 * 1e12)

        # Only calculate activation energy if we have multiple temperature points
        if len(temperatures) > 1:
            logger.info("Calculating activation energy...")
            Ea, A = calculate_activation_energy(temperatures, diffusion_coefficients, 
                                             output_dir, logger)
        else:
            logger.info("Skipping activation energy calculation - need at least two temperature points")

        # 4. VACF analysis
        logger.info("Calculating VACF...")
        plot_vacf(trajectories, temperatures, proton_index, timestep, output_dir, logger)

        logger.info("All analyses completed successfully!")

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise
            