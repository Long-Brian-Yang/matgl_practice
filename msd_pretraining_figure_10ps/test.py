#!/usr/bin/env python

import re
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.signal import savgol_filter
from ase.io.trajectory import Trajectory
from pathlib import Path
import argparse


def parse_log_diffusivities(log_file: str) -> dict:
    """
    Parse a given log file to extract lines like:
      'Total Results for XXXK:' 
      'Diffusion coefficient: YYY [cm²/s]'
    and build a dictionary mapping {temperature (float): diffusion_coefficient (float)}.

    Example lines:
      2024-12-24 20:08:57,444 - INFO - Total Results for 700K:
      2024-12-24 20:08:57,444 - INFO -   Diffusion coefficient: 2.915000e-06 [cm²/s]

    Returns:
        dict: {700.0: 2.915e-06, 800.0: 6.9054e-06, ...}
    """
    logD = {}           # Dictionary for temperature-to-diffusivity mapping
    current_temp = None  # Temporary variable to store temperature when we read a matching line

    with open(log_file, 'r') as f:
        for line in f:
            # Match a temperature line: "Total Results for 700K"
            match_temp = re.search(r'Total Results for (\d+)K', line)
            if match_temp:
                current_temp = float(match_temp.group(1))
                continue

            # Match a diffusion coefficient line: "Diffusion coefficient: 2.915000e-06 [cm²/s]"
            match_diff = re.search(r'Diffusion coefficient:\s+([\d\.eE+\-]+)\s+\[cm²/s\]', line)
            if match_diff and current_temp is not None:
                diff_val = float(match_diff.group(1))
                logD[current_temp] = diff_val
                current_temp = None  # Reset after storing the value

    if not logD:
        print(f"Warning: no diffusion data found in {log_file}!")
    else:
        print(f"Parsed diffusion coefficients from log:\n{logD}")

    return logD


def get_proton_indices(trajectory):
    """
    Get indices of all hydrogen atoms (H) from the first frame of the given trajectory.

    Args:
        trajectory (Trajectory): ASE Trajectory object

    Returns:
        list of int: List of indices for atoms with symbol 'H'
    """
    atoms = trajectory[0]
    symbols = atoms.get_chemical_symbols()
    return [i for i, symbol in enumerate(symbols) if symbol == 'H']


def calculate_msd_sliding_window(trajectory, proton_indices, timestep=0.5, window_size=None, loginterval=10):
    """
    Calculate Mean Squared Displacement (MSD) using a sliding window method.
    This method computes MSD_x, MSD_y, MSD_z, and MSD_total for a set of proton atoms.

    Args:
        trajectory (Trajectory): ASE Trajectory object
        proton_indices (list): Indices of proton atoms
        timestep (float): Time step in femtoseconds (default=0.5 fs)
        window_size (int): Window size for the MSD calculation (default=None)
        loginterval (int): Interval between frames in the original MD run (default=10)

    Returns:
        tuple: time (ps), msd_x, msd_y, msd_z, msd_total, D_x, D_y, D_z, D_total
               D_x, D_y, D_z, D_total are in units of Å²/ps
    """
    # Extract positions of all frames for the given proton_indices
    positions_all = np.array([atoms.get_positions() for atoms in trajectory])
    positions = positions_all[:, proton_indices]  # shape: (n_frames, n_protons, 3)
    n_frames = len(positions)

    # Determine window_size if not provided
    if window_size is None:
        window_size = min(n_frames // 2, 2000)
    shift_t = window_size // 10  # Overlapping shift for the sliding window

    # Initialize accumulators
    msd_x = np.zeros(window_size)
    msd_y = np.zeros(window_size)
    msd_z = np.zeros(window_size)
    msd_total = np.zeros(window_size)
    counts = np.zeros(window_size)

    # Get box size (assuming orthorhombic cell or ignoring off-diagonal)
    box_size = trajectory[0].get_cell().diagonal()

    # Sliding window loop
    for start in range(0, n_frames - window_size, shift_t):
        window = slice(start, start + window_size)
        ref_pos = positions[start]              # reference positions at the start frame
        disp = positions[window] - ref_pos[None, :, :]  # shape: (window_size, n_protons, 3)

        # Apply periodic boundary condition correction
        for i in range(3):
            mask = np.abs(disp[..., i]) > box_size[i] / 2
            disp[..., i][mask] -= np.sign(disp[..., i][mask]) * box_size[i]

        # Accumulate MSD for each time index
        msd_x += np.mean(disp[..., 0]**2, axis=1)
        msd_y += np.mean(disp[..., 1]**2, axis=1)
        msd_z += np.mean(disp[..., 2]**2, axis=1)
        msd_total += np.mean(np.sum(disp**2, axis=2), axis=1)
        counts += 1

    # Average over the counts
    msd_x /= counts
    msd_y /= counts
    msd_z /= counts
    msd_total /= counts

    # Optionally apply Savitzky-Golay filter to smooth the total MSD
    window_length = min(11, window_size // 40)
    if window_length > 3 and window_length % 2 == 0:
        window_length += 1
        msd_total = savgol_filter(msd_total, window_length, 2)

    # Convert time index to picoseconds
    # (timestep in fs, loginterval means how many fs per saved frame)
    # So each index corresponds to (timestep * loginterval) fs -> in ps is / 1000
    time = np.arange(window_size) * (timestep * loginterval) / 1000.0

    # Fit the MSD with a linear model in the range [start_fit : end_fit]
    start_fit = int(window_size * 0.1)
    end_fit = int(window_size * 0.9)
    X = sm.add_constant(time[start_fit:end_fit])  # shape: (n, 2)

    model_x = sm.OLS(msd_x[start_fit:end_fit], X).fit()
    D_x = model_x.params[1] / 2  # In Å²/ps

    model_y = sm.OLS(msd_y[start_fit:end_fit], X).fit()
    D_y = model_y.params[1] / 2

    model_z = sm.OLS(msd_z[start_fit:end_fit], X).fit()
    D_z = model_z.params[1] / 2

    model_total = sm.OLS(msd_total[start_fit:end_fit], X).fit()
    D_total = model_total.params[1] / 6  # 3D -> 6 in denominator for total

    return time, msd_x, msd_y, msd_z, msd_total, D_x, D_y, D_z, D_total


def analyze_msd_multiple(traj_files, labels=None, timestep=0.5,
                         window_size=None, loginterval=10,
                         extra_log_diff=None):
    """
    Analyze multiple trajectory files, calculate MSD and diffusion coefficients,
    and plot the results in two figures:
      - msd_analysis_all_components.png  (with x, y, z, total MSD)
      - msd_total.png                    (only total MSD)

    If extra_log_diff is provided (a dict mapping temperature->D from external log),
    the code will match the temperature in the label (like 'md_700K') and
    append the externally parsed diffusion coefficient to the plot legend.

    Args:
        traj_files (list): List of .traj file paths
        labels (list): Corresponding labels for these trajectories
        timestep (float): Timestep in fs
        window_size (int): Window size for MSD calculation
        loginterval (int): Logging interval between frames in MD
        extra_log_diff (dict): {temperature float: D from log}, optional
    """
    if extra_log_diff is None:
        extra_log_diff = {}

    fontsize = 24
    components = ['x', 'y', 'z', 'total']

    # Create a 2x2 figure for x, y, z, total MSD
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    axes = axes.flatten()

    # Create a separate figure for total MSD only
    plt.figure(figsize=(12, 8))

    for traj_file, label in zip(traj_files, labels):
        print(f"Analyzing trajectory: {traj_file}")

        # Load the trajectory
        trajectory = Trajectory(traj_file)

        # Identify proton indices
        proton_indices = get_proton_indices(trajectory)
        print(f"Found {len(proton_indices)} protons at indices: {proton_indices}")

        # Calculate MSD and diffusion
        time, msd_x, msd_y, msd_z, msd_total, D_x, D_y, D_z, D_total = calculate_msd_sliding_window(
            trajectory, proton_indices, timestep=timestep,
            window_size=window_size, loginterval=loginterval
        )

        # Convert from Å²/ps to cm²/s
        # 1 Å = 1e-8 cm, 1 ps = 1e-12 s
        # So 1 Å²/ps = (1e-8 cm)² / (1e-12 s) = 1e-16 / 1e-12 cm²/s = 1e-4 cm²/s
        # Actually, that's not correct: let's break it carefully:
        # 1 Å² = 1e-16 cm²
        # 1 ps = 1e-12 s
        # => 1 Å²/ps = 1e-16 cm² / 1e-12 s = 1e-4 cm²/s
        # We'll do it step by step as in your original code:
        D_x_cm2s = D_x * 1e-16 / 1e-12
        D_y_cm2s = D_y * 1e-16 / 1e-12
        D_z_cm2s = D_z * 1e-16 / 1e-12
        D_total_cm2s = D_total * 1e-16 / 1e-12

        # Attempt to parse temperature from label (e.g. "md_700K")
        temp_match = re.search(r'(\d+)K', label)
        if temp_match:
            t_val = float(temp_match.group(1))
            # If there's a match in extra_log_diff, append that to the label
            if t_val in extra_log_diff:
                D_from_log = extra_log_diff[t_val]
                label += f" (D={D_from_log:.2e} cm²/s)"

        # Prepare data for x, y, z, total in the 2x2 figure
        msds = [msd_x, msd_y, msd_z, msd_total]
        Ds = [D_x_cm2s, D_y_cm2s, D_z_cm2s, D_total_cm2s]

        for ax, msd_array, D_val, component in zip(axes, msds, Ds, components):
            ax.plot(time, msd_array, label=label)
            # Add a rough linear fit line just for visualization
            slope = np.polyfit(time, msd_array, 1)[0]
            ax.plot(time, time * slope, '--', alpha=0.5)

            ax.set_title(f"{component.upper()}-direction MSD" if component != 'total' else "Total MSD",
                         fontsize=fontsize)
            ax.set_xlabel("Time (ps)", fontsize=fontsize-4)
            ax.set_ylabel("MSD (Å²)", fontsize=fontsize-4)
            ax.tick_params(labelsize=fontsize-6)
            ax.legend(fontsize=fontsize-6)
            ax.grid(True, alpha=0.3)

        # Also plot total MSD in the separate figure
        plt.figure(2)
        plt.plot(time, msd_total, label=label)
        slope_total = np.polyfit(time, msd_total, 1)[0]
        plt.plot(time, time * slope_total, '--', alpha=0.5)

        print(f"Results for {label}:")
        print(f"  Total diffusion coefficient (from MSD calc): {D_total_cm2s:6.4e} [cm²/s]")

    # Save the x, y, z, total figure
    fig.tight_layout()
    fig.savefig('msd_analysis_all_components.png', dpi=300, bbox_inches='tight')

    # Finalize the total MSD figure
    plt.figure(2)
    plt.title("M3GNet Pre-training by VASP", fontsize=fontsize)
    plt.xlabel("Time (ps)", fontsize=fontsize-4)
    plt.ylabel("MSD (Å²)", fontsize=fontsize-4)
    plt.tick_params(labelsize=fontsize-6)
    plt.legend(fontsize=fontsize-6)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('msd_total.png', dpi=300, bbox_inches='tight')
    plt.close('all')


def main():
    """
    Main function: 
      1) Parse command-line arguments
      2) Optionally read a log file to get external diffusion data
      3) Run the MSD analysis for each .traj file
      4) Plot results, saving them to .png files
    """
    parser = argparse.ArgumentParser(description='Analyze multiple trajectory files with an optional log file.')
    parser.add_argument('traj_files', nargs='+',
                        help='Trajectory files (e.g., md_800K.traj md_900K.traj)')
    parser.add_argument('--timestep', type=float, default=0.5,
                        help='Timestep in fs (default: 0.5)')
    parser.add_argument('--loginterval', type=int, default=10,
                        help='Logging interval (default: 10)')
    parser.add_argument('--window-size', type=int,
                        help='Window size for MSD calculation')
    parser.add_argument('--log-file', type=str, default="./md_simulation.log",
                        help='Path to log file containing "Total Results for XXXK" lines.')

    args = parser.parse_args()

    # If the user provided a log file, parse it to get a dictionary of {temp -> D}
    if args.log_file:
        extra_log_diff = parse_log_diffusivities(args.log_file)
    else:
        extra_log_diff = {}

    # Generate labels from the trajectory filenames
    labels = [Path(traj).stem for traj in args.traj_files]

    # Analyze and plot MSD
    analyze_msd_multiple(
        args.traj_files,
        labels=labels,
        timestep=args.timestep,
        window_size=args.window_size,
        loginterval=args.loginterval,
        extra_log_diff=extra_log_diff
    )


if __name__ == "__main__":
    main()
