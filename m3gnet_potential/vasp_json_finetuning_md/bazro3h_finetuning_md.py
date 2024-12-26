from __future__ import annotations
import warnings
import numpy as np
from ase import Atom, Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory
from pymatgen.io.ase import AseAtomsAdaptor
from mp_api.client import MPRester
import matplotlib.pyplot as plt
import logging
import statsmodels.api as sm
import torch
import os

import matgl
from matgl.ext.ase import PESCalculator, MolecularDynamics


class StructureModifier:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def add_protons(self, atoms: Atoms, n_protons: int) -> Atoms:
        """
        Add protons to the structure based on theoretical understanding.

        Args:
            atoms (Atoms): Initial structure
            n_protons (int): Number of protons to add

        Returns:
            Atoms: Structure with protons added

        References:
        1. Kreuer, K. D., Solid State Ionics (1999)
        2. Björketun, M. E., et al., PRB (2005)
        3. Gomez, M. A., et al., SSI (2010)
        """
        # Theoretical OH bond length (from Gomez et al.)
        OH_BOND_LENGTH = 0.98  # Å
        MAX_NEIGHBOR_DIST = 3.0  # Å for neighbor search

        # Find oxygen atoms
        o_indices = [i for i, symbol in enumerate(atoms.get_chemical_symbols())
                     if symbol == 'O']

        if len(o_indices) < n_protons:
            self.logger.warning(
                f"Number of protons ({n_protons}) exceeds number of O atoms ({len(o_indices)})")
            n_protons = len(o_indices)

        # Track used oxygen atoms
        used_oxygens = []

        # Get cell and PBC
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()

        # Add protons near selected oxygen atoms
        for i in range(n_protons):
            # Select oxygen atom that hasn't been used
            available_oxygens = [
                idx for idx in o_indices if idx not in used_oxygens]
            if not available_oxygens:
                self.logger.warning(
                    "No more available oxygen atoms for proton incorporation")
                break

            o_idx = available_oxygens[0]
            used_oxygens.append(o_idx)
            o_pos = atoms.positions[o_idx]

            # Find neighboring oxygen atoms to determine optimal proton
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

            # Calculate optimal proton position
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

            # Calculate proton position
            h_pos = o_pos + direction * OH_BOND_LENGTH

            min_allowed_dist = 0.8  # Å
            for pos in atoms.positions:
                dist = atoms.get_distance(-1, len(atoms) - 1, mic=True)
                if dist < min_allowed_dist:
                    is_valid = False
                    break
            if any(pbc):
                scaled_pos = np.linalg.solve(cell.T, h_pos.T).T
                scaled_pos = scaled_pos % 1.0
                h_pos = cell.T @ scaled_pos

            is_valid = True
            for pos in atoms.positions:
                dist = np.linalg.norm(h_pos - pos)
                if dist < 0.5:
                    is_valid = False
                    break

            if not is_valid:
                self.logger.warning(
                    f"Invalid proton position near O atom {o_idx}, trying different direction")
                continue

            # Add proton
            atoms.append(Atom('H', position=h_pos))

            # Log OH bond length for verification
            oh_dist = atoms.get_distance(-1, o_idx, mic=True)
            self.logger.info(f"Added proton {i+1}/{n_protons}:")
            self.logger.info(f"  Near O atom: {o_idx}")
            self.logger.info(f"  Position: {h_pos}")
            self.logger.info(f"  OH distance: {oh_dist:.3f} Å")

        self.logger.info(f"Successfully added {n_protons} protons")
        self.logger.info(f"Final composition: {atoms.get_chemical_formula()}")

        return atoms

# def get_bazro3_structure():
#     """Get BaZrO3 structure from Materials Project database"""
#     mpr = MPRester(api_key="kzum4sPsW7GCRwtOqgDIr3zhYrfpaguK")
#     entries = mpr.get_entries("BaZrO3", property_data=["energy_per_atom"])

#     if not entries:
#         raise ValueError("No BaZrO3 structure found in the database")

#     sorted_entries = sorted(entries, key=lambda e: e.energy_per_atom)
#     structure = sorted_entries[0].structure
#     structure = structure.get_primitive_structure()

#     return structure


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


def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Get BaZrO3 structure
    test_structure = get_bazro3_structure()
    logger.info("Successfully loaded BaZrO3 structure")

    # Load fine-tuned model directly from saved model
    model_path = "/Users/brian/Documents/matgl_practice/m3gnet_potential/vasp_json_training_finetuning_m3gnet_potential/finetuned_model"
    pot = matgl.load_model(model_path)
    logger.info("Successfully loaded fine-tuned model")

    ase_adaptor = AseAtomsAdaptor()
    atoms = ase_adaptor.get_atoms(test_structure)

    # Add proton to the structure
    modifier = StructureModifier()
    atoms = modifier.add_protons(atoms, n_protons=1)
    proton_index = len(atoms) - 1
    logger.info(f"Modified structure composition: {atoms.get_chemical_formula()}")

    # Setup MD
    atoms.calc = PESCalculator(potential=pot)
    MaxwellBoltzmannDistribution(atoms, temperature_K=900)

    # Create trajectory file
    traj = Trajectory('md_bazro3h_finetuned.traj', 'w', atoms)

    # Setup MD simulation
    driver = MolecularDynamics(
        atoms,
        potential=pot,
        temperature=900,
        timestep=1.0,
        friction=0.002,
        trajectory=traj
    )

    # Run MD
    n_steps = 2000
    logger.info("Starting MD simulation with fine-tuned model...")

    for step in range(n_steps):
        driver.run(1)
        if step % 100 == 0:
            logger.info(f"Step {step}/{n_steps} completed")

    traj.close()

    # Load trajectory for analysis
    trajectory = Trajectory('md_bazro3h_finetuned.traj', 'r')

    # Calculate and plot MSD with consistent style
    time, msd = calculate_msd(trajectory, proton_index, timestep=1.0)

    # Plot MSD
    fontsize = 24
    plt.figure(figsize=(10, 6))
    plt.plot(time, msd, label="MSD")

    # Add linear fit
    model = sm.OLS(msd, time)
    result = model.fit()
    slope = result.params[0]
    plt.plot(time, time * slope, label="fitted line")
    plt.title("M3GNet fine-tuned by VASP", fontsize=fontsize)
    plt.xlabel("Time (ps)", fontsize=fontsize)
    plt.ylabel("MSD (Å²)", fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    plt.legend(fontsize=fontsize-4)
    plt.tight_layout()

    # Calculate and print diffusion coefficient
    D = slope / 6  # divide by degree of freedom (x, y, z, -x, -y, -z)
    print(f"Diffusion coefficient: {D*1e-16*1e12:6.4e} [cm^2/s]")

    plt.savefig('msd_evolution_finetuned.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("Diffusion analysis completed")
    logger.info(f"Maximum MSD: {np.max(msd):.2f} Å²")
    logger.info(f"Average MSD: {np.mean(msd):.2f} Å²")


if __name__ == "__main__":
    main()
