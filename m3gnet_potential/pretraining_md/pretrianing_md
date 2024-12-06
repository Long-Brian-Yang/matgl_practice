from __future__ import annotations
import warnings
import numpy as np
from ase import Atom
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory
from ase.io import write
from pymatgen.io.ase import AseAtomsAdaptor
from mp_api.client import MPRester
import matplotlib.pyplot as plt
import logging

import matgl
from matgl.ext.ase import PESCalculator, MolecularDynamics, Relaxer

class StructureModifier:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def add_protons(self, atoms, n_protons: int):
        """Add protons near oxygen atoms"""
        OH_BOND_LENGTH = 0.98  # Å
        MAX_NEIGHBOR_DIST = 3.0  # Å
        
        o_indices = [i for i, symbol in enumerate(atoms.get_chemical_symbols())
                     if symbol == 'O']
        
        if len(o_indices) < n_protons:
            n_protons = len(o_indices)
            
        used_oxygens = []
        cell = atoms.get_cell()
        
        for i in range(n_protons):
            available_oxygens = [idx for idx in o_indices if idx not in used_oxygens]
            if not available_oxygens:
                break
                
            o_idx = available_oxygens[0]
            used_oxygens.append(o_idx)
            o_pos = atoms.positions[o_idx]
            
            # Add proton near oxygen
            h_pos = o_pos + np.array([0, 0, OH_BOND_LENGTH])
            atoms.append(Atom('H', position=h_pos))
            
        return atoms

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Get MP structures
    mpr = MPRester(api_key="kzum4sPsW7GCRwtOqgDIr3zhYrfpaguK")
    entries = mpr.get_entries_in_chemsys(["Ba", "Zr", "O"])
    structures = [e.structure for e in entries]
    test_structure = structures[0]
    
    # Load pretrained model
    logger.info("Loading pre-trained M3GNet model...")
    pot = matgl.load_model("M3GNet-MP-2021.2.8-DIRECT-PES")
    
    # Convert to ASE atoms and add protons
    ase_adaptor = AseAtomsAdaptor()
    atoms = ase_adaptor.get_atoms(test_structure)
    
    # Add protons
    modifier = StructureModifier()
    atoms = modifier.add_protons(atoms, n_protons=1)
    logger.info(f"Modified structure composition: {atoms.get_chemical_formula()}")
    
    # Create trajectory file for relaxation
    relax_traj = Trajectory('relaxation_pretrained.traj', 'w')
    
    # Relax structure
    logger.info("Starting structure relaxation...")
    relaxer = Relaxer(potential=pot)
    modified_structure = ase_adaptor.get_structure(atoms)
    
    # Convert back to ASE atoms for relaxation
    atoms_to_relax = ase_adaptor.get_atoms(modified_structure)
    atoms_to_relax.calc = PESCalculator(potential=pot)
    
    relax_results = relaxer.relax(
        modified_structure, 
        fmax=0.01,
        trajectory=relax_traj  # Use 'trajectory' instead of 'trajectory_observer'
    )
    
    # Convert trajectory to xyz format
    write("relaxation_pretrained.xyz", relax_traj)
    relax_traj.close()
    
    # Setup MD
    atoms = ase_adaptor.get_atoms(relax_results["final_structure"])
    MaxwellBoltzmannDistribution(atoms, temperature_K=600)
    
    # Create trajectory file for MD
    md_traj = Trajectory('md_pretrained.traj', 'w', atoms)
    
    # Setup MD simulation
    driver = MolecularDynamics(
        atoms,
        potential=pot,
        temperature=600,
        temperature_damping_timescale=100.0,
        logfile="md_pretrained.log",
        trajectory=md_traj
    )
    
    # Run MD
    energies = []
    temperatures = []
    logger.info("Starting MD simulation...")
    
    for step in range(200):
        driver.run(1)
        energies.append(atoms.get_potential_energy())
        temperatures.append(driver.get_temperature())
        if step % 10 == 0:
            logger.info(f"Step {step}: E = {energies[-1]:.3f} eV, T = {temperatures[-1]:.1f} K")
    
    md_traj.close()
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(121)
    plt.plot(energies)
    plt.xlabel('MD Step')
    plt.ylabel('Potential Energy (eV)')
    plt.title('Energy Evolution - Pretrained Model')
    
    plt.subplot(122)
    plt.plot(temperatures)
    plt.xlabel('MD Step')
    plt.ylabel('Temperature (K)')
    plt.title('Temperature Evolution - Pretrained Model')
    
    plt.tight_layout()
    plt.savefig('md_evolution_pretrained.png')
    plt.close()
    
    logger.info("MD simulation completed")
    logger.info(f"Final energy: {energies[-1]:.3f} eV")
    logger.info(f"Average temperature: {np.mean(temperatures):.1f} K")

if __name__ == "__main__":
    main()