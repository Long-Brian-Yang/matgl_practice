from ase.io import read, write
import numpy as np
from ase.build import make_supercell
from ase.neighborlist import NeighborList
import random
import math
from typing import List, Dict, Union, Optional
from ase.atoms import Atoms

# def calculate_local_energy(structure: Atoms, 
#                          index: int, 
#                          neighbors: np.ndarray, 
#                          dopant_indices: List[int], 
#                          min_distance: float = 6.0) -> float:
#     """
#     Calculate local energy based on dopant-dopant interactions
    
#     Args:
#         structure: ASE Atoms object representing the crystal structure
#         index: Index of the atom to calculate energy for
#         neighbors: Indices of neighboring atoms
#         dopant_indices: List of indices where dopants are located
#         min_distance: Minimum preferred distance between dopants
    
#     Returns:
#         float: Calculated local energy
#     """
#     energy = 0.0
#     repulsion_strength = 1.0  # Arbitrary energy unit
    
#     if index in dopant_indices:
#         for neighbor_idx in neighbors:
#             if neighbor_idx in dopant_indices:
#                 dist = structure.get_distance(index, neighbor_idx, mic=True)
#                 if dist < min_distance:
#                     energy += repulsion_strength * (min_distance - dist)
    
#     return energy
def calculate_local_energy(structure, index, neighbors, dopant_indices, min_distance=6.0):
    """
    Calculate local energy based on physical principles
    """
    energy = 0.0
    
    if index in dopant_indices:
        # Y-Y repulsion 
        for neighbor_idx in neighbors:
            if neighbor_idx in dopant_indices:
                dist = structure.get_distance(index, neighbor_idx, mic=True)
                if dist < min_distance:
                    energy += (min_distance - dist) ** 2
        
        # Local charge balance consideration
        nearby_oxygen = sum(1 for n in neighbors if structure[n].symbol == 'O')
        energy += abs(8 - nearby_oxygen)  
        
        # Lattice strain contribution
        strain_energy = 0.0
        for n in neighbors:
            if structure[n].symbol in ['Zr', 'Ba']:
                dist = structure.get_distance(index, n, mic=True)
                strain_energy += (dist - 2.1) ** 2
        energy += strain_energy
    
    return energy

def verify_composition(structure: Atoms, 
                      target_composition: Dict[str, int]) -> bool:
    """
    Verify if the structure has the expected composition
    
    Args:
        structure: ASE Atoms object to verify
        target_composition: Dictionary of expected element counts
    
    Returns:
        bool: True if composition matches, False otherwise
    """
    symbols = [atom.symbol for atom in structure]
    actual_composition = {symbol: symbols.count(symbol) for symbol in set(symbols)}
    
    return actual_composition == target_composition

def sort_structure(structure: Atoms) -> Atoms:
    """
    Sort atoms in the structure by element type and position
    
    Args:
        structure: ASE Atoms object to sort
    
    Returns:
        Atoms: Sorted structure
    """
    def sort_key(atom):
        # Define element priority (Ba first, then Zr, O, and Y)
        priority = {'Ba': 0, 'Zr': 1, 'O': 2, 'Y': 3, 'H': 4}
        return (
            priority[atom.symbol],
            atom.position[2],  # z coordinate
            atom.position[1],  # y coordinate
            atom.position[0]   # x coordinate
        )
    
    # Get sorted indices
    sorted_indices = sorted(range(len(structure)), key=lambda x: sort_key(structure[x]))
    # Return sorted structure
    return structure[sorted_indices]

def monte_carlo_doping(input_file: str, 
                      output_file: str, 
                      supercell_matrix: List[int] = [3,3,3], 
                      dopant: str = "Y", 
                      target_atom: str = "Zr", 
                      doping_fraction: float = 0.37, 
                      temperature: float = 800, 
                      mc_steps: int = 1000,
                      cutoff_radius: float = 8.0) -> Optional[Dict[str, int]]:
    """
    Perform Monte Carlo doping with energy considerations and atomic ordering
    
    Args:
        input_file: Path to input structure file
        output_file: Path to save optimized structure
        supercell_matrix: Supercell dimensions
        dopant: Dopant element symbol
        target_atom: Target atom to replace
        doping_fraction: Fraction of target atoms to replace
        temperature: Temperature for Monte Carlo simulation
        mc_steps: Number of Monte Carlo steps
        cutoff_radius: Cutoff radius for neighbor calculation
    
    Returns:
        Optional[Dict[str, int]]: Final composition or None if error occurs
    """
    try:
        # Read and create supercell
        structure = read(input_file)
        P = np.diag(supercell_matrix)
        supercell = make_supercell(structure, P)
        
        # Find all possible doping sites
        target_indices = [i for i, atom in enumerate(supercell) if atom.symbol == target_atom]
        num_dopants = max(1, int(round(len(target_indices) * doping_fraction)))
        
        print(f"\nStructure contains {len(supercell)} total atoms")
        print(f"Found {len(target_indices)} {target_atom} atoms")
        print(f"Will attempt to dope {num_dopants} sites with {dopant}")
        
        # Initial random doping
        dopant_indices = random.sample(target_indices, num_dopants)
        for idx in dopant_indices:
            supercell[idx].symbol = dopant
            
        # Setup neighbor list
        nl = NeighborList([cutoff_radius/2] * len(supercell), bothways=True, self_interaction=False)
        
        # Monte Carlo simulation
        kB = 8.617333262e-5  # Boltzmann constant in eV/K
        beta = 1.0 / (kB * temperature)
        
        accepted_moves = 0
        print("\nStarting Monte Carlo optimization...")
        
        for step in range(mc_steps):
            nl.update(supercell)
            
            # Randomly select a dopant and a target atom
            old_dopant = random.choice(dopant_indices)
            new_position = random.choice([i for i in target_indices if i not in dopant_indices])
            
            # Calculate initial local energy
            old_neighbors = nl.get_neighbors(old_dopant)[0]
            new_neighbors = nl.get_neighbors(new_position)[0]
            
            initial_energy = (calculate_local_energy(supercell, old_dopant, old_neighbors, dopant_indices) + 
                            calculate_local_energy(supercell, new_position, new_neighbors, dopant_indices))
            
            # Temporarily swap atoms
            supercell[old_dopant].symbol = target_atom
            supercell[new_position].symbol = dopant
            
            # Calculate new local energy
            new_dopant_indices = [i for i in dopant_indices if i != old_dopant] + [new_position]
            
            final_energy = (calculate_local_energy(supercell, new_position, new_neighbors, new_dopant_indices) + 
                          calculate_local_energy(supercell, old_dopant, old_neighbors, new_dopant_indices))
            
            # Accept or reject move based on Metropolis criterion
            delta_E = final_energy - initial_energy
            if delta_E < 0 or random.random() < math.exp(-beta * delta_E):
                dopant_indices = new_dopant_indices
                accepted_moves += 1
            else:
                # Reject move, revert changes
                supercell[old_dopant].symbol = dopant
                supercell[new_position].symbol = target_atom
            
            if (step + 1) % (mc_steps // 10) == 0:
                print(f"Step {step + 1}/{mc_steps}, Acceptance rate: {accepted_moves/(step+1):.2%}")
        
        # Remove H atom if present
        supercell = supercell[[atom.index for atom in supercell if atom.symbol != 'H']]
        
        # Sort structure
        supercell = sort_structure(supercell)
        
        # Final composition check
        symbols = [atom.symbol for atom in supercell]
        composition = {symbol: symbols.count(symbol) for symbol in set(symbols)}
        
        print("\nFinal composition:")
        for element, count in composition.items():
            print(f"{element}: {count}")
            
        # Verify composition
        expected_dopant_count = num_dopants
        actual_dopant_count = composition.get(dopant, 0)
        if actual_dopant_count != expected_dopant_count:
            print(f"\nWarning: Expected {expected_dopant_count} {dopant} atoms, "
                  f"but found {actual_dopant_count}")
        
        # Save final structure
        write(output_file, supercell)
        print(f"\nOptimized structure saved as {output_file}")
        
        return composition
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    composition = monte_carlo_doping(
        input_file="BaZrO3.cif",
        output_file="Y_BaZrO3_MC.cif",
        supercell_matrix=[3,3,3],
        doping_fraction=0.37,  
        temperature=800,       
        mc_steps=1000,
        cutoff_radius=8.0
    )