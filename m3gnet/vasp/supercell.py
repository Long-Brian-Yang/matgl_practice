from ase.io import read, write
from ase.build import make_supercell
import numpy as np

def create_supercell(input_file, output_file, supercell_matrix=[2,2,2]):
    """
    Create a supercell without doping
    
    Parameters:
    -----------
    input_file : str
        Path to input CIF file
    output_file : str
        Path to output CIF file
    supercell_matrix : list
        Supercell dimensions [nx, ny, nz]
    """
    try:
        # Read the initial structure
        structure = read(input_file)
        print(f"\nOriginal cell:")
        print(f"Total atoms: {len(structure)}")
        
        # Create supercell
        P = np.diag(supercell_matrix)
        supercell = make_supercell(structure, P)
        
        # Count composition
        symbols = [atom.symbol for atom in supercell]
        composition = {symbol: symbols.count(symbol) for symbol in set(symbols)}
        
        print(f"\nSupercell {supercell_matrix}x{supercell_matrix}x{supercell_matrix}:")
        print("Composition:")
        for element, count in composition.items():
            print(f"{element}: {count}")
        
        # Save structure
        write(output_file, supercell)
        print(f"\nStructure saved as {output_file}")
        
        return composition
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    composition = create_supercell(
        input_file="BaZrO3.cif",     
        output_file="BaZrO3_333.cif", 
        supercell_matrix=[3,3,3]      
    )