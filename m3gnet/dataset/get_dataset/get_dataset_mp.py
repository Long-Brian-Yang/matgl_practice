import os
import sys
import json
import numpy as np
import logging
from mp_api.client import MPRester
from pymatgen.core import Structure

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def get_common_perovskite_systems():
    """Get list of common perovskite chemical systems"""
    # Common A-site elements
    a_site = ["Sr", "Ba", "Ca", "Pb", "K", "Na", "La", "Pr", "Nd", "Gd"]
    
    # Common B-site elements
    b_site = ["Ti", "Zr", "Nb", "Ta", "Mn", "Fe", "Co", "Ni", "Ru", "Cr"]
    
    # Generate all combinations
    systems = []
    for a in a_site:
        for b in b_site:
            systems.append(f"{a}-{b}-O")
            
    return systems

def is_likely_perovskite(structure):
    """Check if the structure is likely to be a perovskite with relaxed criteria"""
    # 1. 基本的化学计量比检查
    comp = structure.composition
    if "O" not in comp:
        return False
    
    # 获取非氧元素
    non_o_elements = [el for el in comp.elements if el.symbol != "O"]
    if len(non_o_elements) != 2:  # 应该有两种阳离子（A和B）
        return False
    
    # 检查氧的比例
    total_atoms = sum(comp.values())
    o_ratio = comp["O"] / total_atoms
    if not np.isclose(o_ratio, 0.6, atol=0.1):
        return False
    
    # 2. 晶格检查
    a, b, c = structure.lattice.abc
    alpha, beta, gamma = structure.lattice.angles
    
    # 检查晶格角度
    angle_tolerance = 20
    near_90_angles = sum(1 for angle in [alpha, beta, gamma] 
                        if abs(angle - 90) < angle_tolerance)
    if near_90_angles < 2:
        return False
    
    # 检查晶格参数
    max_deviation = max(abs(x - y)/max(x, y) 
                       for x, y in [(a, b), (b, c), (c, a)])
    if max_deviation > 0.3:
        return False
    
    return True

def get_perovskite_data(api_key, num_materials=100):
    """Get ABO3 perovskite data using optimized batch search"""
    logger = setup_logging()
    
    dataset = {
        "structures": [],
        "labels": {
            "energies": [],
            "stresses": [],
            "forces": []
        }
    }
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    with MPRester(api_key) as mpr:
        try:
            # 构建化学系统列表
            a_site = ["Sr", "Ba", "Ca", "Pb", "K", "Na", "La", "Pr", "Nd", "Gd"]
            b_site = ["Ti", "Zr", "Nb", "Ta", "Mn", "Fe", "Co", "Ni", "Ru", "Cr"]
            
            # 创建所有可能的化学系统组合
            all_systems = [f"{a}-{b}-O" for a in a_site for b in b_site]
            
            # 使用化学系统批量查询
            all_docs = []
            batch_size = 10  # 每次查询10个系统
            
            for i in range(0, len(all_systems), batch_size):
                batch_systems = all_systems[i:i + batch_size]
                for system in batch_systems:
                    try:
                        docs = mpr.materials.summary.search(
                            chemsys=system,
                            fields=["material_id", "formula_pretty", "elements",
                                   "energy_per_atom", "nsites"]
                        )
                        all_docs.extend(docs)
                        logger.info(f"Found {len(docs)} materials for system {system}")
                    except Exception as e:
                        logger.error(f"Error searching system {system}: {str(e)}")
            
            logger.info(f"Found {len(all_docs)} total candidates")
            
            # 预筛选可能的钙钛矿
            filtered_docs = [doc for doc in all_docs if doc.nsites % 5 == 0]
            logger.info(f"Filtered to {len(filtered_docs)} potential perovskites")
            
            # 批量处理结构
            structures_cache = {}
            for doc in filtered_docs:
                if success_count >= num_materials:
                    break
                    
                try:
                    # 获取结构对象
                    structure = mpr.get_structure_by_material_id(doc.material_id)
                    
                    if not is_likely_perovskite(structure):
                        skipped_count += 1
                        logger.info(f"Skipping {doc.material_id}: Not likely a perovskite")
                        continue
                    
                    # 准备结构数据
                    structure_data = {
                        "lattice": structure.lattice.matrix.tolist(),
                        "species": [str(sp) for sp in structure.species],
                        "coords": structure.frac_coords.tolist()
                    }
                    
                    # 准备标签数据
                    energy = doc.energy_per_atom * len(structure)
                    stress = np.zeros(9).tolist()  # 占位用的应力张量
                    forces = np.zeros((len(structure), 3)).tolist()  # 占位用的力
                    
                    # 添加到数据集
                    dataset["structures"].append(structure_data)
                    dataset["labels"]["energies"].append(energy)
                    dataset["labels"]["stresses"].append(stress)
                    dataset["labels"]["forces"].append(forces)
                    
                    success_count += 1
                    logger.info(f"Successfully added {doc.material_id} ({doc.formula_pretty})")
                    
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Error processing {doc.material_id}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in main search: {str(e)}")
    
    logger.info(f"\nResults summary:")
    logger.info(f"Successfully processed: {success_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Skipped (non-perovskite): {skipped_count}")
    logger.info(f"Total processed: {success_count + failed_count + skipped_count}")
    
    return dataset

def save_dataset(dataset, output_file):
    """Save dataset to JSON file with improved error handling"""
    try:
        # Use temporary file
        temp_file = output_file + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump(dataset, f, indent=4)
        
        # Verify JSON format
        with open(temp_file, 'r') as f:
            json.load(f)
            
        # If verification successful, replace original file
        if os.path.exists(output_file):
            os.remove(output_file)
        os.rename(temp_file, output_file)
        
        print(f"Dataset saved to: {output_file}")
        print(f"\nDataset statistics:")
        print(f"Number of structures: {len(dataset['structures'])}")
            
    except Exception as e:
        print(f"Error saving dataset: {str(e)}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <mp_api_key> [num_materials] [output_file]")
        sys.exit(1)

    api_key = sys.argv[1]
    num_materials = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    output_file = sys.argv[3] if len(sys.argv) > 3 else "perovskite_dataset.json"

    dataset = get_perovskite_data(api_key, num_materials)
    save_dataset(dataset, output_file)

if __name__ == "__main__":
    main()