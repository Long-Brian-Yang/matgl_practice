import logging
import warnings
import numpy as np
import matgl
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from matgl.ext.ase import PESCalculator

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_model(model_path: str):
    """验证模型预测精度"""
    # 1. 加载模型
    logger.info(f"加载模型: {model_path}")
    model = matgl.load_model(model_path)

    # 2. 准备结构
    structure = Structure(
        lattice=[[4.192, 0.0, 0.0], [0.0, 4.192, 0.0], [0.0, 0.0, 4.192]],
        species=["Ba", "Zr", "O", "O", "O"],
        coords=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5]
        ]
    )

    # 3. 真实值
    true_energy = -41.72459923
    true_forces = np.array([
        [0.0, 0.0, 0.0],
        [-0.0, -0.0, -0.0],
        [0.0, -0.0, 0.0],
        [-0.0, 0.0, 0.0],
        [0.0, 0.0, -0.0]
    ])

    # 4. 预测
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = PESCalculator(potential=model)

    pred_energy = atoms.get_potential_energy()
    pred_forces = atoms.get_forces()

    # 5. 计算MAE
    energy_mae = np.abs(pred_energy - true_energy) / len(atoms)
    forces_mae = np.mean(np.abs(pred_forces - true_forces))

    # 6. 输出结果
    print("\n====== 模型验证结果 ======")
    print(f"预测能量: {pred_energy:.6f} eV")
    print(f"真实能量: {true_energy:.6f} eV")
    print(f"能量 MAE: {energy_mae:.6f} eV/atom  {'✓' if energy_mae < 0.01 else '✗'} (要求 < 0.01)")
    print(f"\n力 MAE: {forces_mae:.6f} eV/Å  {'✓' if forces_mae < 0.05 else '✗'} (要求 < 0.05)")


if __name__ == "__main__":
    MODEL_PATH = "./trained_model/final_model"
    validate_model(MODEL_PATH)
