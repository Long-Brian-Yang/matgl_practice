import os
import json
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from pymatgen.core import Structure
from matgl.ext.ase import PESCalculator
from pymatgen.io.ase import AseAtomsAdaptor
from scipy import stats
import matgl

# 设置 DGL 后端
os.environ['DGLBACKEND'] = 'pytorch'

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    计算 MAE、RMSE、R² 三个指标。
    """
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)

    # 计算 R²
    mean_y = np.mean(y_true)
    ss_tot = np.sum((y_true - mean_y) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

def predict_properties(dataset_path: str = "./data/perovskite_dataset.json",
                       output_dir: str = "./predictions_output",
                       model_path: str = None) -> str:
    """
    对数据集中所有结构进行预测，计算体系的总能量和原子力。
    如果提供 model_path 且路径存在，则加载微调模型，否则加载预训练模型。

    Args:
        dataset_path (str): JSON 数据集路径，要求包含 "labels" 中的 "energies" 以及可选 "forces"。
        output_dir (str): 保存预测结果的输出目录。
        model_path (str): 模型路径（用于加载微调模型）。

    Returns:
        str: 预测结果 JSON 文件的路径。
    """
    logger = logging.getLogger(__name__)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"加载数据集：{dataset_path}")
        with open(dataset_path, "r") as f:
            data = json.load(f)
        
        structures = []
        dft_energies = data["labels"]["energies"]
        # 如果数据集提供了 DFT 力数据，则读取
        dft_forces = data["labels"].get("forces", None)

        for idx, item in enumerate(data["structures"]):
            struct = Structure(
                lattice=item["lattice"],
                species=item["species"],
                coords=item["coords"]
            )
            entry = {
                "structure": struct,
                "dft_energy_total": dft_energies[idx]
            }
            if dft_forces is not None:
                entry["dft_forces"] = dft_forces[idx]  # shape: (num_atoms, 3)
            structures.append(entry)
        logger.info(f"共加载结构数：{len(structures)}")

        # 加载模型：若指定 model_path 且存在，则加载微调模型；否则加载预训练模型
        if model_path and Path(model_path).exists():
            logger.info(f"加载微调模型：{model_path}")
            pot = matgl.load_model(model_path)
        else:
            logger.info("加载预训练模型：M3GNet-MP-2021.2.8-PES")
            pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")

        calculator = PESCalculator(potential=pot)

        predictions = []
        for i, struct_data in enumerate(structures, 1):
            logger.info(f"处理结构 {i}/{len(structures)}")
            atoms = AseAtomsAdaptor().get_atoms(struct_data['structure'])
            atoms.calc = calculator

            # 预测总能量 (eV) 和原子力 (eV/Å)
            predicted_energy = atoms.get_potential_energy()
            predicted_forces = atoms.get_forces()

            pred_entry = {
                "num_atoms": len(struct_data["structure"]),
                "dft_energy_total": float(struct_data["dft_energy_total"]),
                "predicted_energy_total": float(predicted_energy),
                "predicted_forces": predicted_forces.tolist()
            }
            if "dft_forces" in struct_data:
                pred_entry["dft_forces"] = struct_data["dft_forces"]
            predictions.append(pred_entry)
        
        output_file = Path(output_dir) / "predictions.json"
        with open(output_file, "w") as f:
            json.dump(predictions, f, indent=4)
        logger.info(f"预测结果已保存至：{output_file}")

        return str(output_file)

    except Exception as e:
        logger.error(f"预测失败：{str(e)}")
        raise

def _plot_subplots_2d(axs, x_data, y_data, x_label, y_label, title, metrics):
    """
    在一行 (axs[0], axs[1]) 上分别绘制：
    1) 左：散点图 + 对角线
    2) 右：2D 直方图 (hist2d)
    并在左侧散点图上显示 MAE, RMSE, R² 等信息。

    注：根据您的需求，散点图使用与示例相似的颜色和点大小。
    """
    # ---- 左侧散点图 ----
    axs[0].scatter(
        x_data,
        y_data,
        color=(31/255, 119/255, 180/255), 
        alpha=0.6,
        s=10
    )
    min_val = min(x_data.min(), y_data.min())
    max_val = max(x_data.max(), y_data.max())
    axs[0].plot([min_val, max_val], [min_val, max_val],
                color='grey', linestyle='--', linewidth=1)
    axs[0].set_xlabel(x_label)
    axs[0].set_ylabel(y_label)
    axs[0].set_title(title)

    # 显示 MAE, RMSE, R²
    text_str = (f"MAE = {metrics['mae']:.3f}\n"
                f"RMSE = {metrics['rmse']:.3f}\n"
                f"R² = {metrics['r2']:.3f}")
    axs[0].text(
        0.05, 0.95,
        text_str,
        transform=axs[0].transAxes,
        bbox=dict(facecolor='white', alpha=0.8),
        verticalalignment='top',
        fontsize=10
    )

    # ---- 右侧 2D 直方图 ----
    hb = axs[1].hist2d(
        x_data, y_data,
        bins=50,
        range=[[min_val, max_val], [min_val, max_val]],
        cmap='plasma'
    )
    # 添加 colorbar
    cb = plt.colorbar(hb[3], ax=axs[1])
    cb.set_label("Counts", fontsize=10)
    axs[1].set_xlabel(x_label)
    axs[1].set_ylabel(y_label)
    axs[1].set_title(title)

def plot_energy_force_comparison(predictions_file: str, output_dir: str) -> None:
    """
    读取预测结果 JSON，并分别绘制：
    1) 总能量 对比图（散点 + 2D直方图），显示 MAE、RMSE、R²
    2) 原子力 对比图（若数据包含 DFT 力），显示 MAE、RMSE、R²
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(predictions_file, "r") as f:
        predictions = json.load(f)

    # ============ 1) 总能量对比 ============ #
    dft_energy_total = np.array([p["dft_energy_total"] for p in predictions])
    pred_energy_total = np.array([p["predicted_energy_total"] for p in predictions])

    energy_metrics = calculate_metrics(dft_energy_total, pred_energy_total)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    _plot_subplots_2d(
        axs,
        dft_energy_total,
        pred_energy_total,
        x_label="DFT Energy (eV)",
        y_label="NNP Energy (eV)",
        title="Total Energy Comparison",
        metrics=energy_metrics
    )
    plt.tight_layout()
    energy_plot_path = Path(output_dir) / "energy_comparison.png"
    plt.savefig(energy_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("【总能量预测】")
    print(f"  MAE  = {energy_metrics['mae']:.3f} eV")
    print(f"  RMSE = {energy_metrics['rmse']:.3f} eV")
    print(f"  R²   = {energy_metrics['r2']:.3f}")

    # ============ 2) 原子力对比 ============ #
    # 如果没有 dft_forces，则跳过力的对比
    if "dft_forces" not in predictions[0]:
        print("\n【未检测到 DFT 力数据，跳过力对比绘图】")
        return

    # 将所有结构的力拼接到一起 (flatten)
    dft_forces_all = []
    pred_forces_all = []
    for p in predictions:
        dft_f = np.array(p["dft_forces"])      # shape: (num_atoms, 3)
        pred_f = np.array(p["predicted_forces"])
        dft_forces_all.append(dft_f.flatten())  # 变成一维
        pred_forces_all.append(pred_f.flatten())
    dft_forces_all = np.concatenate(dft_forces_all)
    pred_forces_all = np.concatenate(pred_forces_all)

    force_metrics = calculate_metrics(dft_forces_all, pred_forces_all)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    _plot_subplots_2d(
        axs,
        dft_forces_all,
        pred_forces_all,
        x_label="DFT Force (eV/Å)",
        y_label="NNP Force (eV/Å)",
        title="Atomic Force Comparison",
        metrics=force_metrics
    )
    plt.tight_layout()
    force_plot_path = Path(output_dir) / "force_comparison.png"
    plt.savefig(force_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("\n【原子力预测】")
    print(f"  MAE  = {force_metrics['mae']:.3f} eV/Å")
    print(f"  RMSE = {force_metrics['rmse']:.3f} eV/Å")
    print(f"  R²   = {force_metrics['r2']:.3f}")

if __name__ == "__main__":
    # 配置日志输出
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        # 1. 预测（可切换微调模型 / 预训练模型）
        predictions_file = predict_properties(
            dataset_path="./data/perovskite_dataset.json",
            output_dir="./predictions_output",
            model_path="./finetuned_model/final_model"  # 若存在则加载微调模型，否则加载预训练
        )
        # 2. 绘制对比图：能量 & 力
        plot_energy_force_comparison(predictions_file, "./predictions_output")

    except Exception as e:
        logging.error(f"程序执行失败：{str(e)}")
