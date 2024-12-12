# M3GNet Project Complete Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Execution Instructions](#2-execution-instructions)
3. [Core Function Documentation](#3-core-function-documentation)
4. [Execution Examples](#4-execution-examples)

## 1. Project Overview

A comprehensive machine learning project for materials science using M3GNet, including training, fine-tuning, and molecular dynamics simulations for studying proton dynamics in materials.

## 2. Execution Instructions

### 2.1 Pretraining Model (pretraining.py)

```bash
python pretraining.py [arguments]

Key Arguments:
--max-epochs: Maximum training epochs (default: 1)
--batch-size: Training batch size (default: 1)
--learning-rate: Learning rate (default: 1e-3)
--force-weight: Force loss weight (default: 1.0)
--stress-weight: Stress loss weight (default: 0.1)
--decay-steps: Learning rate decay steps (default: 100)
--decay-alpha: Learning rate decay factor (default: 0.01)
--patience: Early stopping patience (default: 10)
--output-dir: Output directory (default: ./trained_model)
--dataset-path: Dataset path (default: ./data/dataset.json)
--device: Training device (default: cpu)
--debug: Enable debug mode
```

### 2.2 Fine-tuning Model (finetuning.py)

```bash
python finetuning.py [arguments]

Key Arguments:
--model-path: Path to pre-trained model (default: None)
--max-epochs: Maximum training epochs (default: 20)
--batch-size: Training batch size (default: 1)
--learning-rate: Learning rate (default: 1e-4)
--stress-weight: Stress loss weight (default: 0.01)
--patience: Early stopping patience (default: 10)
--output-dir: Output directory (default: ./finetuned_model)
--dataset-path: Dataset path (default: ./data/dataset.json)
--device: Training device (default: cpu)
--debug: Enable debug mode
```

### 2.3 MD Simulations

#### 2.3.1 Pretraining MD (pretraining_md.py)

```bash
python pretraining_md.py [arguments]

Key Arguments:
--temperatures: MD temperatures in K (default: [600])
--timestep: Time step in fs (default: 2.0)
--friction: Friction coefficient (default: 0.002)
--n-steps: Number of MD steps (default: 1000)
--n-protons: Number of protons to add (default: 1)
--output-dir: Directory to save outputs (default: ./pretraining_md_results)
--device: Training device (default: cpu)
--debug: Enable debug mode
```

#### 2.3.2 Finetuning MD (finetuning_md.py)

```bash
python finetuning_md.py [arguments]

Key Arguments:
--temperatures: MD temperatures in K (default: [900])
--timestep: Time step in fs (default: 1.0)
--friction: Friction coefficient (default: 0.002)
--n-steps: Number of MD steps (default: 2000)
--n-protons: Number of protons to add (default: 1)
--output-dir: Directory to save outputs (default: ./finetuning_md_results)
--model-path: Path to fine-tuned model (default: ./finetuned_model/final_model)
--device: Device for computation (default: cpu)
--debug: Enable debug mode
```

## 3. Core Function Documentation

### 3.1 Data Processing (dataset_json.py)

```python
def load_structures_from_json(json_path):
    """
    Load crystal structures from JSON file
    Returns: structures, energies
    """

def prepare_data(json_path, batch_size=16):
    """
    Prepare dataset for training
    Returns: train_loader, val_loader, test_loader
    """
```

### 3.2 MD Simulation Functions

```python
def add_protons(atoms: Atoms, n_protons: int) -> Atoms:
    """
    Add protons to structure for MD simulation
    Returns: Modified structure with added protons
    """

def calculate_msd(trajectory, atom_index, timestep=1.0):
    """
    Calculate Mean Square Displacement
    Returns: time array, MSD array
    """

def analyze_msd(trajectories: list, atom_index: int, temperatures: list,
                timestep: float, output_dir: Path, logger: logging.Logger):
    """
    Analyze MSD data and create visualization
    """

def run_md_simulation(args) -> None:
    """
    Run molecular dynamics simulation at multiple temperatures
    """
```

## 4. Execution Examples

### 4.1 Basic Usage

```bash
# Training
python pretraining.py

# Fine-tuning
python finetuning.py

# MD Simulations
python pretraining_md.py
python finetuning_md.py
```

### 4.2 Advanced Usage

```bash
# Custom training
python pretraining.py --max-epochs 100 --batch-size 8 --device cuda
python finetuning.py --max-epochs 50 --learning-rate 5e-5

# Custom MD simulation
python pretraining_md.py --temperatures 600 800 1000 --n-steps 5000
python finetuning_md.py --temperatures 900 1100 --timestep 0.5
```

### 4.3 Directory Structure

```
diffusion/
├── data/
│   └── dataset.json
├── trained_model/
│   ├── checkpoints/
│   └── final_model/
├── finetuned_model/
│   ├── checkpoints/
│   └── final_model/
├── pretraining_md_results/
│   ├── logs/
│   └── T_xxk/
│         └── xx.traj/
├── finetuning_md_results/
│   ├── logs/
│   └── T_xxk/
│         └── xx.traj/
├── pretraining.py
├── finetuning.py
├── dataset.py
├── pretraining_md.py
└── finetuning_md.py
```
