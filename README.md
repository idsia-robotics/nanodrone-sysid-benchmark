# Quadrotor Control Benchmark & Identification

This repository contains benchmarks and identification methods for quadrotor control, serving as a support resource for our paper submitted to the **Special Issue on Machine Learning and Control Engineering**.

## Overview

Control engineering is evolving rapidly with the integration of machine learning algorithms. However, benchmarking and comparing these algorithms remains a challenge due to a lack of shared code and high-fidelity models. This project addresses this gap by providing:

- **Reference implementations** for system identification using Physics-based, Neural, Hybrid (Residual), and LSTM approaches.
- **Training and Testing scripts** to reproduce results.
- **Data handling** utilities for quadrotor datasets.
- Tools for visualization and comparison.

## Repository Structure (Main Branch)

The `main` branch focuses on the PyTorch-based identification models, training, and evaluation.

```
├── models/             # Model architectures (Phys, Neural, Hybrid, LSTM)
├── dataset/            # Data loading (PyTorch Dataset)
├── train/              # Training scripts
├── test/               # Testing scripts
├── results/            # Results analysis and notebooks
├── data/               # Datasets
├── out/                # Exported models and predictions
├── scalers/            # Data scalers
├── utils/              # Utility functions (plots, metrics, etc.)
├── animations/         # Visualizations of flights
├── figures/            # Generated plots
├── requirements.txt    # Python dependencies
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)

### Step 1: Install Dependencies

Install the standard requirements:

```bash
pip install -r requirements.txt
```

### Step 2: Install PyTorch3D

This project uses `pytorch3d` for 3D transformations (quaternions).

**Option A: Use provided wheels (in dev branch)**
Wheels for some platforms might be available in the `dev` branch. You can check them out or download them.

**Option B: Build from source / Download wheels**
You can download wheels from the [PyTorch3D Wheel Builder](https://miropsota.github.io/torch_packages_builder/pytorch3d/) or follow the official [PyTorch3D installation guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).

## Usage

### 1. Training

The `train/` directory contains scripts to train different models.

Example: Train an LSTM model
```bash
python train/train_lstm.py --train_trajs '["random", "square", "chirp"]' --epochs 5000
```

Example: Train a Physics+Residual model
```bash
python train/train_phys+res.py
```

### 2. Testing

The `test/` directory contains scripts to evaluate trained models.

Example: Test LSTM model
```bash
python test/test_lstm.py
```

### 3. Comparison & Results

Use `results/model_comparison.py` or `results/model_comparison.ipynb` to compare different models and generate plots.

```bash
python results/model_comparison.py
```

## Models Description

`models/models.py` implements:

1.  **`PhysQuadModel`**: Physics-based model using rigid body dynamics and RK4 integration.
2.  **`NeuralQuadModel`**: Purely data-driven MLP model.
3.  **`ResidualQuadModel`**: Hybrid model (Physics + Neural Residual).
4.  **`QuadLSTM`**: LSTM-based model for temporal dependencies.

## EXTRA RESOURCES

Additional resources are available in the `.dev` branch (check out `origin/dev`), including:

-   **`simulator/`**: A high-fidelity JAX-based quadrotor dynamics simulator.
-   **`processing/`**: Scripts for processing raw ROS bag files into CSV/Parquet formats used by this repo.
-   **`wheels/`**: Pre-built wheels for PyTorch3D (check if they match your system).
