# Quadrotor Control Benchmark & Identification

This repository contains benchmarks and identification methods for quadrotor control, serving as a support resource for our paper submitted to the **Special Issue on Machine Learning and Control Engineering**.

## Overview

Control engineering is evolving rapidly with the integration of machine learning algorithms. However, benchmarking and comparing these algorithms remains a challenge due to a lack of shared code and high-fidelity models. This project addresses this gap by providing:

- A set of **challenging benchmark control applications**.
- High-fidelity **simulation models** (JAX-based) and **identification models** (PyTorch-based).
- **Reference implementations** for system identification using Physics-based, Neural, and Hybrid (Residual) approaches.
- Tools for data processing and visualization.

The goal is to facilitate reproduction of results and enable fair comparisons between state-of-the-art control and identification methods.

## Repository Structure

```
├── simulator/          # JAX-based quadrotor dynamics simulator
│   ├── quadrotor_sys.py
│   └── check_jax_torch_quad_model_equivalence.py
├── identification/     # PyTorch-based system identification models & training
│   ├── models.py       # Model architectures (Phys, Neural, Residual, LSTM)
│   ├── dataset.py      # Data loading and preprocessing
│   ├── losses.py       # Custom loss functions
│   └── train/          # Training scripts
├── data/               # Directory for datasets (train/test split expected)
├── processing/         # Data processing scripts (bag files to CSV, EDA)
├── wheels/             # Custom built wheels (e.g., PyTorch3D)
├── utils/              # Utility functions
└── requirements.txt    # Python dependencies
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

This project uses `pytorch3d` for 3D transformations (quaternions, etc.). Installing it can sometimes be tricky depending on your OS and CUDA version.

**Option A: Use provided wheels (if applicable)**
Check the `wheels/` directory or visit the [PyTorch3D Wheel Builder](https://miropsota.github.io/torch_packages_builder/pytorch3d/) to download the correct wheel for your system. Then install it:

```bash
pip install wheels/pytorch3d-*.whl
# OR
pip install <url_to_wheel>
```

**Option B: Build from source**
Follow the official [PyTorch3D installation guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).

## Usage

### 1. Data Preparation

Raw data typically comes in ROS bag format. Use the scripts in `processing/` to convert them to CSV files suitable for training.

```bash
# Example: Convert bag to CSV
python processing/bag_to_csv.py
```

Processed data should be placed in `data/real/processed/train/` and `data/real/processed/test/`. The filename convention expected by training scripts is `{trajectory_type}_20251017_run{run_number}.csv`.

### 2. System Identification (Training)

The `identification/` module supports training various models to identify the quadrotor dynamics.

To train an LSTM model on specific trajectories:

```bash
python identification/train/train_lstm.py --train_trajs '["random", "square", "chirp"]' --epochs 5000
```

**Arguments:**
- `--train_trajs`: JSON string list of trajectory types to use for training.
- `--device`: Compute device (e.g., `cuda:0` or `cpu`).
- `--epochs`: Number of training epochs.
- `--horizon`: Prediction horizon for the model.

### 3. Simulation & Verification

The project includes a JAX-based high-fidelity simulator in `simulator/quadrotor_sys.py`.

To verify the equivalence between the JAX simulator and the PyTorch identification models:

```bash
python simulator/check_jax_torch_quad_model_equivalence.py
```

## Models Description

The `identification/models.py` file contains implementations of several model types:

1.  **`PhysQuadModel`**: A physics-based model using rigid body dynamics and RK4 integration. It uses `pytorch3d` for quaternion operations.
2.  **`NeuralQuadModel`**: A purely data-driven MLP (Multi-Layer Perceptron) model that predicts state updates.
3.  **`ResidualQuadModel`**: A hybrid model that combines `PhysQuadModel` as a baseline and uses a `NeuralQuadModel` to learn the residual dynamics (physics errors).
4.  **`QuadLSTM`**: An LSTM-based model suitable for capturing temporal dependencies and memory effects in the dynamics.

## Citation

If you use this code in your research, please cite the accompanying paper (details to be added upon publication).
