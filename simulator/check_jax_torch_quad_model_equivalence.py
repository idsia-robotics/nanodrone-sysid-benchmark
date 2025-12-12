import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Ensure the project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from identification.models import PhysQuadModel
from simulator.quadrotor_sys import quad_dynamics
from simulator.utils.solvers import simulate_rollout, step_dynamics_rk4
from scipy.spatial.transform import Rotation as R

def main():
    print("Checking JAX vs PyTorch model equivalence...")

    # === Config ===
    dt = 0.01

    # === Initialize model ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # === Dynamics parameters ===
    phys_params = {
        "g": 9.81,
        "m": 0.045,
        "J": np.diag([2.3951e-5, 2.3951e-5, 3.2347e-5]),
        "thrust_to_weight": 2.0,
        "max_torque": np.array([1e-2, 1e-2, 3e-3]),
    }

    model = PhysQuadModel(phys_params, dt).to(device)

    # === Load Data ===
    # Path relative to this script: ../data/debug/figure8-mellinger/mellinger/ctrl_normalized.csv
    data_path = os.path.join(os.path.dirname(__file__), "../data/debug/figure8-mellinger/mellinger/ctrl_normalized.csv")

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    df = pd.read_csv(data_path).dropna()

    # Initial state
    x0 = np.zeros(13)
    x0[9] = 1.0 # qw = 1 (identity quaternion)

    # Controls from data
    u = df.values[:, 1:] # [T, tau_x, tau_y, tau_z] (normalized)

    print("Running JAX rollout...")
    # JAX Rollout
    # simulate_rollout expects u to be numpy array
    X = simulate_rollout(x0, u, dt, quad_dynamics, phys_params, step_fn=step_dynamics_rk4)[:-1]

    pos_jax   = X[:, 0:3]
    vel_jax   = X[:, 3:6]
    quat_jax  = X[:, 6:10]
    omega_jax = X[:, 10:13]

    time = np.arange(X.shape[0]) * dt

    # Convert JAX quaternions to Euler
    r_jax = R.from_quat(quat_jax)   # expects [x,y,z,w]
    euler_jax = r_jax.as_euler('xyz', degrees=False)

    print("Running PyTorch rollout...")
    # PyTorch Rollout
    x_all = []
    x = torch.tensor(x0, dtype=torch.float32, device=device).unsqueeze(0)  # (1,13)

    for i in range(df.shape[0]):
        x_all.append(x.clone())
        u_phys = torch.tensor(df.values[i, 1:], dtype=torch.float32, device=device).unsqueeze(0)
        x = model._step_from_phys(x, u_phys)

    X_torch = torch.cat(x_all, dim=0).cpu().numpy()

    pos_t   = X_torch[:, 0:3]
    vel_t   = X_torch[:, 3:6]
    quat_t   = X_torch[:, 6:10]
    omega_t = X_torch[:, 10:13]

    # Convert PyTorch quats to Euler
    r_t = R.from_quat(quat_t)
    euler_t = r_t.as_euler('xyz', degrees=False)

    # === Verification ===
    # Calculate errors
    pos_err = np.linalg.norm(pos_jax - pos_t, axis=1).mean()
    vel_err = np.linalg.norm(vel_jax - vel_t, axis=1).mean()
    print(f"Mean Position Error: {pos_err:.6f}")
    print(f"Mean Velocity Error: {vel_err:.6f}")

    if pos_err < 1e-4 and vel_err < 1e-4:
        print("SUCCESS: JAX and PyTorch models are equivalent.")
    else:
        print("WARNING: Models differ significantly.")

    # === Plotting ===
    fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # Position
    ax[0].plot(time, pos_jax[:,0], label='x JAX')
    ax[0].plot(time, pos_jax[:,1], label='y JAX')
    ax[0].plot(time, pos_jax[:,2], label='z JAX')
    ax[0].plot(time, pos_t[:,0], '--', label='x TORCH')
    ax[0].plot(time, pos_t[:,1], '--', label='y TORCH')
    ax[0].plot(time, pos_t[:,2], '--', label='z TORCH')
    ax[0].set_ylabel('Position [m]')
    ax[0].legend()
    ax[0].grid(True)

    # Velocity
    ax[1].plot(time, vel_jax[:,0], label='vx JAX')
    ax[1].plot(time, vel_jax[:,1], label='vy JAX')
    ax[1].plot(time, vel_jax[:,2], label='vz JAX')
    ax[1].plot(time, vel_t[:,0], '--', label='vx TORCH')
    ax[1].plot(time, vel_t[:,1], '--', label='vy TORCH')
    ax[1].plot(time, vel_t[:,2], '--', label='vz TORCH')
    ax[1].set_ylabel('Velocity [m/s]')
    ax[1].legend()
    ax[1].grid(True)

    # Euler angles
    ax[2].plot(time, euler_jax[:,0], label='roll JAX')
    ax[2].plot(time, euler_jax[:,1], label='pitch JAX')
    ax[2].plot(time, euler_jax[:,2], label='yaw JAX')
    ax[2].plot(time, euler_t[:,0], '--', label='roll TORCH')
    ax[2].plot(time, euler_t[:,1], '--', label='pitch TORCH')
    ax[2].plot(time, euler_t[:,2], '--', label='yaw TORCH')
    ax[2].set_ylabel('Euler angles [rad]')
    ax[2].legend()
    ax[2].grid(True)

    # Angular velocity
    ax[3].plot(time, omega_jax[:,0], label='wx JAX')
    ax[3].plot(time, omega_jax[:,1], label='wy JAX')
    ax[3].plot(time, omega_jax[:,2], label='wz JAX')
    ax[3].plot(time, omega_t[:,0], '--', label='wx TORCH')
    ax[3].plot(time, omega_t[:,1], '--', label='wy TORCH')
    ax[3].plot(time, omega_t[:,2], '--', label='wz TORCH')
    ax[3].set_ylabel('Angular vel [rad/s]')
    ax[3].set_xlabel('Time [s]')
    ax[3].legend()
    ax[3].grid(True)

    output_file = "equivalence_check.png"
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()
