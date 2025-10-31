import numpy as np
import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt
import torch

from quadrotor_sys import quad_dynamics
from dataset import QuadDataset
from models import PhysQuadModel

# --- Config ---
dt = 0.01
sys_params = {
    "g": 9.81,
    "m": 0.032,
    "J": np.diag([1.43e-6, 1.43e-6, 2.89e-6]),
    "thrust_to_weight": 2.0,
    "max_torque": jnp.array([1e-4, 1e-4, 3e-5]),
}

# --- Load parquet ---
df = pd.read_parquet("../../data/sim/experiment_points_1809.parquet")

state_cols = ["x","y","z","vx","vy","vz","qx","qy","qz","qw","wx","wy","wz"]
u_cols = ["thrust","torque_roll","torque_pitch","torque_yaw"]

X_logged = df[state_cols].to_numpy()   # (N,13)
U_logged = df[u_cols].to_numpy()       # (N,4)
t_logged = df["t"].to_numpy()

# --- Re-simulate with JAX dynamics ---
X_resim = []
for k in range(len(U_logged)-1):
    x_k = X_logged[k]
    u_k = U_logged[k]
    x_pred = x_k + dt * np.array(quad_dynamics(x_k, u_k, sys_params))
    X_resim.append(x_pred)
X_resim = np.array(X_resim)

# --- Torch model roll-out ---
torch_params = {
    "g": 9.81,
    "m": 0.032,
    "J": np.diag([1.43e-6, 1.43e-6, 2.89e-6]),
    "thrust_to_weight": 2.0,
    "max_torque": np.array([1e-4, 1e-4, 3e-5]),
}
model = PhysQuadModel(torch_params, dt)
model.eval()

X_torch = []
with torch.no_grad():
    for k in range(len(U_logged)-1):
        x_k = torch.tensor(X_logged[k], dtype=torch.float32).unsqueeze(0)
        u_k = torch.tensor(U_logged[k], dtype=torch.float32).unsqueeze(0)
        x_pred, _ = model(x_k, u_k)
        X_torch.append(x_pred.squeeze(0).numpy())
X_torch = np.array(X_torch)

# --- True next states from log ---
X_next_logged = X_logged[1:]   # states after applying U[k]

# --- Errors ---
err_resim = X_next_logged - X_resim
err_torch = X_next_logged - X_torch

print("=== Stats ===")
print("Resim vs logged  | Max:", np.max(np.abs(err_resim)), "RMSE:", np.sqrt(np.mean(err_resim**2)))
print("Torch vs logged  | Max:", np.max(np.abs(err_torch)), "RMSE:", np.sqrt(np.mean(err_torch**2)))

# --- Plots per state variable ---
time_axis = t_logged[1:]  # matches X_next_logged

fig, axs = plt.subplots(len(state_cols), 1, figsize=(10, 20), sharex=True)
for i, name in enumerate(state_cols):
    axs[i].plot(time_axis, X_next_logged[:, i], label="logged", color="k")
    axs[i].plot(time_axis, X_resim[:, i], "--", label="resim (jax)", color="blue")
    axs[i].plot(time_axis, X_torch[:, i], ":", label="torch", color="orange")
    axs[i].set_ylabel(name)
    axs[i].legend(loc="upper right", fontsize=8)
axs[-1].set_xlabel("time [s]")
fig.suptitle("Trajectory comparison: Logged vs Re-sim vs Torch")
plt.tight_layout()
plt.show()

# --- Error norms ---
plt.figure(figsize=(10,4))
plt.plot(time_axis, np.linalg.norm(err_resim, axis=1), label="||logged - resim||")
plt.plot(time_axis, np.linalg.norm(err_torch, axis=1), label="||logged - torch||")
plt.xlabel("time [s]")
plt.ylabel("state error norm")
plt.title("Error norms")
plt.legend()
plt.show()
