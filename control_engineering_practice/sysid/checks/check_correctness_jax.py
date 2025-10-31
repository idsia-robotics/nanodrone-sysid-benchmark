import numpy as np
import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt

from quadrotor_sys import quad_dynamics

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

X_logged = df[state_cols].to_numpy()   # shape (N,13)
U_logged = df[u_cols].to_numpy()       # shape (N,4)
t_logged = df["t"].to_numpy()

# --- Per-step consistency check ---
errors = []
for k in range(len(U_logged) - 1):  # last row has no "next state" in parquet
    x_k = X_logged[k]
    u_k = U_logged[k]
    # Predict next state using dynamics
    x_pred = x_k + dt * np.array(quad_dynamics(x_k, u_k, sys_params))
    # Compare with logged next state
    diff = X_logged[k+1] - x_pred
    errors.append(diff)

errors = np.array(errors)

# --- Stats ---
max_err = np.max(np.abs(errors))
rmse = np.sqrt(np.mean(errors**2))

print("Max abs error:", max_err)
print("RMSE:", rmse)

print("\nPer-dimension max errors:")
for i, name in enumerate(state_cols):
    print(f"{name:>4s}: {np.max(np.abs(errors[:, i])):.3e}")

# --- Plots ---
time_err = t_logged[1:len(errors)+1]  # align times with error steps

fig, axs = plt.subplots(len(state_cols), 1, figsize=(10, 20), sharex=True)
for i, name in enumerate(state_cols):
    axs[i].plot(time_err, errors[:, i], label=f"error {name}")
    axs[i].axhline(0, color="k", linestyle="--", linewidth=0.5)
    axs[i].set_ylabel(name)
    axs[i].legend()
axs[-1].set_xlabel("time [s]")
fig.suptitle("Per-step state prediction errors")
plt.tight_layout()
plt.show()

# Combined error norm
err_norm = np.linalg.norm(errors, axis=1)
plt.figure(figsize=(10,4))
plt.plot(time_err, err_norm, label="||error||")
plt.xlabel("time [s]")
plt.ylabel("state error norm")
plt.title("Overall error norm per step")
plt.legend()
plt.show()
