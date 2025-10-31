import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import FormatStrFormatter

import os, sys

# find the folder that contains "idsia_mpc"
current_dir = os.path.dirname(os.path.abspath(__file__))
while True:
    if os.path.exists(os.path.join(current_dir, "idsia_mpc")):
        PROJECT_ROOT = current_dir
        break
    parent = os.path.dirname(current_dir)
    if parent == current_dir:  # reached filesystem root
        raise RuntimeError("Could not find project root containing 'idsia_mpc'")
    current_dir = parent

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("Using project root:", PROJECT_ROOT)

from idsia_mpc.control_engineering_practice.plot_utils import setup_matplotlib
from idsia_mpc.control_engineering_practice.quat_utils import quat_to_euler
from idsia_mpc.control_engineering_practice.sysid.models import QuadLSTM

# === CONFIG ===
setup_matplotlib()
mode = "lstm"   # choose: "physics", "neural", "residual"

# --- Config ---
dt = 0.01
horizon = "full"

file_name = f'random1_20251017_run1'
df = pd.read_parquet(os.path.join('../../data/real/processed/new', f"{file_name}.parquet"))
df = df.rename(columns={"torch_yaw": "torque_yaw"})  # fix typo if present

model_path = f"../out/new/models/{mode}_quad_model_multistep_h100.pt"

state_cols = ["x","y","z","vx","vy","vz","qx","qy","qz","qw","wx","wy","wz"]
u_cols = ["thrust","torque_roll","torque_pitch","torque_yaw"]

X_logged = df[state_cols].to_numpy()
U_logged = df[u_cols].to_numpy()
t_logged = df["t"].to_numpy()

# --- Convert quaternions to Euler angles ---
quat_logged = X_logged[:, 6:10]
euler_logged = quat_to_euler(quat_logged)  # (N,3)
# Replace quaternion part with euler angles
X_logged_euler = np.hstack([X_logged[:, :6], euler_logged, X_logged[:, 10:]])  # shape (N, 12)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Initialize submodels ===
phys_params = {
    "g": 9.81,
    "m": 0.032,
    "J": np.diag([1.43e-6, 1.43e-6, 2.89e-6]),
    "thrust_to_weight": 2.0,
    "max_torque": np.array([1e-4, 1e-4, 3e-5]),
}

model = QuadLSTM().to(device)

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Loaded pretrained model from {model_path}")
except Exception as e:
    raise Exception("<UNK> Loaded pretrained model is not found.")

model.eval()

# === Rollout ===
X_torch_onestep = []
X_torch_multistep = []

with torch.no_grad():
    if horizon == "full":
        # Single full-trajectory forward pass
        x0 = torch.tensor(X_logged[0], dtype=torch.float32, device=device).unsqueeze(0)  # (1,13)
        u_seq_full = torch.tensor(U_logged[:-1], dtype=torch.float32, device=device).unsqueeze(0)  # (1,T-1,4)
        x_seq_pred = model(x0, u_seq_full)  # (1, T-1, 13)
        X_torch_multistep.append(x_seq_pred.squeeze(0).cpu().numpy())
        X_torch_onestep = [x_seq_pred[0, 0, :].cpu().numpy()]  # optional one-step for consistency

    else:
        # Standard fixed-horizon evaluation
        for k in range(len(U_logged) - horizon):
            # --- One-step prediction ---
            x0 = torch.tensor(X_logged[k], dtype=torch.float32, device=device).unsqueeze(0)
            u_seq = torch.tensor(U_logged[k:k+1], dtype=torch.float32, device=device).unsqueeze(0)  # (1,1,4)
            x_pred_seq = model(x0, u_seq)  # output shape (1,1,13)
            x_pred = x_pred_seq[:, -1, :]
            X_torch_onestep.append(x_pred.squeeze(0).cpu().numpy())

            # --- Multi-step rollout ---
            x_j = torch.tensor(X_logged[k], dtype=torch.float32, device=device).unsqueeze(0)
            rollout = []
            max_h = min(horizon, len(U_logged) - k - 1)
            u_seq_full = torch.tensor(U_logged[k:k+max_h], dtype=torch.float32, device=device).unsqueeze(0)  # (1,H,4)
            x_seq_pred = model(x0, u_seq_full)  # (1,H,13)
            rollout.append(x_seq_pred.squeeze(0).cpu().numpy())
            X_torch_multistep.append(rollout[0])

X_torch_multistep = np.array(X_torch_multistep, dtype=object)

if horizon == "full":
    # unpack single rollout
    X_torch_multistep = X_torch_multistep[0]
    X_torch_onestep = X_torch_multistep[0:1]  # for consistency in later code

# --- Convert predicted quaternions to Euler ---
if horizon == "full":
    X_first_from_rollouts = np.asarray(X_torch_multistep, dtype=np.float64)
else:
    X_first_from_rollouts = np.stack(
        [np.asarray(roll[0], dtype=np.float64) for roll in X_torch_multistep],
        axis=0
    )

quat_pred = X_first_from_rollouts[:, 6:10]
euler_pred = quat_to_euler(quat_pred)
X_torch_euler = np.hstack([
    X_first_from_rollouts[:, :6],
    euler_pred,
    X_first_from_rollouts[:, 10:]
])

# --- True next states (aligned) ---
N = len(U_logged)
if horizon == "full":
    X_next_logged = X_logged_euler[1:N]        # x₁ … x_T
    U_used = U_logged[:-1]                     # u₀ … u_{T−1}
    t_vec = t_logged[1:N]
else:
    H = int(horizon)
    X_next_logged = X_logged_euler[1:N - H + 1]
    U_used = U_logged[:N - H]
    t_vec = t_logged[1:N - H + 1]

# --- Compute error ---
err_torch = X_next_logged - X_torch_euler

# =====================================================
# --- Detailed Error Metrics ---
# =====================================================
state_names = ['x','y','z','vx','vy','vz','roll','pitch','yaw','wx','wy','wz']

abs_err = np.abs(err_torch)
max_abs = abs_err.max(axis=0)
rmse = np.sqrt(np.mean(err_torch**2, axis=0))
mean_err = np.mean(err_torch, axis=0)
std_err = np.std(err_torch, axis=0)

overall_max = np.max(max_abs)
overall_rmse = np.sqrt(np.mean(err_torch**2))

print("\n=== 🔍 DETAILED ERROR METRICS (Multistep Model vs Logged) ===")
print("---------------------------------------------------")
print(f"{'State':<8} | {'Mean':>10} | {'Std':>10} | {'MaxAbs':>10} | {'RMSE':>10}")
print("-"*60)
for i, name in enumerate(state_names):
    print(f"{name:<8} | {mean_err[i]:>10.3e} | {std_err[i]:>10.3e} | {max_abs[i]:>10.3e} | {rmse[i]:>10.3e}")
print("-"*60)
print(f"{'OVERALL':<8} | {'':>10} | {'':>10} | {overall_max:>10.3e} | {overall_rmse:>10.3e}")
print("---------------------------------------------------\n")

# === Consistency Check: Multistep vs Onestep ===
if horizon == "full":
    # in "full" mode there's only one trajectory, so compare first predicted step with itself
    first_from_rollouts = X_torch_multistep[0:1, :]  # (1,13)
else:
    # in standard mode, take the first step of each rollout
    first_from_rollouts = np.array([roll[0] for roll in X_torch_multistep])
# ensure both are 2D before subtraction
X_torch_onestep = np.atleast_2d(X_torch_onestep)
first_from_rollouts = np.atleast_2d(first_from_rollouts)

consistency_err = np.abs(X_torch_onestep - first_from_rollouts)
print(f"🔧 Consistency Check: mean diff = {consistency_err.mean():.3e}, max diff = {consistency_err.max():.3e}")

# =====================================================
# --- Summary Table Figure ---
# =====================================================
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('off')

table_data = [["State", "Mean", "Std", "MaxAbs", "RMSE"]] + \
    [[state_names[i],
      f"{mean_err[i]:.3e}",
      f"{std_err[i]:.3e}",
      f"{max_abs[i]:.3e}",
      f"{rmse[i]:.3e}"] for i in range(len(state_names))] + \
    [["OVERALL", "", "", f"{overall_max:.3e}", f"{overall_rmse:.3e}"]]

table = ax.table(cellText=table_data, loc='center')
table.scale(1, 1.4)
plt.tight_layout()
plt.show()

# =====================================================
# --- Simplified Full-Trajectory Comparison ---
# =====================================================
print("\n=== 🧭 Simplifying: Full Trajectory Simulation ===")

# True (logged) and predicted (model) trajectories are already in Euler form
x = np.asarray(X_logged_euler[1:N], dtype=np.float64)     # ground truth (aligned)
x_pred = np.asarray(X_torch_euler, dtype=np.float64)      # prediction

assert x.shape == x_pred.shape, f"Shape mismatch: x={x.shape}, x_pred={x_pred.shape}"
print(f"✅ Using full trajectory simulation: {x.shape[0]} timesteps, {x.shape[1]} states")

# =====================================================
# --- Save Dataset ---
# =====================================================
out_dir = f"../out/predictions/real/new/{mode}_model_multistep"
os.makedirs(out_dir, exist_ok=True)

data = {"t": t_vec}
for j, name in enumerate(state_names):
    data[name] = x[:, j]
    data[f"{name}_pred"] = x_pred[:, j]

df_pred = pd.DataFrame(data)
csv_path = f"{out_dir}/{file_name}.csv"
parquet_path = f"{out_dir}/{file_name}.parquet"

df_pred.to_csv(csv_path, index=False)
df_pred.to_parquet(parquet_path, index=False)

print(f"✅ Saved full-trajectory prediction dataset to:")
print(f"   → {csv_path}")
print(f"   → {parquet_path}")


# =====================================================
# --- Comparison Plot: Logged vs Predictions ---
# =====================================================
fig, axs = plt.subplots(4, 4, figsize=(20, 10), sharex=True, dpi=200)

state_labels = [
    ['$x$ [m]', '$y$ [m]', '$z$ [m]'],
    [r'$\varphi$ [rad]', r'$\theta$ [rad]', r'$\psi$ [rad]'],
    [r'$v_x$ [m/s]', r'$v_y$ [m/s]', r'$v_z$ [m/s]'],
    [r'$\omega_x$ [rad/s]', r'$\omega_y$ [rad/s]', r'$\omega_z$ [rad/s]']
]
state_indices = [
    [0, 1, 2],
    [6, 7, 8],
    [3, 4, 5],
    [9, 10, 11]
]

for r in range(4):
    for c in range(3):
        idx = state_indices[r][c]
        ax = axs[r, c]
        ax.plot(t_vec, X_next_logged[:, idx], color='tab:blue', label="logged")
        ax.plot(t_vec, X_torch_euler[:, idx], color='tab:orange', linestyle='--', label="multistep pred")
        ax.set_ylabel(state_labels[r][c])
        ax.grid(True)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

input_cols = ['thrust', 'torque_roll', 'torque_pitch', 'torque_yaw']
input_labels = [r'$T$ [N]', r'$\tau_{\varphi}$ [Nm]', r'$\tau_{\theta}$ [Nm]', r'$\tau_{\psi}$ [Nm]']

for r in range(4):
    ax = axs[r, 3]
    col = input_cols[r]
    if col in df.columns:
        ax.plot(t_logged, df[col], color='tab:purple', label="real input")
    ax.set_ylabel(input_labels[r])
    ax.grid(True)

fig.text(0.5, 0.04, "Time [s]", ha='center', va='center', fontsize=14)
fig.legend(*axs[0, 0].get_legend_handles_labels(), loc='upper center', ncols=4, bbox_to_anchor=(0.5, 1.02))
plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.15, wspace=0.35)
plt.show()

# =====================================================
# --- Error Figure (4x3 grid) ---
# =====================================================
fig, axs = plt.subplots(4, 3, figsize=(16, 8), sharex=True, dpi=200)
state_indices = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]

for r in range(4):
    for c in range(3):
        idx = state_indices[r][c]
        ax = axs[r, c]
        ax.plot(t_vec, err_torch[:, idx], color='tab:red', label='error')
        ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
        ax.set_ylabel(state_labels[r][c])
        ax.grid(True)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3e'))

fig.text(0.5, 0.04, "Time [s]", ha='center', va='center', fontsize=14)
fig.legend(*axs[0, 0].get_legend_handles_labels(), loc='upper center', ncols=3, bbox_to_anchor=(0.5, 1.01))
plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.2, wspace=0.3)
plt.show()
