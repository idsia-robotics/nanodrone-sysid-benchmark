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
from idsia_mpc.control_engineering_practice.sysid.models import NeuralQuadModel, QuadMultiStepModel

# === CONFIG ===
setup_matplotlib()
mode = "neural"   # choose: "physics", "neural", "residual"

# --- Config ---
dt = 0.01
horizon = 50
data_type = "real"   # renamed from 'type' to avoid shadowing built-in
traj_id_train = '1848'
traj_id_test = '1809'

if data_type == "sim":
    traj_type_test = "points"
    file_name_sim = f'experiment_points_{traj_id_test}.parquet'
    df = pd.read_parquet(os.path.join('../../data/sim/', file_name_sim))
elif data_type == "real":
    traj_type_test = "melon" if traj_id_test == "1809" else "random"
    traj_type_train = "random" if traj_id_train == "1848" else "melon"
    file_name_real = f'experiment_{traj_type_test}_{traj_id_test}.parquet'
    df = pd.read_parquet(os.path.join('../../data/real/processed/', file_name_real))
    df = df.rename(columns={"torch_yaw": "torque_yaw"})  # fix typo if present

model_path = f"../out/models/{mode}_quad_model_multistep_{data_type}_{traj_type_train}_{traj_id_train}.pt"

# === Columns ===
state_cols = ["x","y","z","vx","vy","vz","qx","qy","qz","qw","wx","wy","wz"]
u_cols = ["thrust","torque_roll","torque_pitch","torque_yaw"]

X_logged = df[state_cols].to_numpy()
U_logged = df[u_cols].to_numpy()
t_logged = df["t"].to_numpy()

# --- Convert quaternions to Euler angles ---
quat_logged = X_logged[:, 6:10]
euler_logged = quat_to_euler(quat_logged)  # (N,3)
X_logged_euler = np.hstack([X_logged[:, :6], euler_logged, X_logged[:, 10:]])  # (N,12)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Model loading ===
neural_model = NeuralQuadModel(dt).to(device)
model = QuadMultiStepModel(neural_model=neural_model, mode=mode).to(device)

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Loaded pretrained model from {model_path}")
except Exception:
    raise FileNotFoundError(f"❌ Model not found at {model_path}")

model.eval()

# === Rollout (aligned with dataset logic) ===
X_torch_onestep = []
X_torch_multistep = []

with torch.no_grad():
    for k in range(len(U_logged) - horizon):
        # One-step
        x_k = torch.tensor(X_logged[k], dtype=torch.float32, device=device).unsqueeze(0)
        u_k = torch.tensor(U_logged[k], dtype=torch.float32, device=device).unsqueeze(0)
        x_pred = model.one_step(x_k, u_k)
        X_torch_onestep.append(x_pred.squeeze(0).cpu().numpy())

        # Multi-step rollout (same as training dataset)
        x_j = x_k.clone()
        rollout = []
        for j in range(horizon):
            u_j = torch.tensor(U_logged[k + j], dtype=torch.float32, device=device).unsqueeze(0)
            x_j = model.one_step(x_j, u_j)
            rollout.append(x_j.squeeze(0).cpu().numpy())
        X_torch_multistep.append(np.stack(rollout, axis=0))

X_torch_onestep = np.stack(X_torch_onestep, axis=0)  # (N-h, 13)
# Keep X_torch_multistep as a list (not object array)

# --- Convert predicted quaternions to Euler ---
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
])  # (N-h, 12)

# --- True next states (aligned) ---
N = len(U_logged)
X_next_logged = X_logged_euler[1 : N - horizon + 1]
U_used = U_logged[: N - horizon]
t_vec = t_logged[1 : N - horizon + 1]

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

# === Consistency Check ===
first_from_rollouts = np.stack([r[0] for r in X_torch_multistep], axis=0)
consistency_err = np.abs(X_torch_onestep - first_from_rollouts)
print(f"🔧 Consistency Check: mean diff = {consistency_err.mean():.3e}, max diff = {consistency_err.max():.3e}")

# =====================================================
# --- Save multistep predictions ---
# =====================================================
out_dir = f"../out/predictions/{data_type}/{mode}_model_multistep"
os.makedirs(out_dir, exist_ok=True)

data = {"t": t_vec}
for i, name in enumerate(state_names):
    data[name] = X_next_logged[:, i]
    data[f"{name}_pred"] = X_torch_euler[:, i]

# --- Convert and fill rollouts ---
max_h = horizon
num_states = 12
rollout_array = np.full((len(X_torch_multistep), max_h, num_states), np.nan)

for i, roll in enumerate(X_torch_multistep):
    if roll is None or len(roll) == 0:
        continue
    roll = np.asarray(roll, dtype=np.float64)
    h = min(len(roll), max_h)
    quat_seq = roll[:h, 6:10]
    euler_seq = quat_to_euler(quat_seq)
    if euler_seq.ndim == 1:
        euler_seq = euler_seq[np.newaxis, :]
    roll_euler = np.hstack([roll[:h, :6], euler_seq, roll[:h, 10:]])
    rollout_array[i, :h, :] = roll_euler

print(f"✅ filled rollouts: {np.sum(np.isfinite(rollout_array[:,:,0]))} / {rollout_array.shape[0]*rollout_array.shape[1]}")

for h in range(max_h):
    for j, name in enumerate(state_names):
        data[f"{name}_pred_h{h+1}"] = rollout_array[:, h, j]

df_pred = pd.DataFrame(data)
csv_path = f"{out_dir}/experiment_{traj_type_test}_{traj_id_test}.csv"
parquet_path = f"{out_dir}/experiment_{traj_type_test}_{traj_id_test}.parquet"
df_pred.to_csv(csv_path, index=False)
df_pred.to_parquet(parquet_path, index=False)
print(f"✅ Saved multistep prediction dataset:\n   → {csv_path}\n   → {parquet_path}")


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
