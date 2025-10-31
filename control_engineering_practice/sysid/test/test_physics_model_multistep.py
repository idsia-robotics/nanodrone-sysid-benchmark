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
from idsia_mpc.control_engineering_practice.sysid.models import PhysQuadModel, QuadMultiStepModel

# === CONFIG ===
setup_matplotlib()
mode = "physics"   # choose: "physics", "neural", "residual"

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

phys_model = PhysQuadModel(phys_params, dt).to(device)

# === Combined multi-step model ===
model = QuadMultiStepModel(
    phys_model=phys_model,
    mode=mode
).to(device)

model.eval()

# === Rollout ===
X_torch_onestep = []
X_torch_multistep = []
with torch.no_grad():
    for k in range(len(U_logged) - horizon):
        x_k = torch.tensor(X_logged[k], dtype=torch.float32, device=device).unsqueeze(0)
        u_k = torch.tensor(U_logged[k], dtype=torch.float32, device=device).unsqueeze(0)
        x_pred = model.one_step(x_k, u_k)  # ✅ call one_step instead of forward
        X_torch_onestep.append(x_pred.squeeze(0).cpu().numpy())

        # Multi-step rollout
        x_j = x_k.clone()
        rollout = []
        max_h = min(horizon, len(U_logged) - k - 1)
        for j in range(max_h):
            u_j = torch.tensor(U_logged[k + j], dtype=torch.float32, device=device).unsqueeze(0)
            x_j = model.one_step(x_j, u_j)  # ✅ this ensures correct single-step behavior
            rollout.append(x_j.squeeze(0).cpu().numpy())
        X_torch_multistep.append(np.stack(rollout, axis=0))

X_torch_onestep = np.array(X_torch_onestep)
X_torch_multistep = np.array(X_torch_multistep, dtype=object)

# --- Convert predicted quaternions to Euler ---
X_first_from_rollouts = np.stack(
    [np.asarray(roll[0], dtype=np.float64) for roll in X_torch_multistep],
    axis=0
)
quat_pred = X_first_from_rollouts[:, 6:10]
euler_pred = quat_to_euler(quat_pred)

# Replace quaternion part with Euler angles
X_torch_euler = np.hstack([
    X_first_from_rollouts[:, :6],
    euler_pred,
    X_first_from_rollouts[:, 10:]
])  # shape (N-1, 12)

# --- True next states from log ---
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

# === Consistency Check: Multistep vs Onestep ===
first_from_rollouts = np.array([roll[0] for roll in X_torch_multistep])
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
# --- Save Prediction Dataset ---
# =====================================================
state_names = ['x','y','z','vx','vy','vz','roll','pitch','yaw','wx','wy','wz']
# =====================================================
# --- Save Multistep Predictions (optimized) ---
# =====================================================
out_dir = f"../out/predictions/{data_type}/{mode}_model_multistep"
os.makedirs(out_dir, exist_ok=True)

data = {"t": t_vec}

# --- Add logged and 1-step predicted states ---
for i, name in enumerate(state_names):
    data[name] = X_next_logged[:, i]
    data[f"{name}_pred"] = X_torch_euler[:, i]

# --- Prepare multistep rollouts (convert quat → euler for each horizon) ---
max_h = horizon
num_states = 12  # x,y,z,vx,vy,vz,roll,pitch,yaw,wx,wy,wz
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

# quick sanity check
print(f"✅ filled rollouts: {np.sum(np.isfinite(rollout_array[:,:,0]))} / {rollout_array.shape[0]*rollout_array.shape[1]}")


# --- Add multistep columns efficiently ---
for h in range(max_h):
    for j, name in enumerate(state_names):
        col_name = f"{name}_pred_h{h+1}"
        data[col_name] = rollout_array[:, h, j]

# --- Create DataFrame in one go ---
df_pred = pd.DataFrame(data).copy()  # copy() avoids fragmentation warning

# --- Save ---
csv_path = f"{out_dir}/experiment_{traj_type_test}_{traj_id_test}.csv"
parquet_path = f"{out_dir}/experiment_{traj_type_test}_{traj_id_test}.parquet"

df_pred.to_csv(csv_path, index=False)
df_pred.to_parquet(parquet_path, index=False)

print(f"✅ Saved multistep prediction dataset to:")
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
