import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import FormatStrFormatter
from idsia_mpc.control_engineering_practice.quat_utils import quat_to_euler
from idsia_mpc.control_engineering_practice.plot_utils import setup_matplotlib

from idsia_mpc.control_engineering_practice.sysid.dataset import QuadDataset
from idsia_mpc.control_engineering_practice.sysid.models import PhysQuadModel

# --- Config ---
setup_matplotlib()
dt = 0.01
type = "real"
traj_id = '1848'
traj_type = "random"
file_name_sim = f'experiment_points_{traj_id}.parquet'
file_name_real = f'experiment_{traj_type}_{traj_id}.parquet'
# file_name = 'experiment_random_1848.parquet'

# --- Load parquet ---
df_sim = pd.read_parquet(os.path.join('../../data/sim/', file_name_sim))
df_real = pd.read_parquet(os.path.join('../../data/real/processed/', file_name_real))

if type == "real":
    df = df_real
    df = df.rename(columns={"torch_yaw": "torque_yaw"})  # fix typo if present
elif type == "sim":
    df = df_sim

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

# --- Torch model roll-out ---
torch_params = {
    "g": 9.81,
    "m": 0.032,
    "J": np.diag([1.43e-6, 1.43e-6, 2.89e-6]),
    "thrust_to_weight": 2.0,
    "max_torque": np.array([1e-4, 1e-4, 3e-5]),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PhysQuadModel(torch_params, dt).to(device)
model.eval()

X_torch = []
with torch.no_grad():
    for k in range(len(U_logged) - 1):
        x_k = torch.tensor(X_logged[k], dtype=torch.float32, device=device).unsqueeze(0)
        u_k = torch.tensor(U_logged[k], dtype=torch.float32, device=device).unsqueeze(0)
        x_pred, _ = model(x_k, u_k)
        X_torch.append(x_pred.squeeze(0).cpu().numpy())
X_torch = np.array(X_torch)

# --- Convert predicted quaternions to Euler ---
quat_pred = X_torch[:, 6:10]
euler_pred = quat_to_euler(quat_pred)
X_torch_euler = np.hstack([X_torch[:, :6], euler_pred, X_torch[:, 10:]])  # shape (N-1, 12)

# --- True next states from log ---
X_next_logged = X_logged_euler[1:]  # match rollouts
U_used = U_logged[:-1]               # same length as predictions
t_vec = t_logged[1:]                 # time vector matching X_next_logged

# --- Errors ---
err_torch = X_next_logged - X_torch_euler

# =====================================================
# --- Compute and display error metrics ---
# =====================================================
state_names = ['x','y','z','vx','vy','vz','roll','pitch','yaw','wx','wy','wz']

abs_err = np.abs(err_torch)
max_abs = abs_err.max(axis=0)
rmse = np.sqrt(np.mean(err_torch**2, axis=0))
mean_err = np.mean(err_torch, axis=0)
std_err = np.std(err_torch, axis=0)

overall_max = np.max(max_abs)
overall_rmse = np.sqrt(np.mean(err_torch**2))

print("\n=== 🔍 DETAILED ERROR METRICS (Torch vs Logged) ===")
print("---------------------------------------------------")
print(f"{'State':<8} | {'Mean':>10} | {'Std':>10} | {'MaxAbs':>10} | {'RMSE':>10}")
print("-"*60)
for i, name in enumerate(state_names):
    print(f"{name:<8} | {mean_err[i]:>10.3e} | {std_err[i]:>10.3e} | {max_abs[i]:>10.3e} | {rmse[i]:>10.3e}")
print("-"*60)
print(f"{'OVERALL':<8} | {'':>10} | {'':>10} | {overall_max:>10.3e} | {overall_rmse:>10.3e}")
print("---------------------------------------------------\n")

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
# SAVE PREDICTION DATASET
# =====================================================
state_names = ['x','y','z','vx','vy','vz','roll','pitch','yaw','wx','wy','wz']

df_pred = pd.DataFrame({
    "t": t_vec
})
# Add logged & predicted states
for i, name in enumerate(state_names):
    df_pred[f"{name}"] = X_next_logged[:, i]
    df_pred[f"{name}_pred"] = X_torch_euler[:, i]
# Add control inputs
for i, name in enumerate(u_cols):
    df_pred[name] = U_used[:, i]

# --- Ensure output folder exists ---
out_dir = f"../out/predictions/{type}/physics_model_onestep"
os.makedirs(out_dir, exist_ok=True)

# --- Save prediction dataset ---
csv_path = f"{out_dir}/experiment_{traj_type}_{traj_id}.csv"
parquet_path = f"{out_dir}/experiment_{traj_type}_{traj_id}.parquet"

df_pred.to_csv(csv_path, index=False)
df_pred.to_parquet(parquet_path, index=False)

print(f"✅ Saved prediction dataset to:")
print(f"   → {csv_path}")
print(f"   → {parquet_path}")


from matplotlib.ticker import FormatStrFormatter

# --- Create 4x4 layout ---
fig, axs = plt.subplots(4, 4, figsize=(20, 10), sharex=True, dpi=200)

# --- Define state variables (first 3 columns) ---
state_labels = [
    ['$x$ [m]', '$y$ [m]', '$z$ [m]'],
    [r'$\varphi$ [rad]', r'$\theta$ [rad]', r'$\psi$ [rad]'],
    [r'$v_x$ [m/s]', r'$v_y$ [m/s]', r'$v_z$ [m/s]'],
    [r'$\omega_x$ [rad/s]', r'$\omega_y$ [rad/s]', r'$\omega_z$ [rad/s]']
]
state_indices = [
    [0, 1, 2],   # position
    [6, 7, 8],   # euler angles
    [3, 4, 5],   # linear velocities
    [9, 10, 11]  # angular velocities
]

# --- Inputs (last column) ---
input_cols = ['thrust', 'torque_roll', 'torque_pitch', 'torque_yaw']
input_labels = [r'$T$ [N]', r'$\tau_{\varphi}$ [Nm]', r'$\tau_{\theta}$ [Nm]', r'$\tau_{\psi}$ [Nm]']

# --- Plot states ---
for r in range(4):
    for c in range(3):
        idx = state_indices[r][c]
        ax = axs[r, c]

        ax.plot(t_vec, X_next_logged[:, idx], color='tab:blue', label="logged")
        ax.plot(t_vec, X_torch_euler[:, idx], color='tab:orange', linestyle='--', label="torch pred")

        ax.set_ylabel(state_labels[r][c])
        ax.grid(True)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# --- Plot inputs (last column) ---
for r in range(4):
    col = input_cols[r]
    ax = axs[r, 3]

    # keep color scheme consistent: purple/green for inputs
    if col in df.columns:
        ax.plot(t_logged, df[col], color='tab:purple', label="real input")
        ax.plot(t_logged, df[col], color='tab:green', linestyle='--', label="sim input")

    ax.set_ylabel(input_labels[r])
    ax.grid(True)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# --- Shared labels ---
fig.text(0.5, 0.04, "Time [s]", ha='center', va='center', fontsize=14)

# --- Collect all unique legend entries ---
handles, labels = [], []
for ax in axs.flat:
    h, l = ax.get_legend_handles_labels()
    for handle, label in zip(h, l):
        if label not in labels:
            handles.append(handle)
            labels.append(label)

# --- Combined legend for all (including inputs) ---
fig.legend(handles, labels, loc='upper center', ncols=6, bbox_to_anchor=(0.5, 1.01))

# --- Layout tweaks for alignment and readability ---
plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.1, wspace=0.35)
plt.show()


# =====================================================
# --- Error Figure: 4 x 3 (no inputs) ---
# =====================================================

state_labels = [
    ['$x$ [m]', '$y$ [m]', '$z$ [m]'],
    [r'$\varphi$ [rad]', r'$\theta$ [rad]', r'$\psi$ [rad]'],
    [r'$v_x$ [m/s]', r'$v_y$ [m/s]', r'$v_z$ [m/s]'],
    [r'$\omega_x$ [rad/s]', r'$\omega_y$ [rad/s]', r'$\omega_z$ [rad/s]']
]
state_indices = [
    [0, 1, 2],   # position
    [3, 4, 5],   # euler
    [6, 7, 8],   # linear velocities
    [9, 10, 11]  # angular velocities
]

fig, axs = plt.subplots(4, 3, figsize=(16, 8), sharex=True, dpi=200)

for r in range(4):
    for c in range(3):
        idx = state_indices[r][c]
        ax = axs[r, c]
        ax.plot(t_vec, err_torch[:, idx], color='tab:red', label='error')
        ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
        ax.set_ylabel(state_labels[r][c])
        ax.grid(True)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))

fig.text(0.5, 0.04, "Time [s]", ha='center', va='center', fontsize=14)

# shared legend
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncols=3, bbox_to_anchor=(0.5, 1.01))

plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.2, wspace=0.3)
plt.show()