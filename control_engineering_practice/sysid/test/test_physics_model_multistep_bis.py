import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

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
from idsia_mpc.control_engineering_practice.sysid.models import QuadLSTM, PhysQuadModel, MotorsPhysQuadModel, \
    QuadMultiStepModel
from idsia_mpc.control_engineering_practice.sysid.dataset import QuadMultiStepDataset, combine_concat_dataset

# ======================================================
# === CONFIGURATION ====================================
# ======================================================
setup_matplotlib()

mode = "physics"     # "physics", "neural", or "residual"
fold = "test"     # "train", "valid", "test"
scale = True      # always True for inference
horizon = 1000
dt = 0.01

batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = f"../out/new/models/{mode}_quad_model_multistep_h100.pt"
pred_out_dir = f"../out/predictions/real/new/{mode}_model_multistep"
os.makedirs(pred_out_dir, exist_ok=True)

# ======================================================
# === LOAD DATASETS ====================================
# ======================================================
test_trajs = ["melon"]
test_ds = []
for traj in test_trajs:
    for run in [1,2,3,4,5]:
        try:
            file_name = f'{traj}_20251017_run{run}.parquet'
            df = pd.read_parquet(os.path.join('../../data/real/processed/new/train', file_name))
            df = df.rename(columns={"torch_yaw": "torque_yaw"})  # fix typo if present
            ds = QuadMultiStepDataset(df, horizon=horizon, split='train', use_quaternions=True)
            test_ds.append(ds)
        except Exception as e:
            print(e)
            continue

test_dataset = combine_concat_dataset(ConcatDataset(test_ds), scale=False, fold="test", scaler_dir="/home/rbusetto/nanodrone-sysid-mpc/idsia_mpc/control_engineering_practice/sysid/train/scalers")
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ======================================================
# === LOAD MODEL =======================================
# ======================================================
# === Initialize submodels ===
# phys_params = {
#     "g": 9.81,
#     "m": 0.032,
#     "J": np.diag([1.43e-6, 1.43e-6, 2.89e-6]),
#     "thrust_to_weight": 2.0,
#     "max_torque": np.array([1e-4, 1e-4, 3e-5]),
# }

phys_params = {
    "g": 9.81,
    "m": 0.045,
    "J": np.diag([2.3951e-5, 2.3951e-5, 3.2347e-6]),
    "thrust_to_weight": 2.0,
    "max_torque": np.array([1e-2, 1e-2, 3e-3]),
}

phys_model = PhysQuadModel(phys_params, dt).to(device)
phys_motor_model = MotorsPhysQuadModel(phys_model)
model = QuadMultiStepModel(phys_motor_model, mode=mode).to(device)

print(f"✅ Loaded model: {model_path}")

# ======================================================
# === RUN PREDICTIONS =================================
# ======================================================
preds, trues = [], []

with torch.no_grad():
    for x0, u_seq, x_seq_true in test_loader:
        x0, u_seq = x0.to(device), u_seq.to(device)
        x_pred = model(x0, u_seq).cpu()
        preds.append(x_pred)
        trues.append(x_seq_true)

preds = torch.cat(preds, dim=0).numpy()
trues = torch.cat(trues, dim=0).numpy()

print(preds.shape)
print(trues.shape)

# ======================================================
# === DENORMALIZE ======================================
# ======================================================
scaler_dir = "/home/rbusetto/nanodrone-sysid-mpc/idsia_mpc/control_engineering_practice/sysid/train/scalers"
x_scaler_path = os.path.join(scaler_dir, "x_scaler.pkl")
x_scaler = joblib.load(x_scaler_path)

preds = x_scaler.inverse_transform(preds.reshape(-1, preds.shape[-1])).reshape(preds.shape)
trues = x_scaler.inverse_transform(trues.reshape(-1, trues.shape[-1])).reshape(trues.shape)

# This gives shape (H, 13)
preds_seq = preds[0]
trues_seq = trues[0]

# === Convert quaternion → Euler angles ===
euler_preds = quat_to_euler(preds_seq[:, 6:10])  # (H, 3)
euler_trues = quat_to_euler(trues_seq[:, 6:10])  # (H, 3)

# === Replace quaternion part with Euler ===
preds_euler = np.hstack([
    preds_seq[:, :3],     # x, y, z, vx, vy, vz
    euler_preds,          # roll, pitch, yaw
    preds_seq[:, 7:]     # wx, wy, wz
])  # shape (H, 12)

trues_euler = np.hstack([
    trues_seq[:, :3],
    euler_trues,
    trues_seq[:, 7:]
])  # shape (H, 12)

# === Final plotting arrays ===
preds_plot = preds_euler
trues_plot = trues_euler

print(preds_plot.shape, trues_plot.shape)

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# --- Create 4x4 layout ---
fig, axs = plt.subplots(4, 3, figsize=(20, 10), sharex=True, dpi=200)

# --- Define state variables (fill first 3 columns) ---
state_cols = [
    ['x', 'y', 'z'],
    ['roll', 'pitch', 'yaw'],
    ['vx', 'vy', 'vz'],
    ['wx', 'wy', 'wz']
]
state_labels = [
    ['$x$ [m]', '$y$ [m]', '$z$ [m]'],
    [r'$\varphi$ [rad]', r'$\theta$ [rad]', r'$\psi$ [rad]'],
    [r'$v_x$ [m/s]', r'$v_y$ [m/s]', r'$v_z$ [m/s]'],
    [r'$\omega_x$ [rad/s]', r'$\omega_y$ [rad/s]', r'$\omega_z$ [rad/s]']
]

# --- Define time vector (assume uniform dt) ---
dt = 0.01
t = np.arange(trues_plot.shape[0]) * dt

# --- Plot states ---
for r in range(4):
    for c in range(3):
        state_name = state_cols[r][c]
        ax = axs[r, c]
        ax.plot(t, trues_plot[:, r*3 + c], color='tab:blue', label='True')
        ax.plot(t, preds_plot[:, r*3 + c], color='tab:orange', linestyle='--', label='Pred')
        ax.set_ylabel(state_labels[r][c], fontsize=11)
        ax.grid(True)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

# --- Labels & layout ---
fig.text(0.5, 0.04, "Time [s]", ha='center', va='center', fontsize=14)
fig.legend(*axs[0, 0].get_legend_handles_labels(), loc='upper center', ncols=4, bbox_to_anchor=(0.5, 1.02))
plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.15, wspace=0.35)
plt.show()

