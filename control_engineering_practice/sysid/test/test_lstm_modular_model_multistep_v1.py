import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import ConcatDataset, DataLoader

# ---------------------------------------------------------------------
# === Find project root ===
# ---------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
while True:
    if os.path.exists(os.path.join(current_dir, "idsia_mpc")):
        PROJECT_ROOT = current_dir
        break
    parent = os.path.dirname(current_dir)
    if parent == current_dir:
        raise RuntimeError("Could not find project root containing 'idsia_mpc'")
    current_dir = parent

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f"Using project root: {PROJECT_ROOT}")

# ---------------------------------------------------------------------
# === Imports from project ===
# ---------------------------------------------------------------------
from idsia_mpc.control_engineering_practice.plot_utils import setup_matplotlib
from idsia_mpc.control_engineering_practice.quat_utils import quat_to_euler
from idsia_mpc.control_engineering_practice.sysid.models import QuadLSTMModular
from idsia_mpc.control_engineering_practice.sysid.dataset import QuadMultiStepDataset, combine_concat_dataset

# ---------------------------------------------------------------------
# === CONFIG ===
# ---------------------------------------------------------------------
setup_matplotlib()

ALL_TRAJS = ["random1", "random2", "melon", "square", "multisine"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
horizon = 10
dt = 0.01

# ---------------------------------------------------------------------
# === Locate trained model automatically ===
# ---------------------------------------------------------------------
model_root = "../out/new/models"

# find all available LSTM model files
model_files = sorted(
    [f for f in os.listdir(model_root) if f.startswith("lstm_") and f.endswith(".pt")],
    key=lambda x: os.path.getmtime(os.path.join(model_root, x)),
    reverse=True,
)

if not model_files:
    raise RuntimeError("❌ No trained model found in ../out/new/models/")

print("\n📂 Available trained models:")
for idx, name in enumerate(model_files, start=1):
    mtime = os.path.getmtime(os.path.join(model_root, name))
    print(f"  [{idx}] {name}  (modified: {pd.to_datetime(mtime, unit='s'):%Y-%m-%d %H:%M})")

# --- Ask user to select one ---
while True:
    try:
        choice = int(input(f"\nSelect model [1–{len(model_files)}]: ").strip())
        if 1 <= choice <= len(model_files):
            break
        else:
            print(f"⚠️ Please enter a number between 1 and {len(model_files)}.")
    except ValueError:
        print("⚠️ Invalid input. Please enter a valid number.")

# --- Load selected model ---
model_file = model_files[choice - 1]
model_name = model_file.replace(".pt", "")
group = model_name.split("_")[1]
model_path = os.path.join(model_root, model_file)

groups = {
    "pos": (0,3),
    "vel": (3,6),
    "rot": (6,9),
    "omega": (9,12),
}

i_start, i_end = groups[group]

print(f"\n✅ Selected model: {model_name}")

# ---------------------------------------------------------------------
# === Load training trajectory info ===
# ---------------------------------------------------------------------
scaler_dir = f"/home/rbusetto/nanodrone-sysid-mpc/idsia_mpc/control_engineering_practice/sysid/train/{model_name}/scalers"
traj_info_path = os.path.join(scaler_dir, "trajectories.json")

if not os.path.exists(traj_info_path):
    raise FileNotFoundError(f"❌ trajectories.json not found for model: {model_name}")

with open(traj_info_path, "r") as f:
    traj_info = json.load(f)

train_trajs = traj_info["train_trajs"]
# test_trajs = [t for t in ALL_TRAJS if t not in train_trajs]
test_trajs = ["melon"]

print(f"🧩 Train trajectories: {train_trajs}")
print(f"🧪 Test trajectories (auto-selected): {test_trajs}")

# ---------------------------------------------------------------------
# === Load test datasets ===
# ---------------------------------------------------------------------
test_ds = []
for traj in test_trajs:
    for run in [1, 2, 3, 4, 5]:
        file_name = f"{traj}_20251017_run{run}.parquet"
        file_path = os.path.join("../../data/real/processed/new/test", file_name)
        try:
            df = pd.read_parquet(file_path)
            df = df.rename(columns={"torch_yaw": "torque_yaw"})
            ds = QuadMultiStepDataset(df, horizon=horizon, split="train")
            test_ds.append(ds)
        except Exception as e:
            print(f"⚠️ Skipped {file_name}: {e}")

test_dataset = combine_concat_dataset(
    ConcatDataset(test_ds), scale=True, fold="test", scaler_dir=scaler_dir
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"📦 Loaded {len(test_ds)} test datasets")

# ---------------------------------------------------------------------
# === Load trained model ===
# ---------------------------------------------------------------------
ckpt = torch.load(model_path, map_location=device)
cfg = ckpt["config"]
model = QuadLSTMModular(**cfg).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"✅ Model loaded from {model_path}")

# ---------------------------------------------------------------------
# === Run predictions ===
# ---------------------------------------------------------------------
preds, trues = [], []
with torch.no_grad():
    for x0, u_seq, x_seq_true in test_loader:
        x0, u_seq = x0.to(device), u_seq.to(device)
        x0 = x0[:, i_start:i_end]
        x_seq_true = x_seq_true[:, :, i_start:i_end]
        x_pred = model(x0, u_seq).cpu()
        preds.append(x_pred)
        trues.append(x_seq_true)

preds = torch.cat(preds, dim=0).numpy()
trues = torch.cat(trues, dim=0).numpy()

# ---------------------------------------------------------------------
# === Denormalize ===
# ---------------------------------------------------------------------
x_scaler_full = joblib.load(os.path.join(scaler_dir, "x_scaler.pkl"))

# Rebuild a mini-scaler with only the needed dimensions
from sklearn.preprocessing import StandardScaler
x_scaler_sub = StandardScaler()
x_scaler_sub.mean_ = x_scaler_full.mean_[i_start:i_end]
x_scaler_sub.scale_ = x_scaler_full.scale_[i_start:i_end]
x_scaler_sub.var_ = x_scaler_full.var_[i_start:i_end]
x_scaler_sub.n_features_in_ = i_end - i_start
x_scaler_sub.feature_names_in_ = getattr(x_scaler_full, "feature_names_in_", None)

# Now safely inverse-transform only 3D data
preds = x_scaler_sub.inverse_transform(preds.reshape(-1, preds.shape[-1])).reshape(preds.shape)
trues = x_scaler_sub.inverse_transform(trues.reshape(-1, trues.shape[-1])).reshape(trues.shape)

# Take the first trajectory for visualization
preds_seq, trues_seq = preds[0], trues[0]

# === Directly use SO(3) log angles ===
# State structure: [x, y, z, vx, vy, vz, rx, ry, rz, wx, wy, wz]
preds_plot = preds_seq
trues_plot = trues_seq

# ---------------------------------------------------------------------
# === RMSE Logging ===
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# === Dynamic labels and units based on group ===
# ---------------------------------------------------------------------
group_labels = {
    "pos":  (['$x$', '$y$', '$z$'], ['x', 'y', 'z'], ['[m]', '[m]', '[m]']),
    "vel":  (['$v_x$', '$v_y$', '$v_z$'], ['vx', 'vy', 'vz'], ['[m/s]', '[m/s]', '[m/s]']),
    "rot":  (['$\\phi$', '$\\theta$', '$\\psi$'], ['roll', 'pitch', 'yaw'], ['[rad]', '[rad]', '[rad]']),
    "omega":(['$\\omega_x$', '$\\omega_y$', '$\\omega_z$'], ['wx', 'wy', 'wz'], ['[rad/s]', '[rad/s]', '[rad/s]']),
}

if group not in group_labels:
    raise ValueError(f"❌ Unknown group '{group}'. Must be one of {list(group_labels.keys())}")

latex_labels, short_labels, units = group_labels[group]


def log_metrics(trues, preds):
    trues = np.reshape(trues, (-1, trues.shape[-1]))
    preds = np.reshape(preds, (-1, preds.shape[-1]))

    print("=== State-wise Metrics ===")
    header = f"{'State':<10} | {'RMSE':>10} | {'MAE':>10} | {'R²':>8} | {'NRMSE [%]':>10}"
    print(header)
    print("-" * len(header))

    rmse_vals, r2_vals = [], []

    for i, lbl in enumerate(latex_labels):
        true_i, pred_i = trues[:, i], preds[:, i]

        rmse_i = rmse(true_i, pred_i)
        mae_i = mean_absolute_error(true_i, pred_i)
        r2_i = r2_score(true_i, pred_i)
        nrmse_i = 100 * rmse_i / (np.max(true_i) - np.min(true_i) + 1e-8)

        rmse_vals.append(rmse_i)
        r2_vals.append(r2_i)

        print(f"{lbl:<10} | {rmse_i:10.5f} | {mae_i:10.5f} | {r2_i:8.3f} | {nrmse_i:10.2f}")

    print("-" * len(header))
    print(f"{'Overall':<10} | {np.mean(rmse_vals):10.5f} | {'-':>10} | {np.mean(r2_vals):8.3f} | {'-':>10}")

log_metrics(trues_plot, preds_plot)

# ---------------------------------------------------------------------
# === Plot results ===
# ---------------------------------------------------------------------
fig, axs = plt.subplots(1, len(latex_labels), figsize=(14, 3), sharex=True, dpi=200)
t = np.arange(trues_plot.shape[0]) * dt

for c in range(len(latex_labels)):
    ax = axs[c]
    ax.plot(t, trues_plot[:, c], color='tab:blue', label='True')
    ax.plot(t, preds_plot[:, c], color='tab:orange', linestyle='--', label='Pred')
    ax.set_ylabel(f"{latex_labels[c]} {units[c]}", fontsize=11)
    ax.grid(True)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

fig.text(0.5, 0.04, "Time [s]", ha='center', va='center', fontsize=14)
fig.legend(*axs[0].get_legend_handles_labels(), loc='upper center', ncols=4, bbox_to_anchor=(0.5, 1.02))
plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.15, wspace=0.35)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# === Export results to CSV ===
# ---------------------------------------------------------------------
pred_out_dir = f"../out/new/predictions/{model_name}"
os.makedirs(pred_out_dir, exist_ok=True)

data = {'t': t}
for i, name in enumerate(short_labels):
    data[f'{name}_true'] = trues_plot[:, i]
    data[f'{name}_pred'] = preds_plot[:, i]

df_export = pd.DataFrame(data)
csv_path = os.path.join(pred_out_dir, f"{'_'.join(test_trajs)}.csv")
df_export.to_csv(csv_path, index=False)
print(f"✅ Exported predictions to {csv_path}")
