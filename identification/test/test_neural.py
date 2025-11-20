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
# === Imports from project ===
# ---------------------------------------------------------------------
from identification.models import NeuralQuadModel
from identification.dataset import QuadDataset, combine_concat_dataset

# ---------------------------------------------------------------------
# === CONFIG ===
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
horizon = 50
dt = 0.01

# ---------------------------------------------------------------------
# === Locate trained model automatically ===
# ---------------------------------------------------------------------
model_root = "../out/models"

# find all available LSTM model files
model_files = sorted(
    [f for f in os.listdir(model_root) if f.startswith("neural") and f.endswith(".pt")],
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
model_path = os.path.join(model_root, model_file)

print(f"\n✅ Selected model: {model_name}")

# ---------------------------------------------------------------------
# === Load training trajectory info ===
# ---------------------------------------------------------------------
scaler_dir = f"../scalers/{model_name}/"
traj_info_path = os.path.join(scaler_dir, "trajectories.json")

if not os.path.exists(traj_info_path):
    raise FileNotFoundError(f"❌ trajectories.json not found for model: {model_name}")

with open(traj_info_path, "r") as f:
    traj_info = json.load(f)

train_trajs = traj_info["train_trajs"]
test_trajs = ["melon"]

print(f"🧩 Train trajectories: {train_trajs}")
print(f"🧪 Test trajectories (auto-selected): {test_trajs}")

# ---------------------------------------------------------------------
# === Load test datasets ===
# ---------------------------------------------------------------------
test_ds = []
for traj in test_trajs:
    for run in [1]:
        file_name = f"{traj}_20251017_run{run}.csv"
        file_path = os.path.join("../../data/real/processed/test", file_name)
        try:
            df = pd.read_csv(file_path)
            ds = QuadDataset(df, horizon=horizon)
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
model = NeuralQuadModel(**cfg).to(device)
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
        x_pred = model(x0, u_seq).cpu()
        preds.append(x_pred)
        trues.append(x_seq_true)

preds = torch.cat(preds, dim=0).numpy()
trues = torch.cat(trues, dim=0).numpy()

# ---------------------------------------------------------------------
# === Denormalize ===
# ---------------------------------------------------------------------
# --- Load scaler ---
x_scaler = joblib.load(os.path.join(scaler_dir, "x_scaler.pkl"))

# ---- index sets (same as training scaling) ----
idx_scale    = np.array([0,1,2, 3,4,5, 9,10,11])
idx_no_scale = np.array([6,7,8])   # rotations stay untouched

# ---- Shapes ----
B, H, D = preds.shape

# ---- Flatten ----
preds_flat = preds.reshape(-1, D)
trues_flat = trues.reshape(-1, D)

# ---- Copies ----
preds_inv = preds_flat.copy()
trues_inv = trues_flat.copy()

# ---- Inverse-transform only selected dims ----
preds_inv[:, idx_scale] = x_scaler.inverse_transform(preds_flat[:, idx_scale])
trues_inv[:, idx_scale] = x_scaler.inverse_transform(trues_flat[:, idx_scale])

# ---- Leave rotations untouched ----
preds_inv[:, idx_no_scale] = preds_flat[:, idx_no_scale]
trues_inv[:, idx_no_scale] = trues_flat[:, idx_no_scale]

# ---- Reshape back to original (B, H, D) ----
preds = preds_inv.reshape(B, H, D)
trues = trues_inv.reshape(B, H, D)

state_names = ["x", "y", "z", "vx", "vy", "vz", "rx", "ry", "rz", "wx", "wy", "wz"]

# =====================================================
# --- Convert to DataFrame (similar to previous code) ---
# =====================================================
# Build dataframe per time step (naive constant baseline)
N = preds.shape[0]
data = {}

# time vector (optional): you can pull from your test dataset
# e.g. if test_dataset has 't' inside its dataframe:
if hasattr(test_dataset, "df") and "t" in test_dataset.df.columns:
    t_vec = test_dataset.df["t"].values[:N]
else:
    t_vec = np.arange(N) * 0.01  # fallback 100 Hz assumption
data["t"] = t_vec

# add true states
for i, name in enumerate(state_names):
    data[name] = trues[:, 0, i]  # the first step of x_seq_true is x_{t+1}

# add baseline predictions per horizon
for h in range(1, horizon + 1):
    for i, name in enumerate(state_names):
        data[f"{name}_pred_h{h}"] = preds[:, h - 1, i]  # each step h

df_pred = pd.DataFrame(data)
print(f"✅ Baseline DataFrame shape: {df_pred.shape}")

# =====================================================
# --- Save baseline results ---
# =====================================================
out_dir = f"../out/predictions/real/{model_name}_model_multistep"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "_".join(test_trajs) + "_multistep.parquet")
df_pred.to_parquet(out_path, index=False)
print(f"💾 Saved to {out_path}")

# =====================================================
# --- Quick sanity check plot ---
# =====================================================
N_end = len(df_pred['t'])

plt.figure(figsize=(8, 4))
plt.plot(df_pred["t"][:N_end], df_pred["x"][:N_end], label="x true")
plt.plot(df_pred["t"][:N_end], df_pred["x_pred_h1"][:N_end], "--", label="x pred (h=1)")
plt.xlabel("Time [s]")
plt.ylabel("x [m]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()