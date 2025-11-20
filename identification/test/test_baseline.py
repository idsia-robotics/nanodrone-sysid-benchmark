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
from identification.dataset import QuadDataset, combine_concat_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
horizon = 50
dt = 0.01
model_name = 'baseline'

test_trajs = ["square"]

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
    ConcatDataset(test_ds), scale=False
)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"📦 Loaded {len(test_ds)} test datasets")

state_names = ["x", "y", "z", "vx", "vy", "vz", "rx", "ry", "rz", "wx", "wy", "wz"]

# ---------------------------------------------------------------------
# === Run predictions ===
# ---------------------------------------------------------------------
preds, trues = [], []
with torch.no_grad():
    for x0, u_seq, x_seq_true in test_loader:
        x0, u_seq = x0.to(device), u_seq.to(device)
        x_pred = x0.repeat(1, horizon, 1).cpu()
        preds.append(x_pred)
        trues.append(x_seq_true)

preds = torch.cat(preds, dim=0).numpy()
trues = torch.cat(trues, dim=0).numpy()

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

# Take the first trajectory for visualization
preds_seq, trues_seq = preds[0], trues[0]

# === Directly use SO(3) log angles ===
# State structure: [x, y, z, vx, vy, vz, rx, ry, rz, wx, wy, wz]
preds_plot = preds_seq
trues_plot = trues_seq
# =====================================================
# --- Save baseline results ---
# =====================================================
out_dir = "../out/predictions/real/baseline_model_multistep"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "_".join(test_trajs) + "_multistep.parquet")
df_pred.to_parquet(out_path, index=False)
print(f"💾 Saved to {out_path}")