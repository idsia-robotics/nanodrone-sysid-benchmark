import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset, DataLoader

# ---------------------------------------------------------------------
# === Imports from project ===
# ---------------------------------------------------------------------
from models.models import PhysQuadModel
from dataset.dataset import QuadDataset, combine_concat_dataset

# ---------------------------------------------------------------------
# === CONFIG ===
# ---------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
horizon = 50
dt = 0.01


model_name = "physics"
test_trajs = ["melon"]


print(f"🧪 Test trajectories (auto-selected): {test_trajs}")

# ---------------------------------------------------------------------
# === Load test datasets ===
# ---------------------------------------------------------------------
test_ds = []
for traj in test_trajs:
    for run in [1, 2, 3]:
        file_name = f"{traj}_20251017_run{run}.csv"
        file_path = os.path.join("../data/test/", file_name)
        try:
            df = pd.read_csv(file_path)
            ds = QuadDataset(df, horizon=horizon)
            test_ds.append(ds)
        except Exception as e:
            print(f"⚠️ Skipped {file_name}: {e}")

test_dataset = combine_concat_dataset(
    ConcatDataset(test_ds), scale=False, fold="test"
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"📦 Loaded {len(test_ds)} test datasets")

# ---------------------------------------------------------------------
# === Load trained model ===
# ---------------------------------------------------------------------
phys_params = {
    "g": 9.81,
    "m": 0.045,
    "J": np.diag([2.3951e-5, 2.3951e-5, 3.2347e-6]),
    "thrust_to_weight": 2.0,
    "max_torque": np.array([1e-2, 1e-2, 3e-3]),
}

model = PhysQuadModel(phys_params, dt).to(device)
model.eval()

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
out_dir = f"../out/predictions/{model_name}_model_multistep/"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "_".join(test_trajs) + "_multistep.csv")
df_pred.to_csv(out_path, index=False)
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
