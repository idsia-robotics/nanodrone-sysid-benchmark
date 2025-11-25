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
from thop import profile

# ---------------------------------------------------------------------
# === Imports from project ===
# ---------------------------------------------------------------------
from identification.models import NeuralQuadModel, PhysQuadModel, ResidualQuadModel
from identification.dataset import QuadDataset, combine_concat_dataset

# ---------------------------------------------------------------------
# === CONFIG ===
# ---------------------------------------------------------------------
ALL_TRAJS = ["random", "melon", "square", "chirp"]
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
    [f for f in os.listdir(model_root) if f.startswith("phys+res") and f.endswith(".pt")],
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

test_trajs = ["melon"]

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
phys_params = {
    "g": 9.81,
    "m": 0.045,
    "J": np.diag([2.3951e-5, 2.3951e-5, 3.2347e-6]),
    "thrust_to_weight": 2.0,
    "max_torque": np.array([1e-2, 1e-2, 3e-3]),
}

ckpt = torch.load(model_path, map_location=device, weights_only=True)
state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

phys_model = PhysQuadModel(phys_params, dt).to(device)
neural_model = NeuralQuadModel(**ckpt["config"]).to(device)
model = ResidualQuadModel(
    phys=phys_model,
    neural=neural_model,
    x_scaler=test_dataset.x_scaler,
    u_scaler=test_dataset.u_scaler
).to(device)

model.load_state_dict(state)
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

# ---------------------------------------------------------------------
# === Denormalize ===
# ---------------------------------------------------------------------
x_scaler = joblib.load(os.path.join(scaler_dir, "x_scaler.pkl"))
preds = x_scaler.inverse_transform(preds.reshape(-1, preds.shape[-1])).reshape(preds.shape)
trues = x_scaler.inverse_transform(trues.reshape(-1, trues.shape[-1])).reshape(trues.shape)

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


profile = False
if profile:
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------
    # Dummy inputs for profiling
    # ------------------------------------------------------
    dummy_x0 = torch.randn(1, 12).to(device)
    dummy_seq = torch.randn(1, 50, 4).to(device)

    model = model.to(device).eval()

    # ------------------------------------------------------
    # 1) THOP MACs + Params
    # ------------------------------------------------------
    macs, params = profile(model, inputs=(dummy_x0, dummy_seq), verbose=False)

    print(f"MACs:   {macs / 1e6:.3f} M")
    print(f"Params: {params / 1e3:.1f} K")

    # ------------------------------------------------------
    # 2) True inference latency (ms per single forward)
    # ------------------------------------------------------
    def measure_latency(model, x0, u_seq, warmup=20, iters=200):
        # warmup
        for _ in range(warmup):
            _ = model(x0, u_seq)

        torch.cuda.synchronize()
        start = time.time()

        for _ in range(iters):
            _ = model(x0, u_seq)

        torch.cuda.synchronize()
        end = time.time()

        return (end - start) / iters * 1000  # ms

    T_inf = measure_latency(model, dummy_x0, dummy_seq)
    print(f"Inference time: {T_inf:.4f} ms/step")

    # ------------------------------------------------------
    # Optional: dump in dict format for your LaTeX table
    # ------------------------------------------------------
    metrics_entry = {
        "MACs_M": macs / 1e6,
        "Params_K": params / 1e3,
        "T_inf_ms": T_inf,
    }
    print(metrics_entry)
