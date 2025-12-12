import os
import json

import torch
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset, DataLoader
from torch.onnx import TrainingMode


# ---------------------------------------------------------------------
# === Imports from project ===
# ---------------------------------------------------------------------
from models.models import NeuralQuadModel
from dataset.dataset import QuadDataset, combine_concat_dataset

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
    for run in [1, 2, 3]:
        file_name = f"{traj}_20251017_run{run}.csv"
        file_path = os.path.join("../data/test", file_name)
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
x_scaler = joblib.load(os.path.join(scaler_dir, "x_scaler.pkl"))
preds = x_scaler.inverse_transform(preds.reshape(-1, preds.shape[-1])).reshape(preds.shape)
trues = x_scaler.inverse_transform(trues.reshape(-1, trues.shape[-1])).reshape(trues.shape)


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



# ------------------------------------------------------
# Dummy inputs for profiling
# ------------------------------------------------------
dummy_x0 = torch.randn(1, 12).to(device)
dummy_seq = torch.randn(1, 50, 4).to(device)

model = model.to(device).eval()

# ------------------------------------------------------
# 1) THOP MACs + Params
# ------------------------------------------------------
import time
from thop import profile
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

# =====================================================================
# === OPTIONAL: EXPORT 1-STEP MODEL FOR PROFILING =====================
# =====================================================================
export_to_onnx = False           # <<< toggle here
onnx_export_dir = f"../out/export/{model_name}_model_multistep"
os.makedirs(onnx_export_dir, exist_ok=True)

if export_to_onnx:
    print("\n📦 Exporting Neural model (1-step) for ONNX profiling...")

    # -------------------------------------------------
    # 1. One-step wrapper (ONNX-friendly)
    # -------------------------------------------------
    class NeuralOneStep(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x0, u0):
            """
            x0: (B,12)
            u0: (B,4)
            returns x1: (B,12)
            """
            # Build fake horizon=1 sequence
            u_seq = u0.unsqueeze(1)              # (B,1,4)

            # NeuralQuadModel expects (B,H,4)
            x_seq = self.base(x0, u_seq)         # → (B,1,12)

            return x_seq[:, 0, :]                # return x₁

    export_model = NeuralOneStep(model).to(device).eval()

    # -------------------------------------------------
    # 2. Dummy inputs
    # -------------------------------------------------
    dummy_x0 = torch.zeros(1, 12).to(device)
    dummy_u0 = torch.zeros(1, 4).to(device)

    # -------------------------------------------------
    # 3. Save sample I/O (optional but useful)
    # -------------------------------------------------
    torch.save(
        {"x0": dummy_x0.cpu(), "u0": dummy_u0.cpu()},
        os.path.join(onnx_export_dir, "sample_io.pt")
    )

    # -------------------------------------------------
    # 4. Export ONNX
    # -------------------------------------------------
    onnx_path = os.path.join(onnx_export_dir, f"{model_name}_1step.onnx")

    try:
        torch.onnx.export(
            export_model,
            (dummy_x0, dummy_u0),
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=["x0", "u0"],
            output_names=["x1"],
            dynamic_axes={
                "x0": {0: "batch"},
                "u0": {0: "batch"},
                "x1": {0: "batch"},
            },
            training=TrainingMode.EVAL  # avoid fusion to make profiling accurate
        )
        print(f"🟢 ONNX 1-step model exported → {onnx_path}")

    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
