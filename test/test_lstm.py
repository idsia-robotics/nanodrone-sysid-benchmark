import os
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
from models.models import QuadLSTM
from dataset.dataset import QuadDataset, combine_concat_dataset

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
model_root = "../out/models/"

# find all available LSTM model files
model_files = sorted(
    [f for f in os.listdir(model_root) if f.startswith("lstm") and f.endswith(".pt")],
    key=lambda x: os.path.getmtime(os.path.join(model_root, x)),
    reverse=True,
)

if not model_files:
    raise RuntimeError("❌ No trained model found in ../out/models/")

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

# print(f"🧩 Train trajectories: {train_trajs}")
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
    ConcatDataset(test_ds), scale=True, fold="test", scaler_dir=scaler_dir
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print(f"📦 Loaded {len(test_ds)} test datasets")

# ---------------------------------------------------------------------
# === Load trained model ===
# ---------------------------------------------------------------------
ckpt = torch.load(model_path, map_location=device)
cfg = ckpt["config"]
model = QuadLSTM(**cfg).to(device)
model.load_state_dict(ckpt['model_state'])
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


# =====================================================================
# === OPTIONAL: EXPORT LSTM ONE-STEP MODEL FOR PROFILING ===============
# =====================================================================
export_to_onnx = False           # <<< toggle here
onnx_export_dir = f"../out/export/{model_name}_model_multistep"
os.makedirs(onnx_export_dir, exist_ok=True)

if export_to_onnx:
    print("📦 Exporting 1-step LSTM model for profiling...")

    os.makedirs(onnx_export_dir, exist_ok=True)

    # -------------------------------------------------
    # 1. Create a 1-step wrapper around QuadLSTM
    # -------------------------------------------------
    class LSTMOneStep(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x0, u0):
            """
            x0: (B,12)
            u0: (B,4)
            returns x1: (B,12)
            """
            # Build u_seq with T=1
            u_seq = u0.unsqueeze(1)  # (B,1,4)

            # Run full model
            x_seq = self.base(x0, u_seq)  # (B,1,12)

            return x_seq[:, 0]  # (B,12)


    one_step_model = LSTMOneStep(model).eval().to(device)

    # -------------------------------------------------
    # 2. Save sample I/O
    # -------------------------------------------------
    sample_x0 = torch.tensor(preds[0, 0], dtype=torch.float32)
    sample_u0 = torch.zeros(4, dtype=torch.float32)   # dummy control

    torch.save(
        {"x0": sample_x0, "u0": sample_u0},
        os.path.join(onnx_export_dir, "sample_io.pt")
    )
    print("💾 Saved sample I/O → sample_io.pt")

    # -------------------------------------------------
    # 3. ONNX export (x0, u0) → x1
    # -------------------------------------------------
    dummy_x0 = sample_x0.unsqueeze(0).to(device)  # (1,12)
    dummy_u0 = sample_u0.unsqueeze(0).to(device)  # (1,4)
    onnx_path = os.path.join(onnx_export_dir, f"{model_name}_1step.onnx")

    try:
        torch.onnx.export(
            one_step_model,
            (dummy_x0, dummy_u0),
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['x0', 'u0'],
            output_names=['x1'],
            dynamic_axes={
                'x0': {0: 'batch_size'},
                'u0': {0: 'batch_size'},
                'x1': {0: 'batch_size'}
            },
            training=TrainingMode.TRAINING   # keep ops unfused for profiling
        )
        print(f"🟢 ONNX 1-step LSTM exported → {onnx_path}")
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
