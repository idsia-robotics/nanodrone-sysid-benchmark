import os

import joblib
import numpy as np
import pandas as pd
import test_torch
import test_torch.nn as nn
import test_torch.optim as optim
from test_torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

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

# === Import your model and dataset classes ===
from idsia_mpc.control_engineering_practice.sysid.models import MotorsPhysQuadModel, PhysQuadModel, QuadMultiStepModel, \
    NeuralQuadModel
from idsia_mpc.control_engineering_practice.sysid.dataset import QuadMultiStepDataset, combine_concat_dataset

# === GPU selection ===
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# === Config ===
scale = False
pretrained = False
custom_loss = False

epochs = 10000
batch_size = 512
lr_start = 1e-4
lr_end = 1e-8
dt = 0.01
horizon = 10

mode = "residual"
model_path = f"../out/new/models/{mode}_quad_model_multistep_h{horizon}.pt"

if not os.path.exists(os.path.dirname(model_path)):
    os.makedirs(os.path.dirname(model_path))
print("✅ Model path:", model_path)

train_trajs = ["square", "multisine", "random1"]

train_ds = []
for traj in train_trajs:
    for run in [1, 2, 3, 4, 5]:
        try:
            file_name = f'{traj}_20251017_run{run}.parquet'
            df = pd.read_parquet(os.path.join('../../data/real/processed/new/train', file_name))
            df = df.rename(columns={"torch_yaw": "torque_yaw"})  # fix typo if present
            ds = QuadMultiStepDataset(df, horizon=horizon, split='train')
            train_ds.append(ds)
        except Exception as e:
            print(e)
            continue

valid_trajs = ["random2"]

valid_ds = []
for traj in valid_trajs:
    for run in [1, 2, 3, 4, 5]:
        try:
            file_name = f'{traj}_20251017_run{run}.parquet'
            df = pd.read_parquet(os.path.join('../../data/real/processed/new/train', file_name))
            df = df.rename(columns={"torch_yaw": "torque_yaw"})  # fix typo if present
            ds = QuadMultiStepDataset(df, horizon=horizon, split='train')
            valid_ds.append(ds)
        except Exception as e:
            print(e)
            continue

train_dataset = combine_concat_dataset(ConcatDataset(train_ds), scale=True, fold="train")
valid_dataset = combine_concat_dataset(ConcatDataset(valid_ds), scale=True, fold="valid")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# === Initialize model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Initialize submodels ===
phys_params = {
    "g": 9.81,
    "m": 0.032,
    "J": np.diag([1.43e-6, 1.43e-6, 2.89e-6]),
    "thrust_to_weight": 2.0,
    "max_torque": np.array([1e-4, 1e-4, 3e-5]),
}

scaler_dir = "/home/rbusetto/nanodrone-sysid-mpc/idsia_mpc/control_engineering_practice/sysid/train/scalers"
x_scaler_path = os.path.join(scaler_dir, "x_scaler.pkl")
x_scaler = joblib.load(x_scaler_path)
phys_model = PhysQuadModel(phys_params, dt, x_scaler=x_scaler).to(device)
phys_motor_model = MotorsPhysQuadModel(phys_model).to(device)
neural_model = NeuralQuadModel(dt).to(device)
model = QuadMultiStepModel(phys_motor_model, neural_model, mode=mode).to(device)

print(f"🧠 Initialized QuadMultiStepModel with mode='{mode}'")

if pretrained and os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Loaded pretrained model from {model_path}")
else:
    print("🔧 Training from scratch.")

# === Optimizer & Scheduler ===
optimizer = optim.Adam(model.parameters(), lr=lr_start)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_end)

# === Loss ===
criterion = nn.MSELoss()

# === Training Loop ===
best_val_loss = float("inf")

for epoch in range(epochs):
    # ---------------- TRAIN ----------------
    model.train()
    train_loss = 0.0
    for x0, u_seq, x_seq in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
        x0, u_seq, x_seq = x0.to(device), u_seq.to(device), x_seq.to(device)
        optimizer.zero_grad()

        pred_seq = model(x0, u_seq)  # shape [B, H, D]

        # No quaternion→Euler conversion since quaternions are scaled internally
        loss = criterion(pred_seq, x_seq)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # ---------------- VALID ----------------
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for x0, u_seq, x_seq in valid_loader:
            x0, u_seq, x_seq = x0.to(device), u_seq.to(device), x_seq.to(device)
            pred_seq = model(x0, u_seq)  # shape [B, H, D]
            # No quaternion→Euler conversion since quaternions are scaled internally
            loss = criterion(pred_seq, x_seq)
            valid_loss += loss.item()

    avg_valid_loss = valid_loss / len(valid_loader)
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}, LR={current_lr:.2e}, Train={avg_train_loss:.6f}, Valid={avg_valid_loss:.6f}")

    # Save best model
    if avg_valid_loss < best_val_loss:
        best_val_loss = avg_valid_loss
        torch.save(model.state_dict(), model_path)
        print(f"💾 Saved best model at epoch {epoch+1} with valid loss {avg_valid_loss:.6f}")

    scheduler.step()

# === Save final model ===
torch.save(model.state_dict(), model_path)
print(f"✅ Training complete. Model saved as {model_path}")
