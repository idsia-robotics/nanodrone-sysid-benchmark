# ======================================================
# This block is for validation purposes only.
# The physics model is deterministic and not trainable.
# ======================================================

import os
import numpy as np
import pandas as pd
import test_torch
import test_torch.nn as nn
import test_torch.optim as optim
from test_torch.utils.data import DataLoader
from tqdm import tqdm

# === Import your model and dataset classes ===
from idsia_mpc.control_engineering_practice.sysid.models import QuadMultiStepModel, PhysQuadModel  # ✅ physics backbone if needed
from idsia_mpc.control_engineering_practice.sysid.dataset import QuadMultiStepDataset
from idsia_mpc.control_engineering_practice.sysid.losses import ScaledMSELoss, QuadStateMSELoss
from train_utils import state_quat_to_euler

# === GPU selection ===
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# === Config ===
scale = False
pretrained = False
custom_loss = False

epochs = 1000
batch_size = 64
lr_start = 1e-3
lr_end = 1e-8
dt = 0.01
horizon = 50

mode = "physics"  # or "neural", "residual"
model_path = f"../out/models/{mode}_quad_model_multistep.pt"
print("✅ Model path:", model_path)

# === Load data ===
# df = pd.read_parquet('../../data/real/experiment_random_1848.parquet')
df = pd.read_parquet('../../data/sim/experiment_points_1848.parquet')
df = df.rename(columns={"torch_yaw": "torque_yaw"})  # fix typo if present

train_dataset = QuadMultiStepDataset(df, horizon=horizon, split='train', scale=scale)
valid_dataset = QuadMultiStepDataset(df, horizon=horizon, split='valid', scale=scale)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# === Compute scaling vector (Euler version of states) ===
with torch.no_grad():
    xs_euler = state_quat_to_euler(train_dataset.xs)  # initial states only
scale_vector = torch.std(xs_euler, dim=0).numpy()
print(f"📏 scale_vector shape: {scale_vector.shape}")

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

phys_model = PhysQuadModel(phys_params, dt).to(device)

# === Build multi-step model depending on mode ===
if mode == "physics":
    model = QuadMultiStepModel(phys_model=phys_model, mode="physics").to(device)

print(f"🧠 Initialized QuadMultiStepModel with mode='{mode}'")

# === Optimizer & Scheduler ===
# optimizer = optim.Adam(model.parameters(), lr=lr_start)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_end)

# === Loss ===
if custom_loss:
    criterion = QuadStateMSELoss(model)
else:
    criterion = ScaledMSELoss(scale_vector, eps=1e-8)

# === Training Loop ===
best_val_loss = float("inf")

for epoch in range(epochs):
    # ---------------- TRAIN ----------------
    model.train()
    train_loss = 0.0
    for x0, u_seq, x_seq in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
        x0, u_seq, x_seq = x0.to(device), u_seq.to(device), x_seq.to(device)
        # optimizer.zero_grad()

        pred_seq = model(x0, u_seq)  # shape [B, H, D]

        # convert quaternions to Euler
        # pred_seq: [B, H, 13]
        B, H, D = pred_seq.shape
        pred_seq_flat = pred_seq.reshape(B * H, D)
        x_seq_flat = x_seq.reshape(B * H, D)

        pred_seq_euler_flat = state_quat_to_euler(pred_seq_flat)
        x_seq_euler_flat = state_quat_to_euler(x_seq_flat)

        # reshape back to [B, H, D_euler]
        pred_seq_euler = pred_seq_euler_flat.reshape(B, H, -1)
        x_seq_euler = x_seq_euler_flat.reshape(B, H, -1)

        loss = criterion(pred_seq_euler, x_seq_euler)
        # loss.backward()
        # optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # ---------------- VALID ----------------
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for x0, u_seq, x_seq in valid_loader:
            x0, u_seq, x_seq = x0.to(device), u_seq.to(device), x_seq.to(device)
            pred_seq = model(x0, u_seq)
            B, H, D = pred_seq.shape
            pred_seq_flat = pred_seq.reshape(B * H, D)
            x_seq_flat = x_seq.reshape(B * H, D)

            pred_seq_euler_flat = state_quat_to_euler(pred_seq_flat)
            x_seq_euler_flat = state_quat_to_euler(x_seq_flat)

            # reshape back to [B, H, D_euler]
            pred_seq_euler = pred_seq_euler_flat.reshape(B, H, -1)
            x_seq_euler = x_seq_euler_flat.reshape(B, H, -1)
            loss = criterion(pred_seq_euler, x_seq_euler)
            valid_loss += loss.item()

    avg_valid_loss = valid_loss / len(valid_loader)
    # current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}, Train={avg_train_loss:.6f}, Valid={avg_valid_loss:.6f}")

    # Save best model
    if avg_train_loss < best_val_loss:
        best_val_loss = avg_train_loss
        torch.save(model.state_dict(), model_path)
        print(f"💾 Saved best model at epoch {epoch+1} with valid loss {avg_valid_loss:.6f}")

    # scheduler.step()

# === Save final model ===
torch.save(model.state_dict(), model_path)
print(f"✅ Training complete. Model saved as {model_path}")
