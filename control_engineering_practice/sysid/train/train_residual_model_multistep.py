import os
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
from idsia_mpc.control_engineering_practice.sysid.models import QuadMultiStepModel, PhysQuadModel, NeuralQuadModel  # ✅ physics backbone if needed
from idsia_mpc.control_engineering_practice.sysid.dataset import QuadMultiStepDataset, combine_concat_dataset
from idsia_mpc.control_engineering_practice.sysid.losses import ScaledMSELoss, QuadStateMSELoss
from train_utils import state_quat_to_euler

# === GPU selection ===
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# === Config ===
scale = False
pretrained = True
custom_loss = False

epochs = 10000
batch_size = 64
lr_start = 1e-4
lr_end = 1e-8
dt = 0.01
horizon = 100

# type = "real"
# traj_id = '1848'
# traj_id_valid = '1809'
# traj_type_valid = "melon"
#
# if type == "sim":
#     traj_type = "points"
#     file_name_sim = f'experiment_points_{traj_id}.parquet'
#     df = pd.read_parquet(os.path.join('../../data/sim/', file_name_sim))
# elif type == "real":
#     traj_type = "random"
#     file_name_real = f'experiment_{traj_type}_{traj_id}.parquet'
#     df = pd.read_parquet(os.path.join('../../data/real/processed/', file_name_real))
#     df = df.rename(columns={"torch_yaw": "torque_yaw"})  # fix typo if present
#
# file_name_real_valid = f'experiment_{traj_type_valid}_{traj_id_valid}.parquet'
# df_valid = pd.read_parquet(os.path.join('../../data/real/processed/', file_name_real_valid))
# df_valid = df_valid.rename(columns={"torch_yaw": "torque_yaw"})  # fix typo if present
#
# mode = "residual"  # or "neural", "residual"
# model_path = f"../out/models/{mode}_quad_model_multistep_{type}_{traj_type}_{traj_id}.pt"
# print("✅ Model path:", model_path)
#
# train_dataset = QuadMultiStepDataset(df, horizon=horizon, split='train', scale=scale)
# valid_dataset = QuadMultiStepDataset(df_valid, horizon=horizon, split='train', scale=scale)

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
            df = pd.read_parquet(os.path.join('../../data/real/processed/new/', file_name))
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
            df = pd.read_parquet(os.path.join('../../data/real/processed/new/', file_name))
            df = df.rename(columns={"torch_yaw": "torque_yaw"})  # fix typo if present
            ds = QuadMultiStepDataset(df, horizon=horizon, split='train')
            valid_ds.append(ds)
        except Exception as e:
            print(e)
            continue

train_dataset = combine_concat_dataset(ConcatDataset(train_ds))
valid_dataset = combine_concat_dataset(ConcatDataset(valid_ds))
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
neural_model = NeuralQuadModel(dt).to(device)

# === Build multi-step model depending on mode ===
if mode == "physics":
    model = QuadMultiStepModel(phys_model=phys_model, mode="physics").to(device)
elif mode == "neural":
    model = QuadMultiStepModel(neural_model=neural_model, mode="neural").to(device)
elif mode == "residual":
    model = QuadMultiStepModel(phys_model=phys_model, neural_model=neural_model, mode="residual").to(device)
else:
    raise ValueError(f"Invalid mode: {mode}")

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
        optimizer.zero_grad()

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
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}, LR={current_lr:.2e}, Train={avg_train_loss:.6f}, Valid={avg_valid_loss:.6f}")

    # Save best model
    if avg_train_loss < best_val_loss:
        best_val_loss = avg_train_loss
        torch.save(model.state_dict(), model_path)
        print(f"💾 Saved best model at epoch {epoch+1} with valid loss {avg_valid_loss:.6f}")

    scheduler.step()

# === Save final model ===
torch.save(model.state_dict(), model_path)
print(f"✅ Training complete. Model saved as {model_path}")
