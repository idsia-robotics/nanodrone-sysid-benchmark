import os
import numpy as np
import pandas as pd
import test_torch
import test_torch.nn as nn
import test_torch.optim as optim
from test_torch.utils.data import DataLoader
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
from idsia_mpc.control_engineering_practice.sysid.models import PhysQuadModel, NeuralQuadModel, ResidualQuadModel
from idsia_mpc.control_engineering_practice.sysid.dataset import QuadDataset
from idsia_mpc.control_engineering_practice.sysid.losses import QuadStateMSELoss, ScaledMSELoss
from train_utils import state_quat_to_euler  # includes quat_to_euler_torch internally

# === Example: set GPU 1 only ===
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# === Config ===
scale = False
pretrained = True
custom_loss = False

epochs = 10000
batch_size = 64
lr_start = 1e-5
lr_end = 1e-8
dt = 0.01

# === Physics parameters for PhysQuadModel ===
phys_params = {
    "g": 9.81,
    "m": 0.032,
    "J": np.diag([1.43e-6, 1.43e-6, 2.89e-6]),
    "thrust_to_weight": 2.0,
    "max_torque": np.array([1e-4, 1e-4, 3e-5]),
}

type = "real"
traj_id = '1848'
traj_id_valid = '1809'
traj_type_valid = "melon"

if type == "sim":
    traj_type = "points"
    file_name_sim = f'experiment_points_{traj_id}.parquet'
    df = pd.read_parquet(os.path.join('../../data/sim/', file_name_sim))
elif type == "real":
    traj_type = "random"
    file_name_real = f'experiment_{traj_type}_{traj_id}.parquet'
    df = pd.read_parquet(os.path.join('../../data/real/processed/', file_name_real))
    df = df.rename(columns={"torch_yaw": "torque_yaw"})  # fix typo if present

file_name_real_valid = f'experiment_{traj_type_valid}_{traj_id_valid}.parquet'
df_valid = pd.read_parquet(os.path.join('../../data/real/processed/', file_name_real_valid))
df_valid = df_valid.rename(columns={"torch_yaw": "torque_yaw"})  # fix typo if present

mode = "residual"  # or "neural", "residual"
model_path = f"../out/models/{mode}_quad_model_onestep_{type}_{traj_type}_{traj_id}.pt"
print("✅ Model path:", model_path)

train_dataset = QuadDataset(df, split='train', scale=scale)
valid_dataset = QuadDataset(df_valid, split='valid', scale=scale)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# === Compute scale_vector (based on Euler states) ===
with torch.no_grad():
    xs_euler = state_quat_to_euler(train_dataset.xs)  # convert all training states
scale_vector = torch.std(xs_euler, dim=0).numpy()     # compute std per component

# Optional: check shape
print(f"📏 scale_vector shape: {scale_vector.shape}")

# === Initialize Residual model ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
phys_model = PhysQuadModel(phys_params, dt).to(device)
neural_model = NeuralQuadModel(dt).to(device)
model = ResidualQuadModel(phys_model, neural_model).to(device)

# === Load model if exists ===
if pretrained and os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Loaded pretrained Residual model from {model_path}")
else:
    print("🔧 Training from scratch.")

# === Optimizer, scheduler ===
optimizer = optim.Adam(model.parameters(), lr=lr_start)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_end)

# === Loss ===
if custom_loss:
    criterion = QuadStateMSELoss(model)
else:
    criterion = ScaledMSELoss(scale_vector, eps=1e-8)

# === Training loop ===
best_val_loss = float('inf')
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for x, u, x_next in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
        x, u, x_next = x.to(device), u.to(device), x_next.to(device)
        optimizer.zero_grad()
        pred_next = model(x, u)

        # --- Convert quaternions to Euler before loss ---
        pred_next_euler = state_quat_to_euler(pred_next)
        x_next_euler = state_quat_to_euler(x_next)

        loss = criterion(pred_next_euler, x_next_euler)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # === Validation ===
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for x, u, x_next in valid_loader:
            x, u, x_next = x.to(device), u.to(device), x_next.to(device)
            pred_next = model(x, u)
            pred_next_euler = state_quat_to_euler(pred_next)
            x_next_euler = state_quat_to_euler(x_next)
            loss = criterion(pred_next_euler, x_next_euler)
            valid_loss += loss.item()
    avg_valid_loss = valid_loss / len(valid_loader)

    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}, LR={current_lr:.2e}, Train={avg_train_loss:.6f}, Valid={avg_valid_loss:.6f}")

    if avg_valid_loss < best_val_loss:
        best_val_loss = avg_valid_loss
        torch.save(model.state_dict(), model_path)
        print(f"💾 Saved best model at epoch {epoch+1} with valid loss {avg_valid_loss:.6f}")

    scheduler.step()

# === Save final model ===
torch.save(model.state_dict(), model_path)
print(f"✅ Training complete. Model saved as {model_path}")