import argparse
import json
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset


# ---------------------------------------------------------------------
# === Imports ===
# ---------------------------------------------------------------------
sys.path.append("../..")

from identification.models import NeuralQuadModel
from identification.dataset import (
    QuadDataset,
    combine_concat_dataset,
)
from identification.losses import WeightedMSELoss, WeightedGeodesicLoss

# ---------------------------------------------------------------------
# === CLI arguments ===
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train LSTM quadrotor model with custom trajectories")
parser.add_argument("--train_trajs", type=str, default='["random", "square", "chirp"]')
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--epochs", type=int, default=10000)
parser.add_argument("--horizon", type=int, default=50)
args = parser.parse_args()

train_trajs = json.loads(args.train_trajs)
valid_trajs = train_trajs  # validation uses same trajs
device_str = args.device
epochs = args.epochs
horizon = args.horizon

# --- compose model name automatically ---
model_name = f"neural_v2_" + "_".join(train_trajs)
print(f"🧠 Model name composed automatically: {model_name}")

# ---------------------------------------------------------------------
# === Config ===
# ---------------------------------------------------------------------
pretrained = False
batch_size = 256
lr_start = 1e-5
lr_end = 1e-8
mode = "neural"

os.environ["CUDA_VISIBLE_DEVICES"] = device_str.split(":")[-1]
device = torch.device(device_str if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# === Paths ===
# ---------------------------------------------------------------------
model_dir = f"../out/models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, f"{model_name}.pt")
print(f"✅ Model will be saved to: {model_path}")

scaler_dir = (
    f"../scalers/{model_name}"
)
os.makedirs(scaler_dir, exist_ok=True)

# ---------------------------------------------------------------------
# === Build Datasets ===
# ---------------------------------------------------------------------
def load_split(trajs, base_dir, split):
    datasets = []
    for traj in trajs:
        for run in [1, 2, 3, 4, 5]:
            file_name = f"{traj}_20251017_run{run}.csv"
            file_path = os.path.join(base_dir, file_name)
            try:
                df = pd.read_csv(file_path)
                ds = QuadDataset(df, horizon=horizon)
                datasets.append(ds)
            except Exception as e:
                print(f"⚠️ Skipped {file_path}: {e}")
    print(f"Loaded {len(datasets)} datasets for {split}")
    return datasets


train_ds = load_split(train_trajs, "../../data/real/processed/train/", "train")
valid_ds = load_split(valid_trajs, "../../data/real/processed/valid/", "valid")

train_dataset = combine_concat_dataset(
    ConcatDataset(train_ds), scale=True, fold="train", scaler_dir=scaler_dir
)
valid_dataset = combine_concat_dataset(
    ConcatDataset(valid_ds), scale=True, fold="valid", scaler_dir=scaler_dir
)

# --- Save trajectory info ---
traj_info = {"train_trajs": train_trajs, "valid_trajs": valid_trajs}
traj_info_path = os.path.join(scaler_dir, "trajectories.json")
with open(traj_info_path, "w") as f:
    json.dump(traj_info, f, indent=4)
print(f"📝 Saved trajectory info to {traj_info_path}")

# ---------------------------------------------------------------------
# === Dataloaders ===
# ---------------------------------------------------------------------
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# === Initialize model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = NeuralQuadModel(num_layers=5, hidden_dim=64).to(device)
print(f"🧠 Initialized Model")

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {num_params:,}")

if pretrained and os.path.exists(model_path):
    # Rebuild model from saved parameters
    # Load weights
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"✅ Loaded pretrained model from {model_path}")
else:
    print("🔧 Training from scratch.")

# === Optimizer & Scheduler ===
optimizer = optim.Adam(model.parameters(), lr=lr_start)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_end)

# === Loss ===
# criterion = nn.MSELoss()  # no scaling — model handles normalization
# criterion = WeightedGeodesicLoss(lambda_=0.02)
criterion = WeightedMSELoss(lambda_=0.1)

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

        # Automatically grab all non-callable, non-private attributes
        config = {
            k: v for k, v in vars(model).items()
            if not k.startswith("_") and not callable(v)
        }

        # Build checkpoint dictionary
        checkpoint = {
            "model_state": model.state_dict(),
            "config": {
                "hidden_dim": getattr(model, "hidden_dim", None),
                "num_layers": getattr(model, "num_layers", None),
            },
            "optimizer_state": optimizer.state_dict(),  # optional but useful
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": best_val_loss,
        }

        torch.save(checkpoint, model_path)
        print(f"💾 Saved best model at epoch {epoch+1} with valid loss {avg_valid_loss:.6f}")

    scheduler.step()

# === Save final model ===
torch.save(model.state_dict(), model_path)
print(f"✅ Training complete. Model saved as {model_path}")
