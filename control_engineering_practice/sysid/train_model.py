import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Import your model and dataset classes ===
from models import PhysQuadModel, NeuralQuadModel, ResidualQuadModel, QuadMultiStepModel
from dataset import QuadDataset, QuadMultiStepDataset

import os

from losses import QuadStateMSELoss, ScaledMSELoss

# === Example: set GPU 1 only ===
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# === Config ===
scale = False

pretrained = False
pretrained_horizon = 20
horizon = 40
custom_loss = False

# === Build model name dynamically ===
mode = "onestep" if onestep else f"multistep_h{horizon}"
mode_pretrained = f"multistep_h{pretrained_horizon}" if pretrained else None
learn = "learnparams" if learn_params else "fixedparams"
res = "residual" if residual else "noresidual"
loss_type = "customloss" if custom_loss else "mseloss"
model_type = "bis" if bis else ""
scale_type = 'scaled' if scale else ""

model_path = f"out/quad_hybrid_model{model_type}_{mode}_{learn}_{res}_{loss_type}_{scale_type}.pt"
print("✅ Model path:", model_path)

if pretrained:
    pretrained_model_path = f"quad_hybrid_model_{mode_pretrained}_{learn}_{res}_{loss_type}.pt"
    print("✅ Pretrained model path:", pretrained_model_path)


epochs = 10_000
batch_size = 32
lr_start = 1e-4
lr_end = 1e-8

# === Load data ===
df = pd.read_parquet('../data/parquets/rosbag2_2025_05_19-19_10_43-run6.parquet')

if onestep:
    train_dataset = QuadDataset(df, split='train', scale=scale)
    valid_dataset = QuadDataset(df, split='valid', scale=scale)
    test_dataset = QuadDataset(df, split='test', scale=scale)

    # Compute scale_vector from training data (standard deviation per state dimension)
    scale_vector = torch.std(train_dataset.xs, dim=0).numpy()
    # Optional: Override quaternion std if needed
    scale_vector[6:10] = 0.5  # prevent quaternion domination
else:
    train_dataset = QuadMultiStepDataset(df, horizon=horizon, split='train', scale=scale)
    valid_dataset = QuadMultiStepDataset(df, horizon=horizon, split='valid', scale=scale)
    test_dataset = QuadMultiStepDataset(df, horizon=horizon, split='test', scale=scale)
    # Compute scale vector from x0
    scale_vector = torch.std(train_dataset.xs, dim=0).numpy()
    scale_vector[6:10] = 0.5

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# === Dynamics parameters ===
params = {
    "g": 9.81,
    "m": 0.0393,
    "J": np.diag([1.43e-6, 1.43e-6, 2.89e-6]),
    "thrust_to_weight": 4.0,
    "max_torque": np.array([1e-3, 1e-3, 3e-4]),
}

# === Initialize model ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if onestep:
    if bis:
        model = QuadModelLearnableMotors(params, dt=0.01, learn_params=learn_params, use_residual=residual).to(device)
    else:
        model = QuadModel(params, dt=0.01, learn_params=learn_params, use_residual=residual).to(device)
else:
    model = QuadMultiStepModel(params, dt=0.01, learn_params=learn_params, use_residual=residual).to(device)

# === Load model if exists ===
if pretrained:
    if os.path.exists(pretrained_model_path):
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        print(f"✅ Loaded existing model from {pretrained_model_path}")
    else:
        print(f"🔧 No existing model found. Training from scratch.")
else:
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Loaded existing model from {model_path}")
    else:
        print(f"🔧 No existing model found. Training from scratch.")

# === Optimizer, scheduler ===
optimizer = optim.Adam(model.parameters(), lr=lr_start)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_end)


if custom_loss:
    criterion = QuadStateMSELoss(model)
else:
    criterion = nn.MSELoss()
    # criterion = ScaledMSELoss(scale_vector)

# === Initialize best validation loss ===
best_val_loss = float('inf')

# === Training loop ===
for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]"):
        optimizer.zero_grad()

        if onestep:
            x, duty, V, x_next = [b.to(device) for b in batch]
            pred_next = model(x, duty, V)
            loss = criterion(pred_next, x_next)
        else:
            x0, duty_seq, V_seq, x_seq = [b.to(device) for b in batch]
            pred_seq = model(x0, duty_seq, V_seq)
            pred_flat = pred_seq.view(-1, pred_seq.shape[-1])
            x_seq_flat = x_seq.view(-1, x_seq.shape[-1])
            loss = criterion(pred_flat, x_seq_flat)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # === Validation ===
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for batch in valid_loader:
            if onestep:
                x, duty, V, x_next = [b.to(device) for b in batch]
                pred_next = model(x, duty, V)
                loss = criterion(pred_next, x_next)
            else:
                x0, duty_seq, V_seq, x_seq = [b.to(device) for b in batch]
                pred_seq = model(x0, duty_seq, V_seq)
                pred_flat = pred_seq.view(-1, pred_seq.shape[-1])
                x_seq_flat = x_seq.view(-1, x_seq.shape[-1])
                loss = criterion(pred_flat, x_seq_flat)

            valid_loss += loss.item()

    avg_valid_loss = valid_loss / len(valid_loader)
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch + 1}, LR: {current_lr:.8f}, Train Loss: {avg_train_loss:.6f}, Valid Loss: {avg_valid_loss:.6f}")

    # === Save model if validation loss improved ===
    if avg_valid_loss < best_val_loss:
        best_val_loss = avg_valid_loss
        torch.save(model.state_dict(), model_path)
        print(f"💾 Saved new best model at epoch {epoch + 1} with valid loss {avg_valid_loss:.6f}")

    scheduler.step()

# === Save model ===
torch.save(model.state_dict(), model_path)
print(f"✅ Training complete. Model saved as {model_path}")

# === Final learned physical parameters ===
print("✅ Final learned physical parameters:")
print(f"Mass [kg]: {model.m.item()}")
print(f"Inertia matrix J [kg*m^2]:\n{model.J.detach().cpu().numpy()}")
print(f"Thrust-to-weight ratio [-]: {model.thrust_to_weight.item()}")
print(f"Max torque [Nm]: {model.max_torque.detach().cpu().numpy()}")
print(f"CT [N/(rad/s)^2]: {model.CT.item()}")
print(f"CQ [Nm/(rad/s)^2]: {model.CQ.item()}")
