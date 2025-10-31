#!/usr/bin/env python3
"""
Estimate the effective (possibly scaled) dt in a dataset by comparing
position increments with corresponding velocities.

Works both for scaled and unscaled datasets.
If a scaler directory is provided, computes the theoretical scaled dt too.
"""

import os
import numpy as np
import pandas as pd
import test_torch
from test_torch.utils.data import DataLoader, ConcatDataset
import joblib
import matplotlib.pyplot as plt

from idsia_mpc.control_engineering_practice.sysid.dataset import (
    QuadMultiStepDataset,
    combine_concat_dataset,
)

# ======================================================
# === USER CONFIGURATION ===============================
# ======================================================
DATA_DIR = "../../data/real/processed/new/train"
SCALER_DIR = "/home/rbusetto/nanodrone-sysid-mpc/idsia_mpc/control_engineering_practice/sysid/train/lstm_random1_melon_square/scalers"
HORIZON = 2000
BATCH_SIZE = 1
SCALE = True        # if True, will use scaled dataset
DT_PHYSICAL = 0.01  # seconds

# ======================================================
# === LOAD DATA ========================================
# ======================================================
def load_dataset(trajs, scale, fold="train"):
    datasets = []
    for traj in trajs:
        for run in [1, 2, 3, 4, 5]:
            file_name = f"{traj}_20251017_run{run}.parquet"
            path = os.path.join(DATA_DIR, file_name)
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_parquet(path)
                df = df.rename(columns={"torch_yaw": "torque_yaw"})
                ds = QuadMultiStepDataset(df, horizon=HORIZON, split="train")
                datasets.append(ds)
            except Exception as e:
                print(f"⚠️ Skipped {path}: {e}")
    if not datasets:
        raise RuntimeError("No datasets found.")
    return combine_concat_dataset(ConcatDataset(datasets), scale=scale, fold=fold, scaler_dir=SCALER_DIR)

train_trajs = ["random1", "melon", "square"]
train_dataset = load_dataset(train_trajs, scale=SCALE, fold="train")
loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ======================================================
# === EXTRACT FIRST TRAJECTORY =========================
# ======================================================
x0, u_seq, x_seq = next(iter(loader))

# Ensure both are 2D (N, 13)
x0 = x0.squeeze(0)  # (1, 13)
if x0.ndim == 1:
    x0 = x0.unsqueeze(0)  # -> (1, 13)
x_seq = x_seq.squeeze(0)  # (N, 13)

x_traj = torch.cat([x0, x_seq], dim=0).cpu().numpy()  # (N, 13)

# ======================================================
# === COMPUTE EMPIRICAL SCALED DT ======================
# ======================================================
pos = x_traj[:, 0:3]
vel = x_traj[:, 6:9]

dx = np.diff(pos, axis=0)
v = vel[:-1, :]

mask = np.linalg.norm(v, axis=1) > 1e-6
ratio = np.divide(dx[mask], v[mask], out=np.zeros_like(dx[mask]), where=v[mask]!=0)

dt_est_per_axis = np.nanmedian(ratio, axis=0)
dt_est_mean = np.nanmean(dt_est_per_axis)

print("\n=== EMPIRICAL DT ESTIMATION ===")
print(f"Per-axis dt': {dt_est_per_axis}")
print(f"Mean effective dt': {dt_est_mean:.6f}")

# ======================================================
# === COMPUTE THEORETICAL SCALED DT (if available) =====
# ======================================================
scaler_path = os.path.join(SCALER_DIR, "x_scaler.pkl")
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    sigma_x = np.mean(scaler.scale_[0:3])
    sigma_v = np.mean(scaler.scale_[6:9])
    dt_theoretical = DT_PHYSICAL * (sigma_v / sigma_x)

    print("\n=== THEORETICAL SCALED DT ===")
    print(f"σ_pos = {sigma_x:.6f}, σ_vel = {sigma_v:.6f}")
    print(f"Δt_physical = {DT_PHYSICAL}")
    print(f"Δt_scaled (theoretical) = {dt_theoretical:.6f}")
    print(f"Ratio empirical/theoretical = {dt_est_mean / dt_theoretical:.3f}")
else:
    print("\n(no scaler found, skipping theoretical comparison)")

print("\n✅ Done.")

# ======================================================
# === COMPUTE DERIVED VELOCITIES (FROM POS) ============
# ======================================================
# Using the *estimated* scaled dt
v_from_pos = np.zeros_like(vel)
v_from_pos[1:, :] = dx / dt_est_mean

# Pad to match dimensions
t = np.arange(len(vel)) * 1  # "timesteps" (scaled)
axes = ["x", "y", "z"]

plt.figure(figsize=(12, 8))
for i, ax_name in enumerate(axes):
    plt.subplot(3, 1, i + 1)
    plt.plot(t, vel[:, i], label=f"true v{ax_name}", color="tab:blue")
    plt.plot(t, v_from_pos[:, i], "--", label=f"derived v{ax_name}", color="tab:orange")
    plt.ylabel(f"v{ax_name}")
    plt.grid(True, alpha=0.3)
    if i == 0:
        plt.title("True velocities vs. derived (from position differences)")
plt.xlabel("timestep")
plt.legend()
plt.tight_layout()
plt.show()
