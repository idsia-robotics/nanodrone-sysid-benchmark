import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# === CONFIG ===
data_type = "real"
traj_id = "1848"
horizon = 1  # only one-step baseline

if data_type == "sim":
    traj_type = "points"
    file_name_sim = f"experiment_points_{traj_id}.parquet"
    df = pd.read_parquet(os.path.join("../../data/sim/", file_name_sim))
elif data_type == "real":
    traj_type = "melon" if traj_id == "1809" else "random"
    file_name_real = f"experiment_{traj_type}_{traj_id}.parquet"
    df = pd.read_parquet(os.path.join("../../data/real/processed/", file_name_real))
    df = df.rename(columns={"torch_yaw": "torque_yaw"})  # fix typo if present

# --- States and Inputs ---
state_names = ["x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw", "wx", "wy", "wz"]
t_vec = df["t"].to_numpy()
N = len(df)

# =====================================================
# --- Baseline: constant predictor (x_pred_h1 = x_t)
# =====================================================
data = {"t": t_vec[1:N]}  # shift by 1 to align with prediction target

for name in state_names:
    x_logged = df[name].to_numpy()

    # Ground truth (aligned one step ahead)
    data[name] = x_logged[1:N]

    # One-step constant prediction (predict next = current)
    data[f"{name}_pred"] = x_logged[:N - 1]

# === Create dataframe ===
df_pred = pd.DataFrame(data)
print(f"✅ Baseline 1-step dataset shape: {df_pred.shape}")

# --- Save ---
out_dir = f"../out/predictions/{data_type}/baseline_model_onestep"
os.makedirs(out_dir, exist_ok=True)
csv_path = f"{out_dir}/experiment_{traj_type}_{traj_id}.csv"
parquet_path = f"{out_dir}/experiment_{traj_type}_{traj_id}.parquet"
df_pred.to_csv(csv_path, index=False)
df_pred.to_parquet(parquet_path, index=False)
print(f"✅ Saved baseline 1-step predictions:\n   → {csv_path}\n   → {parquet_path}")

# =====================================================
# --- Consistency check ---
# =====================================================
print("\n=== 🔍 Baseline 1-step consistency check ===")
print(f"{'State':<8} | {'MeanDiff':>12} | {'RMSE':>12}")
print("-" * 40)

for name in state_names:
    true = df[name].to_numpy()[1:N]
    pred = df[name].to_numpy()[:N - 1]
    err = true - pred
    mean_err = np.mean(err)
    rmse = np.sqrt(np.mean(err**2))
    print(f"{name:<8} | {mean_err:>12.3e} | {rmse:>12.3e}")
print("-" * 40)

# =====================================================
# --- Optional: quick plot ---
# =====================================================
plt.figure(figsize=(8, 4), dpi=120)
plt.plot(df_pred["t"][:200], df_pred["x"][:200], label="logged x")
plt.plot(df_pred["t"][:200], df_pred["x_pred"][:200], "--", label="baseline 1-step")
plt.xlabel("Time [s]")
plt.ylabel("x [m]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
