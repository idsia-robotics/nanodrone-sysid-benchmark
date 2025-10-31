import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# === CONFIG ===
horizon = 50
data_type = "real"
traj_id = "1809"

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
# --- Baseline: constant predictor (x_pred_h = x_t)
# =====================================================
# The baseline dataset must have N - horizon samples
data = {"t": t_vec[1 : N - horizon + 1]}  # perfectly aligned length N - horizon

for name in state_names:
    x_logged = df[name].to_numpy()

    # Ground truth for h = 1
    data[name] = x_logged[1 : N - horizon + 1]

    # 1-step prediction
    data[f"{name}_pred"] = x_logged[: N - horizon]

    # Multistep predictions (constant baseline)
    for h in range(1, horizon + 1):
        col_name = f"{name}_pred_h{h}"
        # constant predictor: always x_t
        data[col_name] = x_logged[: N - horizon]

df_pred = pd.DataFrame(data)
print(f"✅ Baseline dataset shape: {df_pred.shape}")

# --- Save ---
out_dir = f"../out/predictions/{data_type}/baseline_model_multistep"
os.makedirs(out_dir, exist_ok=True)
csv_path = f"{out_dir}/experiment_{traj_type}_{traj_id}.csv"
parquet_path = f"{out_dir}/experiment_{traj_type}_{traj_id}.parquet"
df_pred.to_csv(csv_path, index=False)
df_pred.to_parquet(parquet_path, index=False)

print(f"✅ Saved baseline predictions:\n   → {csv_path}\n   → {parquet_path}")

# =====================================================
# --- Performance metrics for each horizon ---
# =====================================================
print("\n=== 🔍 Baseline consistency check: logged vs constant prediction ===")
print(f"{'State':<8} | {'H':>3} | {'MeanDiff':>12} | {'RMSE':>12}")
print("-" * 50)

for name in state_names:
    for h in range(1, horizon + 1):
        true = df[name].to_numpy()[h : N - horizon + h]
        pred = df[name].to_numpy()[: N - horizon]
        err = true - pred
        err = err[np.isfinite(err)]
        mean_err = np.mean(err)
        rmse = np.sqrt(np.mean(err**2))
        print(f"{name:<8} | {h:>3d} | {mean_err:>12.3e} | {rmse:>12.3e}")
    print("-" * 50)

# =====================================================
# --- Optional: Plot comparison for key states ---
# =====================================================
plt.figure(figsize=(8, 4), dpi=120)
plt.plot(df_pred["t"][:200], df_pred["x"][:200], label="logged x")
plt.plot(df_pred["t"][:200], df_pred["x_pred_h1"][:200], "--", label="constant pred (h=1)")
plt.xlabel("Time [s]")
plt.ylabel("x [m]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
