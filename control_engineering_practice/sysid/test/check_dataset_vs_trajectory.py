import os
import numpy as np
import pandas as pd
from idsia_mpc.control_engineering_practice.sysid.dataset import QuadMultiStepDataset
from control_engineering_practice.quat_utils import quat_to_euler

# === CONFIG ===
dt = 0.01
horizon = 50
type = "sim"   # or "real"
traj_type = "points"
traj_id = "1848"

if type == "sim":
    traj_type = "points"
    file_name_sim = f'experiment_points_{traj_id}.parquet'
    df = pd.read_parquet(os.path.join('../../data/sim/', file_name_sim))
elif type == "real":
    traj_type = "random"
    file_name_real = f'experiment_{traj_type}_{traj_id}.parquet'
    df = pd.read_parquet(os.path.join('../../data/real/processed/', file_name_real))
    df = df.rename(columns={"torch_yaw": "torque_yaw"})  # fix typo if present

mode = "neural"  # or "neural", "residual"
model_path = f"../out/models/{mode}_quad_model_multistep_{type}_{traj_type}_{traj_id}.pt"
print("✅ Model path:", model_path)

# === Load data ===
dataset = QuadMultiStepDataset(df, horizon=horizon, split='train', scale=False)

state_cols = ["x","y","z","vx","vy","vz","qx","qy","qz","qw","wx","wy","wz"]
u_cols = ["thrust","torque_roll","torque_pitch","torque_yaw"]

X_logged = df[state_cols].to_numpy()
U_logged = df[u_cols].to_numpy()
N = len(U_logged)
print(f"🔹 Trajectory length: {N}")



print(f"🔹 Dataset created with {len(dataset)} samples (expected {N - horizon})")

# --- Verify structure ---
tol = 1e-10
n_errors = 0
for i in range(len(U_logged) - horizon):
    x_t, u_seq, x_seq = dataset[i]
    # Check x_t
    if not np.allclose(x_t, X_logged[i], atol=tol):
        print(f"❌ Mismatch in x_t at i={i}")
        n_errors += 1
        break
    # Check u_seq
    u_expected = U_logged[i:i+horizon]
    if not np.allclose(u_seq, u_expected, atol=tol):
        print(f"❌ Mismatch in u_seq at i={i}")
        n_errors += 1
        break
    # Check x_seq
    x_expected = X_logged[i+1:i+1+horizon]
    if not np.allclose(x_seq, x_expected, atol=tol):
        print(f"❌ Mismatch in x_seq at i={i}")
        n_errors += 1
        break

if n_errors == 0:
    print(f"✅ Dataset perfectly matches raw trajectory structure.")
else:
    print(f"⚠️ Found {n_errors} mismatches.")

# --- Optional consistency checks ---
print("\n--- Additional checks ---")
print(f"Dataset xs shape:      {dataset.xs.shape}")
print(f"Dataset us_seq shape:  {dataset.us_seq.shape}")
print(f"Dataset xs_seq shape:  {dataset.xs_seq.shape}")
print(f"Expected (N-horizon):  {N - horizon}")
