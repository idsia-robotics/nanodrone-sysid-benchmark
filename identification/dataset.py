import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import os, json, joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from pytorch3d.transforms import (
    quaternion_to_axis_angle,
    axis_angle_to_quaternion,
)

def quat_xyzw_to_wxyz(q):
    # (x,y,z,w) → (w,x,y,z)
    return torch.cat([q[..., 3:], q[..., :3]], dim=-1)

def quat_wxyz_to_xyzw(q):
    # (w,x,y,z) → (x,y,z,w)
    return torch.cat([q[..., 1:], q[..., :1]], dim=-1)

# -----------------------------------------------
# Quaternion → SO(3) log map
# -----------------------------------------------
def quat_to_so3_log(q_xyzw):
    """
    q_xyzw: (...,4) quaternion in (x,y,z,w)
    returns rotation vector r in R^3
    """
    q_wxyz = quat_xyzw_to_wxyz(q_xyzw)
    r = quaternion_to_axis_angle(q_wxyz)  # (...,3)
    return r


# -----------------------------------------------
# Unified dataset (one-step or multi-step)
# -----------------------------------------------
class QuadDataset(Dataset):
    def __init__(self, df,
                 horizon=1,
                 inputs="motors",
                 use_quaternions=False):
        """
        General dataset for both one-step and multi-step prediction.

        Args:
            df: pandas DataFrame with quadrotor states and inputs
            horizon: number of prediction steps (1 for one-step)
            split: 'train', 'valid', or 'test'
            scale: whether to normalize x,u using StandardScaler
            inputs: 'motors' or 'commands'
        """
        # --- Build state (12D) ---
        pos = df[['x', 'y', 'z']].values
        vel = df[['vx', 'vy', 'vz']].values
        omega = df[['wx', 'wy', 'wz']].values

        quat = df[['qx', 'qy', 'qz', 'qw']].values
        if use_quaternions:
            rot_repr = quat
        else:
            rot_repr = quat_to_so3_log(torch.from_numpy(quat).float()).numpy()

        state = np.hstack([pos, vel, rot_repr, omega])  # (N,12)
        N = len(state)

        # --- Inputs ---
        if inputs == 'motors':
            m = df[['m1_erpm', 'm2_erpm', 'm3_erpm', 'm4_erpm']].values
            # Convert ERPM → rad/s (assuming 6 electrical per mechanical revolution)
            u = m * 2 * np.pi / (6 * 60)
        elif inputs == 'commands':
            u = df[['thrust', 'torque_roll', 'torque_pitch', 'torque_yaw']].values
        else:
            raise ValueError("inputs must be 'motors' or 'commands'")

        # --- Sequence generation ---
        if horizon == "full":
            horizon = N - 1
        else:
            horizon = int(horizon)

        xs, us_seq, xs_seq = [], [], []
        for i in range(N - horizon):
            xs.append(state[i].reshape(1, -1))
            us_seq.append(u[i:i + horizon])
            xs_seq.append(state[i + 1:i + 1 + horizon])

        self.xs = torch.tensor(np.stack(xs), dtype=torch.float32)
        self.us_seq = torch.tensor(np.stack(us_seq), dtype=torch.float32)
        self.xs_seq = torch.tensor(np.stack(xs_seq), dtype=torch.float32)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.us_seq[idx], self.xs_seq[idx]

def combine_concat_dataset(concat_dataset, scale=False, fold="train", scaler_dir="./scalers"):
    """
    Combines a ConcatDataset of QuadMultiStepDataset instances into
    a single dataset (x_s, u_seq, x_seq) for training.

    Each tuple corresponds to one time window across all experiments.
    """
    if not isinstance(concat_dataset, ConcatDataset):
        raise TypeError("Input must be a ConcatDataset.")
    assert fold in ["train", "valid", "test"]

    os.makedirs(scaler_dir, exist_ok=True)

    # Collect all samples across all datasets
    all_xs, all_us_seq, all_xs_seq = [], [], []

    for ds in concat_dataset.datasets:
        # Each ds is already a QuadMultiStepDataset
        xs, us_seq, xs_seq = [], [], []
        for i in range(len(ds)):
            x, u, xseq = ds[i]
            xs.append(x)
            us_seq.append(u)
            xs_seq.append(xseq)

        all_xs.append(torch.stack(xs))
        all_us_seq.append(torch.stack(us_seq))
        all_xs_seq.append(torch.stack(xs_seq))

    # Merge all experiments (concatenate along batch dimension)
    final_xs = torch.cat(all_xs, dim=0)
    final_us_seq = torch.cat(all_us_seq, dim=0)
    final_xs_seq = torch.cat(all_xs_seq, dim=0)

    print(f"✅ Combined dataset shapes:")
    print(f"  x0:     {final_xs.shape}")
    print(f"  u_seq:  {final_us_seq.shape}")
    print(f"  x_seq:  {final_xs_seq.shape}")

    # Optional: apply scaling
    if scale:
        x_scaler_path = os.path.join(scaler_dir, "x_scaler.pkl")
        u_scaler_path = os.path.join(scaler_dir, "u_scaler.pkl")

        # state = [p(0-2), v(3-5), r(6-8), omega(9-11)]
        idx_scale    = np.array([0, 1, 2, 3, 4, 5, 9, 10, 11])  # p, v, omega
        idx_no_scale = np.array([6, 7, 8])                      # rotation log

        # We want final_xs to end up as (N, 12) no matter what
        N_x  = final_xs.shape[0]
        D_x  = final_xs.shape[-1]        # 12
        N_xs = final_xs_seq.shape[0]
        H_xs = final_xs_seq.shape[1]

        if fold == "train":
            from sklearn.preprocessing import StandardScaler
            import joblib

            x_scaler = StandardScaler()
            u_scaler = StandardScaler()

            # Flatten states: squeeze any middle dims into feature dim
            xs_flat     = final_xs.reshape(-1, D_x).numpy()          # (N, 12) or (N*?, 12)
            xs_seq_flat = final_xs_seq.reshape(-1, D_x).numpy()      # (N*H, 12)

            # Fit x_scaler only on non-rotation dims
            x_scaler.fit(
                np.concatenate([
                    xs_flat[:, idx_scale],
                    xs_seq_flat[:, idx_scale]
                ], axis=0)
            )

            # Inputs: standard scaling
            u_flat = final_us_seq.reshape(-1, final_us_seq.shape[-1]).numpy()
            u_scaler.fit(u_flat)

            joblib.dump(x_scaler, x_scaler_path)
            joblib.dump(u_scaler, u_scaler_path)
        else:
            import joblib
            x_scaler = joblib.load(x_scaler_path)
            u_scaler = joblib.load(u_scaler_path)

        # ---- Apply transformations ----
        # Flatten (and squeeze any singleton dims) to (N, 12) / (N*H, 12)
        xs_flat     = final_xs.reshape(-1, D_x).numpy()
        xs_seq_flat = final_xs_seq.reshape(-1, D_x).numpy()

        xs_scaled     = xs_flat.copy()
        xs_seq_scaled = xs_seq_flat.copy()

        # Scale only p, v, omega
        xs_scaled[:, idx_scale]     = x_scaler.transform(xs_flat[:, idx_scale])
        xs_seq_scaled[:, idx_scale] = x_scaler.transform(xs_seq_flat[:, idx_scale])

        # Keep rotations untouched
        xs_scaled[:, idx_no_scale]     = xs_flat[:, idx_no_scale]
        xs_seq_scaled[:, idx_no_scale] = xs_seq_flat[:, idx_no_scale]

        # Restore shapes:
        # final_xs -> (N_x, 12)
        final_xs = torch.from_numpy(xs_scaled).float().reshape(N_x, D_x)

        # final_xs_seq -> (N_xs, H_xs, 12)
        final_xs_seq = torch.from_numpy(xs_seq_scaled).float().reshape(
            N_xs, H_xs, D_x
        )

        # Inputs: scale normally and restore original shape
        final_us_seq = torch.from_numpy(
            u_scaler.transform(
                final_us_seq.reshape(-1, final_us_seq.shape[-1]).numpy()
            )
        ).float().reshape_as(final_us_seq)
    else:
        x_scaler = None
        u_scaler = None


    # Wrap into dataset
    class CombinedDataset(torch.utils.data.Dataset):
        def __init__(self, xs, us_seq, xs_seq, x_scaler=None, u_scaler=None):
            self.xs = xs
            self.us_seq = us_seq
            self.xs_seq = xs_seq
            self.x_scaler = x_scaler
            self.u_scaler = u_scaler

        def __len__(self):
            return len(self.xs)

        def __getitem__(self, idx):
            return self.xs[idx], self.us_seq[idx], self.xs_seq[idx]

    return CombinedDataset(final_xs, final_us_seq, final_xs_seq, x_scaler, u_scaler)


if __name__ == '__main__':

    horizon = "full"
    train_trajs = ["square"]

    train_ds = []
    for traj in train_trajs:
        for run in [1, 2, 3, 4, 5]:
            try:
                file_name = f'{traj}_20251017_run{run}.parquet'
                df = pd.read_parquet(os.path.join('../data/real/processed/new/test', file_name))
                df = df.rename(columns={"torch_yaw": "torque_yaw"})  # fix typo if present
                ds = QuadDataset(df, horizon=horizon)
                train_ds.append(ds)
            except Exception as e:
                print(e)
                continue


    train_dataset = combine_concat_dataset(ConcatDataset(train_ds), scale=False, fold="train")
    # === Load first batch ===
    loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    x0, u_seq, x_seq = next(iter(loader))

    # Ensure both are 2D (N, 13)
    x0 = x0.squeeze(0)  # (1, 13)
    if x0.ndim == 1:
        x0 = x0.unsqueeze(0)  # -> (1, 13)
    x_seq = x_seq.squeeze(0)  # (N, 13)

    # === Reconstruct trajectory ===
    x_traj = torch.cat([x0, x_seq], dim=0).cpu().numpy()
    timesteps = np.arange(len(x_traj))

    # === Labels for 12D state (angles instead of quaternion) ===
    labels = [
        "x [m]", "y [m]", "z [m]",
        "vx [m/s]", "vy [m/s]", "vz [m/s]",
        "rx [rad]", "ry [rad]", "rz [rad]",  # <-- SO(3) log vector
        "wx [rad/s]", "wy [rad/s]", "wz [rad/s]"
    ]

    plt.figure(figsize=(14, 9))
    for i in range(x_traj.shape[1]):
        plt.subplot(4, 3, i + 1)
        plt.plot(timesteps, x_traj[:, i])
        plt.title(labels[i])
        plt.grid(True, alpha=0.3)
        plt.xlabel("timestep")

    plt.tight_layout()
    plt.suptitle("First trajectory (x0 + x_seq) with SO(3) angles", fontsize=16, y=1.02)
    plt.show()
