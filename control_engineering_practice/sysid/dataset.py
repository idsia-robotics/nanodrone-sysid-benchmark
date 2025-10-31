import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import os
import json
import joblib
from sklearn.preprocessing import StandardScaler


def quat_SO3_log(q, eps=1e-6):
    """
    Log map from quaternion to so(3) (rotation vector).
    q: (..., 4) tensor [x, y, z, w], assumed normalized.
    Returns: (..., 3) rotation vector
    """
    v = q[..., :3]
    w = q[..., 3:]
    norm_v = torch.linalg.norm(v, dim=-1, keepdim=True)

    angle = 2 * torch.atan2(norm_v, w.clamp(min=-1.0 + eps, max=1.0 - eps))
    small = norm_v < eps

    # default case
    log_q = angle / (norm_v + eps) * v

    # small-angle fallback: log(q) ≈ 2*v
    log_q = torch.where(small, 2.0 * v, log_q)
    return log_q

class QuadDataset(Dataset):
    def __init__(self, df, split='train', train_ratio=.9, valid_ratio=0.1,
                 scale=False, inputs="commands", scaler_path='scalers'):
        """
        One-step dataset: (x, u) -> x_next

        df: pandas DataFrame containing states and inputs
        split: 'train', 'valid', or 'test'
        inputs: 'commands' (thrust/torques) or 'motors' (m1..m4)
        """
        self.xs = []
        self.us = []
        self.x_nexts = []
        self.scale = scale
        self.split = split
        self.scaler_path = scaler_path

        # --- Extract state ---
        vx, vy, vz = df['vx'].values, df['vy'].values, df['vz'].values
        wx, wy, wz = df['wx'].values, df['wy'].values, df['wz'].values
        quat = df[['qx', 'qy', 'qz', 'qw']].values
        pos = df[['x', 'y', 'z']].values

        # state = np.hstack([
        #     pos,
        #     np.stack([vx, vy, vz], axis=1),
        #     quat,
        #     np.stack([wx, wy, wz], axis=1)
        # ])
        # Convert to rotation vector
        quat_torch = torch.from_numpy(quat).float()
        so3_log = quat_SO3_log(quat_torch).numpy()

        # Assemble new state (12D)
        state = np.hstack([
            pos,
            np.stack([vx, vy, vz], axis=1),
            so3_log,
            np.stack([wx, wy, wz], axis=1)
        ])
        print("state shape:", state.shape)

        # --- Extract inputs ---
        if inputs == 'motors':
            u = df[['m1_erpm', 'm2_erpm', 'm3_erpm', 'm4_erpm']].values
        elif inputs == 'commands':
            u = df[['thrust', 'torque_roll', 'torque_pitch', 'torque_yaw']].values
        else:
            raise ValueError("inputs must be 'motors' or 'commands'")

        # --- Build samples ---
        for i in range(len(df) - 1):
            self.xs.append(state[i])
            self.us.append(u[i])
            self.x_nexts.append(state[i + 1])

        self.xs = np.stack(self.xs)
        self.us = np.stack(self.us)
        self.x_nexts = np.stack(self.x_nexts)

        # === Split
        N = len(self.xs)
        train_end = int(N * train_ratio)
        valid_end = train_end + int(N * valid_ratio)

        if split == 'train':
            indices = slice(0, train_end)
        elif split == 'valid':
            indices = slice(train_end, valid_end)
        elif split == 'test':
            indices = slice(valid_end, N)
        else:
            raise ValueError("split must be 'train', 'valid', or 'test'")

        self.xs = torch.tensor(self.xs[indices], dtype=torch.float32)
        self.us = torch.tensor(self.us[indices], dtype=torch.float32)
        self.x_nexts = torch.tensor(self.x_nexts[indices], dtype=torch.float32)

        # === Scaling
        self.x_mean, self.x_std = None, None
        self.u_mean, self.u_std = None, None

        if scale:
            os.makedirs(scaler_path, exist_ok=True)
            scale_file = os.path.join(scaler_path, 'scaling_stats.json')

            if split == 'train':
                self.x_mean = self.xs.mean(dim=0, keepdim=True)
                self.x_std = self.xs.std(dim=0, keepdim=True) + 1e-6
                self.u_mean = self.us.mean(dim=0, keepdim=True)
                self.u_std = self.us.std(dim=0, keepdim=True) + 1e-6

                stats = {
                    'x_mean': self.x_mean.squeeze(0).tolist(),
                    'x_std': self.x_std.squeeze(0).tolist(),
                    'u_mean': self.u_mean.squeeze(0).tolist(),
                    'u_std': self.u_std.squeeze(0).tolist(),
                }
                with open(scale_file, 'w') as f:
                    json.dump(stats, f)
                print(f"📦 Saved scaling stats to: {scale_file}")

            else:
                if not os.path.exists(scale_file):
                    raise FileNotFoundError(f"Scaler file not found at: {scale_file}")
                with open(scale_file, 'r') as f:
                    stats = json.load(f)
                self.x_mean = torch.tensor(stats['x_mean'], dtype=torch.float32).unsqueeze(0)
                self.x_std = torch.tensor(stats['x_std'], dtype=torch.float32).unsqueeze(0)
                self.u_mean = torch.tensor(stats['u_mean'], dtype=torch.float32).unsqueeze(0)
                self.u_std = torch.tensor(stats['u_std'], dtype=torch.float32).unsqueeze(0)
                print(f"📥 Loaded scaling stats from: {scale_file}")

            self.xs = (self.xs - self.x_mean) / self.x_std
            self.x_nexts = (self.x_nexts - self.x_mean) / self.x_std
            self.us = (self.us - self.u_mean) / self.u_std

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.us[idx], self.x_nexts[idx]

class QuadMultiStepDataset(Dataset):
    def __init__(self, df, horizon=10, split='train', train_ratio=1., valid_ratio=0.,
                 scale=False, inputs="motors", scaler_path='scalers', use_quaternions=False):
        """
        Multi-step dataset: (x0, u_seq) -> x_seq

        If horizon == 'full', a single sample is created:
            (x0, [u0 ... u_{T-2}]) -> [x1 ... x_{T-1}]
        """
        self.scale = scale
        self.split = split
        self.scaler_path = scaler_path

        # --- Extract state ---
        vx, vy, vz = df['vx'].values, df['vy'].values, df['vz'].values
        wx, wy, wz = df['wx'].values, df['wy'].values, df['wz'].values
        quat = df[['qx', 'qy', 'qz', 'qw']].values
        pos = df[['x', 'y', 'z']].values

        if use_quaternions:
            state = np.hstack([
                pos,
                np.stack([vx, vy, vz], axis=1),
                quat,
                np.stack([wx, wy, wz], axis=1)
            ])
        else:
            # Convert to rotation vector
            quat_torch = torch.from_numpy(quat).float()
            so3_log = quat_SO3_log(quat_torch).numpy()

            # Assemble new state (12D)
            state = np.hstack([
                pos,
                np.stack([vx, vy, vz], axis=1),
                so3_log,
                np.stack([wx, wy, wz], axis=1)
            ])
            print("state shape:", state.shape)


        # --- Extract inputs ---
        if inputs == 'motors':
            m = df[['m1_erpm', 'm2_erpm', 'm3_erpm', 'm4_erpm']].values
            u = m * 2 * np.pi / (6 * 60)
        elif inputs == 'commands':
            u = df[['thrust', 'torque_roll', 'torque_pitch', 'torque_yaw']].values
        else:
            raise ValueError("inputs must be 'motors' or 'commands'")

        N = len(df)

        # --- Build sequences ---
        if horizon == "full":
            horizon = N - 1  # use all available data
        else:
            horizon = int(horizon)

        max_idx = N - horizon
        xs, us_seq, xs_seq = [], [], []
        for i in range(max_idx):
            xs.append(state[i].reshape(1,-1))
            us_seq.append(u[i:i + horizon])
            xs_seq.append(state[i + 1:i + 1 + horizon])

        # --- Convert to tensors ---
        self.xs = torch.tensor(np.stack(xs), dtype=torch.float32)
        self.us_seq = torch.tensor(np.stack(us_seq), dtype=torch.float32)
        self.xs_seq = torch.tensor(np.stack(xs_seq), dtype=torch.float32)

        # --- Split indices ---
        N_samples = len(self.xs)
        train_end = int(N_samples * train_ratio)
        valid_end = train_end + int(N_samples * valid_ratio)
        if split == 'train':
            self.indices = range(0, train_end)
        elif split == 'valid':
            self.indices = range(train_end, valid_end)
        elif split == 'test':
            self.indices = range(valid_end, N_samples)
        else:
            raise ValueError("split must be 'train', 'valid', or 'test'")

        # --- Scaling placeholders ---
        self.x_mean, self.x_std = None, None
        self.u_mean, self.u_std = None, None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return self.xs[i], self.us_seq[i], self.xs_seq[i]

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

        if fold == "train":
            from sklearn.preprocessing import StandardScaler
            import joblib

            x_scaler = StandardScaler()
            u_scaler = StandardScaler()

            x_flat = final_xs_seq.reshape(-1, final_xs_seq.shape[-1]).numpy()
            u_flat = final_us_seq.reshape(-1, final_us_seq.shape[-1]).numpy()

            # Remove duplicates (within floating tolerance)
            x_flat_unique = np.unique(np.round(x_flat, decimals=6), axis=0)
            u_flat_unique = np.unique(np.round(u_flat, decimals=6), axis=0)

            x_scaler.fit(x_flat_unique)
            u_scaler.fit(u_flat_unique)

            joblib.dump(x_scaler, x_scaler_path)
            joblib.dump(u_scaler, u_scaler_path)
        else:
            import joblib
            x_scaler = joblib.load(x_scaler_path)
            u_scaler = joblib.load(u_scaler_path)

        # Apply transformations
        final_xs = torch.from_numpy(
            x_scaler.transform(final_xs.reshape(-1, final_xs.shape[-1]).numpy())
        ).float()
        final_us_seq = torch.from_numpy(
            u_scaler.transform(final_us_seq.reshape(-1, final_us_seq.shape[-1]).numpy())
        ).float().reshape_as(final_us_seq)
        final_xs_seq = torch.from_numpy(
            x_scaler.transform(final_xs_seq.reshape(-1, final_xs_seq.shape[-1]).numpy())
        ).float().reshape_as(final_xs_seq)

    # Wrap into dataset
    class CombinedMultiStepDataset(torch.utils.data.Dataset):
        def __init__(self, xs, us_seq, xs_seq):
            self.xs = xs
            self.us_seq = us_seq
            self.xs_seq = xs_seq

        def __len__(self):
            return len(self.xs)

        def __getitem__(self, idx):
            return self.xs[idx], self.us_seq[idx], self.xs_seq[idx]

    return CombinedMultiStepDataset(final_xs, final_us_seq, final_xs_seq)

if __name__ == '__main__':

    horizon = 1000
    train_trajs = ["square"]

    train_ds = []
    for traj in train_trajs:
        for run in [1, 2, 3, 4, 5]:
            try:
                file_name = f'{traj}_20251017_run{run}.parquet'
                df = pd.read_parquet(os.path.join('../data/real/processed/new/test', file_name))
                df = df.rename(columns={"torch_yaw": "torque_yaw"})  # fix typo if present
                ds = QuadMultiStepDataset(df, horizon=horizon, split='train')
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

    print(x0.shape, x_seq.shape)

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
