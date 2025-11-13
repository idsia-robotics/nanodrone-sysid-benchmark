import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


from utils.plot_utils import setup_matplotlib

setup_matplotlib()

experiments = ['square', 'random', 'multisine']
runs = [1, 2, 3, 4]
df_real_dict = {}

for experiment in experiments:
    file_name = f'{experiment}_20251017'
    for run in runs:
        try:
            file_name_real = f'{file_name}_run{run}.csv'
            df_real = pd.read_csv(os.path.join('../data/real/processed/train/', file_name_real))
            # Store dynamically
            df_real_dict[f'run{run}'] = df_real
        except Exception as e:
            print(e)
            continue

# --- Create 4x4 layout ---
fig, axs = plt.subplots(4, 3, figsize=(15, 8), sharex=True, dpi=200)

# --- Define state variables (fill first 3 columns) ---
state_cols = [
    ['x', 'y', 'z'],
    ['roll', 'pitch', 'yaw'],
    ['vx', 'vy', 'vz'],
    ['wx', 'wy', 'wz']
]
state_labels = [
    ['$x$ [m]', '$y$ [m]', '$z$ [m]'],
    [r'$\varphi$ [rad]', r'$\theta$ [rad]', r'$\psi$ [rad]'],
    [r'$v_x$ [m/s]', '$v_y$ [m/s]', '$v_z$ [m/s]'],
    [r'$\omega_x$ [rad/s]', r'$\omega_y$ [rad/s]', r'$\omega_z$ [rad/s]']
]

# --- Colors / styles for runs ---
colors = plt.cm.tab10.colors  # up to 10 distinct colors

# --- Plot states ---
for r in range(4):
    for c in range(3):
        col = state_cols[r][c]
        ax = axs[r, c]

        # All real runs
        for i, (name, df_real) in enumerate(df_real_dict.items()):
            label = f"real run {i+1}" if (r == 0 and c == 0) else None
            ax.plot(df_real['t'], df_real[col], color=colors[0], alpha=0.8, label=label)

        ax.set_ylabel(state_labels[r][c])
        ax.grid(True)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# --- Shared X label ---
fig.text(0.5, 0.04, "Time [s]", ha='center', va='center', fontsize=14)

# --- Combined legend ---
handles, labels = [], []
for ax in axs.flat:
    h, l = ax.get_legend_handles_labels()
    for handle, label in zip(h, l):
        if label and label not in labels:
            handles.append(handle)
            labels.append(label)
fig.legend(handles, labels, loc='upper center', ncols=6, bbox_to_anchor=(0.5, 1.01))

# --- Layout tweaks ---
plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.1, wspace=0.35)
plt.show()
#%%
N_start = 0
N_end = N_start + 5000
for i, (name, df_real) in enumerate(df_real_dict.items()):
    plt.plot(df_real['m1_rads'][N_start:N_end], color='tab:blue', alpha=0.4, label=f'real run {i+1}')
    plt.plot(df_real['m3_rads'][N_start:N_end], color='tab:orange', alpha=0.4, label=f'real run {i+1}')
#%%

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

m = 0.045   # kg
g = 9.81    # m/s^2
L = 0.035   # m
J = np.diag([2.3951e-5, 2.3951e-5, 3.2347e-5])  # kg·m²

for name, df_real in df_real_dict.items():
    # --- Normalize quaternion just in case ---
    quat = df_real[["qx", "qy", "qz", "qw"]].to_numpy()
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)

    # --- Transform accelerations (world → body) ---
    ag_wf = df_real[["ax", "ay", "az"]].to_numpy()       # world-frame acceleration
    r = R.from_quat(quat)
    ag_bf = r.apply(ag_wf, inverse=True)                 # body-frame acceleration
    df_real[["ax_body", "ay_body", "az_body"]] = ag_bf   # store it

    # --- Compute total thrust from acceleration ---
    # (assuming z-axis points downward in world frame)
    # T_bf_z = -m * (a_bz - g_body) if you want vertical thrust
    # here we take the *body-frame z-acceleration* as proxy
    T_bf = m * (ag_bf[:, 2] + g)   # N (body-frame z-axis, adding gravity)

    # --- Compute motor thrust proxy (sum of squared speeds) ---
    motor_cols = [f"m{i}_rads2" for i in range(1, 5)]
    T_mot = df_real[motor_cols].sum(axis=1)              # sum of ω² per sample

    # --- Scatter plot for correlation (normalized) ---
    plt.figure(figsize=(5, 4))
    plt.scatter(T_mot, T_bf, s=5, alpha=0.5)
    plt.xlabel("Sum of motor speeds squared (rad²/s²)")
    plt.ylabel("Total body-frame thrust (N)")
    plt.title(f"{name}")
    plt.grid(True)
    plt.tight_layout()

    # Optionally: store results back
    df_real["T_bf"] = T_bf
    df_real["T_mot_sum"] = T_mot
    df_real_dict[name] = df_real