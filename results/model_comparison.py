#%%
import os, sys
import numpy as np
import pandas as pd

from utils.latex_utils import print_latex_table_results
from utils.plot_utils import setup_matplotlib
#%%
setup_matplotlib()

# === Config ===
train_trajs = ["random", "square", "chirp"]
test_trajs = ["melon"]

# === File paths ===
OUT_FOLDER = os.path.join(
    "..",
    "identification",
    "out",
    "predictions",
    "real"
)

# Ensure the output folder exists
os.makedirs(OUT_FOLDER, exist_ok=True)

# Construct file paths
train_name = "_".join(train_trajs)
test_name = "_".join(test_trajs)

file_lstm = os.path.join(
    OUT_FOLDER,
    f"lstm_v2_{train_name}_model_multistep",
    f"{test_name}_multistep.csv"
)

file_base = os.path.join(
    OUT_FOLDER,
    "baseline_model_multistep",
    f"{test_name}_multistep.csv"
)

file_neur = os.path.join(
    OUT_FOLDER,
    f"neural_v2_{train_name}_model_multistep",
    f"{test_name}_multistep.csv"
)

file_phys = os.path.join(
    OUT_FOLDER,
    f"physics_model_multistep",
    f"{test_name}_multistep.csv"
)
#
file_res = os.path.join(
    OUT_FOLDER,
    f"phys+res_{train_name}_model_multistep",
    f"{test_name}_multistep.csv"
)

print("LSTM file path:", file_lstm)
print("Baseline file path:", file_base)

# === Read CSVs ===
df_lstm = pd.read_csv(file_lstm)
df_base = pd.read_csv(file_base)
df_neur = pd.read_csv(file_neur)
df_phys = pd.read_csv(file_phys)
df_res = pd.read_csv(file_res)

print("✅ Loaded datasets:")
print(f"  LSTM model: {df_lstm.shape}")
print(f"  Baseline model: {df_base.shape}")
print(f"  Neural model: {df_neur.shape}")
print(f"  Physics model: {df_phys.shape}")
print(f"  Residual model: {df_res.shape}")
#%%
import torch
from pytorch3d.transforms import quaternion_to_axis_angle, axis_angle_to_quaternion
from scipy.spatial.transform import Rotation as R

def add_rotation_columns(df):
    df = df.copy()
    new_cols = {}

    # Find all rotation-vector triplets: rx*, ry*, rz*
    rx_cols = [c for c in df.columns if c.startswith("rx")]

    for rx_col in rx_cols:
        suffix = rx_col[2:]
        ry_col = f"ry{suffix}"
        rz_col = f"rz{suffix}"

        if ry_col not in df.columns or rz_col not in df.columns:
            continue

        # Extract rotation vectors
        r = df[[rx_col, ry_col, rz_col]].to_numpy(float)

        # Convert to quaternion (N,4)
        q = so3_log_to_quat_np(r)

        new_cols[f"qx{suffix}"] = q[:, 0]
        new_cols[f"qy{suffix}"] = q[:, 1]
        new_cols[f"qz{suffix}"] = q[:, 2]
        new_cols[f"qw{suffix}"] = q[:, 3]

        # Euler angles
        e = quat_to_euler_np(q)

        new_cols[f"roll{suffix}"]  = e[:, 0]
        new_cols[f"pitch{suffix}"] = e[:, 1]
        new_cols[f"yaw{suffix}"]   = e[:, 2]

        # Degrees
        new_cols[f"roll{suffix}_deg"]  = np.degrees(e[:, 0])
        new_cols[f"pitch{suffix}_deg"] = np.degrees(e[:, 1])
        new_cols[f"yaw{suffix}_deg"]   = np.degrees(e[:, 2])

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df
#%%
# ---------------------------------------------------------
# === Apply to all dataframes ===
# ---------------------------------------------------------
df_base, df_lstm, df_neur, df_phys, df_res = [
    add_rotation_columns(df)
    for df in [df_base, df_lstm, df_neur, df_phys, df_res]
]

#%%
# --- Config ---
max_horizon = 50

metrics_base  = compute_errors(df_base,  max_horizon)
metrics_lstm  = compute_errors(df_lstm,  max_horizon)
metrics_neur  = compute_errors(df_neur,  max_horizon)
metrics_phys  = compute_errors(df_phys,  max_horizon)
metrics_res   = compute_errors(df_res,   max_horizon)
#%%
import matplotlib.pyplot as plt

# ============================================================
# === FIGURE: Position, Angular velocity, Orientation errors ==
# ============================================================

fig, axs = plt.subplots(1, 4, figsize=(12, 2), sharex=True)

metric_names = ["pos", "vel", "rot", "omega"]
ylabels = [
    r"$\mathrm{MAE}_{e_p,h}$  [m]",
    r"$\mathrm{MAE}_{e_v,h}$  [m/s]",
    r"$\mathrm{MAE}_{e_R,h}$  [rad]",
    r"$\mathrm{MAE}_{e_{\omega},h}$  [rad/s]"
]

# Plotting order: Baseline LAST
plot_order = ["Physics", "Residual", "Phys+Res", "LSTM", "Naïve"]

model_metrics = {
    "Physics":   metrics_phys,
    "Residual":  metrics_neur,
    "Phys+Res":  metrics_res,
    "LSTM":      metrics_lstm,
    "Naïve":  metrics_base,
}

model_styles = {
    "Physics": ('-', 'tab:blue'),
    "Residual":  ('-', 'tab:orange'),
    "Phys+Res":('-', 'tab:purple'),
    "LSTM":    ('-', 'tab:green'),
    "Naïve":('--', 'tab:red'),
}

from matplotlib.ticker import FormatStrFormatter

ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

for i, metric in enumerate(metric_names):
    ax = axs[i]

    # make tick numbers bigger
    ax.tick_params(labelsize=13, width=1.2, length=4)

    # <-- Add this line for .1f formatting
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Light grey axis box (same as torque-y)
    for spine in ax.spines.values():
        spine.set_edgecolor("lightgray")


    # Plot models in desired *drawing* order
    for model_name in plot_order:
        mm = model_metrics[model_name]
        horizons = np.array(list(mm[metric].keys()))
        values   = np.array(list(mm[metric].values()))

        ls, color = model_styles[model_name]
        ax.plot(horizons, values, ls, color=color,
                linewidth=2, markersize=4, label=model_name)

    ax.set_ylabel(ylabels[i], fontsize=12)
    min_val = min(metrics_base[metric].values())
    max_val = max(metrics_base[metric].values())
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel("$h$  [-]", fontsize=14)
    ax.grid(True, alpha=0.3)

# === Shared Legend ===
handles, labels = axs[0].get_legend_handles_labels()

# Reorder legend so Baseline is FIRST
legend_order = ["Naïve", "Physics", "Residual", "Phys+Res", "LSTM"]
legend_handles = [handles[labels.index(m)] for m in legend_order]

fig.legend(
    legend_handles, legend_order,
    loc="upper center",
    ncols=5,
    bbox_to_anchor=(0.5, 1.15),
    fontsize=14
)

plt.subplots_adjust(top=0.86, bottom=0.12, hspace=0.25, wspace=0.4)
plt.savefig("new_metrics_models.pdf", bbox_inches='tight')
plt.show()

#%%
from matplotlib.ticker import FormatStrFormatter

N_start = 2000
N_end = N_start + 500
plot_multistate_predictions(h=50, N_start=N_start, N_end=N_end)
#%%
import numpy as np
import pandas as pd

# ============================================================
# === CONFIG
# ============================================================
H_TARGETS = [1, 10, 50]

model_order = ["Naïve", "Physics", "Residual", "Phys+Res", "LSTM"]

model_metrics = {
    "Naïve": metrics_base,
    "Physics":  metrics_phys,
    "Residual": metrics_neur,
    "Phys+Res": metrics_res,
    "LSTM":     metrics_lstm,
}


# ============================================================
# === Build rows
# ============================================================
rows = []
for model_name in model_order:
    mm = model_metrics[model_name]

    pos_vals = [mm["pos"][h] for h in H_TARGETS]
    vel_vals = [mm["vel"][h] for h in H_TARGETS]
    rot_vals = [mm["rot"][h] for h in H_TARGETS]
    omg_vals = [mm["omega"][h] for h in H_TARGETS]

    sim_p, sim_v, sim_R, sim_w = compute_simerr(mm)

    rows.append([
        model_name,
        *pos_vals, sim_p,
        *vel_vals, sim_v,
        *rot_vals, sim_R,
        *omg_vals, sim_w,
    ])


print_latex_table_results()


# ============================================================
# === CONFIG
# ============================================================
H_TARGETS = [1, 10, 50]

model_order = ["Naïve", "Physics", "Residual", "Phys+Res", "LSTM"]

model_metrics = {
    "Naïve": metrics_base,
    "Physics":  metrics_phys,
    "Residual":   metrics_neur,
    "Phys+Res": metrics_res,
    "LSTM":     metrics_lstm,
}



# Standard usage
plot_multistate_boxplots(df_base, df_lstm, df_neur, df_res, h=50, N_start=0, N_end=None, max_outliers=1)
