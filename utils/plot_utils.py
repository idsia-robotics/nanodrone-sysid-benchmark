import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FormatStrFormatter

def setup_matplotlib():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "cm",   # Computer Modern math (LaTeX look)
        "font.size": 11,            # matches cas-dc main text
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 14,
        "axes.grid": True,
        "grid.linestyle": "--",
        "axes.xmargin": 0,
        "axes.ymargin": 0,
    })


def plot_reference_trajectory(X_ref, title="Melon"):
    """
    Create a 3D static plot of the reference trajectory (for publication figures).

    Args:
        X_ref: (N, 3) array of reference positions [x, y, z]
        title: optional title (default: 'Melon')
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory
    ax.plot(X_ref[:, 0], X_ref[:, 1], X_ref[:, 2],
            color='tab:blue', linewidth=1.5)

    # Labels and title
    ax.set_xlabel(r"$X$ [m]")
    ax.set_ylabel(r"$Y$ [m]")
    ax.set_zlabel(r"$Z$ [m]")
    ax.set_title(title)

    # Equal aspect ratio
    max_range = np.ptp(X_ref, axis=0).max() / 2.0
    mid = np.mean(X_ref, axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    # View angle (adjust to match your paper)
    ax.view_init(elev=25, azim=45)

    # Grid & layout
    ax.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()

def animate_trajectory(X_m, X_ref, t_vec, traj_type, traj_id, dt, T_final,
                       gif_fps=25, axis_len=0.3, save=True):
    """
    Create a 3D animation of the quadrotor trajectory.

    Args:
        X_m: (N, state_dim) simulated states
        X_ref: (N,3) reference positions
        t_vec: time vector
        traj_type: string, trajectory type
        traj_id: string, unique id
        dt: timestep [s]
        T_final: total duration [s]
        gif_fps: frames per second for gif export
        axis_len: scale of body axes in 3D plot
        save: if True, saves GIF file
    """
    from matplotlib.animation import FuncAnimation
    import matplotlib.pyplot as plt
    import numpy as np

    stride = int(1 / (gif_fps * dt))  # how many sim steps per GIF frame

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    # Draw full reference once
    ax.plot(X_ref[:,0], X_ref[:,1], X_ref[:,2], "k--", label="reference")

    mel_line, = ax.plot([], [], [], "b-", label="Mellinger")
    ax.legend()

    frames = range(0, len(t_vec), stride)

    # add three lines for the body axes
    x_axis_line, = ax.plot([], [], [], "r-", lw=2)  # x-body (red)
    y_axis_line, = ax.plot([], [], [], "g-", lw=2)  # y-body (green)
    z_axis_line, = ax.plot([], [], [], "b-", lw=2)  # z-body (blue)

    def update(frame_idx):
        mel_line.set_data(X_m[:frame_idx,0], X_m[:frame_idx,1])
        mel_line.set_3d_properties(X_m[:frame_idx,2])

        # orientation from state quaternion
        q = X_m[frame_idx, 6:10]  # [x, y, z, w]
        x, y, z, w = q
        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z),     2*(y*z - x*w)],
            [2*(x*z - y*w),         2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])

        p = X_m[frame_idx, 0:3]

        # body axes in world frame
        xb = p + axis_len * R[:,0]
        yb = p + axis_len * R[:,1]
        zb = p + axis_len * R[:,2]

        x_axis_line.set_data([p[0], xb[0]], [p[1], xb[1]])
        x_axis_line.set_3d_properties([p[2], xb[2]])

        y_axis_line.set_data([p[0], yb[0]], [p[1], yb[1]])
        y_axis_line.set_3d_properties([p[2], yb[2]])

        z_axis_line.set_data([p[0], zb[0]], [p[1], zb[1]])
        z_axis_line.set_3d_properties([p[2], zb[2]])

        return mel_line, x_axis_line, y_axis_line, z_axis_line

    ani = FuncAnimation(fig, update, frames=frames, blit=False)
    if save:
        ani.save(f"{traj_type}_{traj_id}.gif", writer="pillow", fps=gif_fps, dpi=100)

    print(f"Saved animation: {traj_type}_{traj_id}.gif ({T_final:.2f} s duration, {gif_fps} fps)")

def animate_sim_vs_real(df_sim, df_real, gif_filename=None, gif_fps=25, axis_len=0.2):
    """
    Animate simulation vs real trajectories in 3D, with both body frames.

    Args:
        df_sim (pd.DataFrame): simulated trajectory dataframe
        df_real (pd.DataFrame): real trajectory dataframe
        gif_filename (str): if provided, save animation as GIF
        gif_fps (int): frames per second for GIF
        axis_len (float): length of body axes arrows
    """

    # Common time base
    t_vec = df_sim["t"].values
    stride = max(1, int(1 / (gif_fps * (t_vec[1] - t_vec[0]))))
    frames = range(0, len(t_vec), stride)

    # Extract positions
    X_sim = df_sim[["x", "y", "z"]].values
    X_real = df_real[["x", "y", "z"]].values
    X_ref = df_sim[["x_r", "y_r", "z_r"]].values

    # Setup figure
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    # Set limits around reference trajectory
    mins = X_ref.min(axis=0) - 0.2
    maxs = X_ref.max(axis=0) + 0.2
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

    # Reference path (dashed black)
    ax.plot(X_ref[:, 0], X_ref[:, 1], X_ref[:, 2], "k--", label="Reference")

    # Init lines
    sim_line, = ax.plot([], [], [], "b-", label="Simulation")
    real_line, = ax.plot([], [], [], "r-", label="Real")
    ax.legend()

    # Body axes for real trajectory (RGB)
    real_x_axis, = ax.plot([], [], [], "r-", lw=2)
    real_y_axis, = ax.plot([], [], [], "g-", lw=2)
    real_z_axis, = ax.plot([], [], [], "b-", lw=2)

    # Body axes for sim trajectory (CMY for contrast)
    sim_x_axis, = ax.plot([], [], [], color="c", lw=2)  # cyan
    sim_y_axis, = ax.plot([], [], [], color="m", lw=2)  # magenta
    sim_z_axis, = ax.plot([], [], [], color="y", lw=2)  # yellow

    # ---- Add time overlay (top-left corner) ----
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes,
                          fontsize=12, fontweight="bold", color="black")

    def quat_to_rotmat(qx, qy, qz, qw):
        """Quaternion -> rotation matrix"""
        return np.array([
            [1 - 2 * (qy**2 + qz**2),     2 * (qx*qy - qz*qw),     2 * (qx*qz + qy*qw)],
            [2 * (qx*qy + qz*qw), 1 - 2 * (qx**2 + qz**2),     2 * (qy*qz - qx*qw)],
            [2 * (qx*qz - qy*qw),     2 * (qy*qz + qx*qw), 1 - 2 * (qx**2 + qy**2)]
        ])

    def update(frame_idx):
        # --- update paths ---
        sim_line.set_data(X_sim[:frame_idx, 0], X_sim[:frame_idx, 1])
        sim_line.set_3d_properties(X_sim[:frame_idx, 2])

        real_line.set_data(X_real[:frame_idx, 0], X_real[:frame_idx, 1])
        real_line.set_3d_properties(X_real[:frame_idx, 2])

        # --- orientation real ---
        qx, qy, qz, qw = df_real.loc[frame_idx, ["qx", "qy", "qz", "qw"]]
        R_real = quat_to_rotmat(qx, qy, qz, qw)
        p_real = X_real[frame_idx, :]

        rx, ry, rz = p_real + axis_len * R_real[:, 0], p_real + axis_len * R_real[:, 1], p_real + axis_len * R_real[:, 2]
        real_x_axis.set_data([p_real[0], rx[0]], [p_real[1], rx[1]])
        real_x_axis.set_3d_properties([p_real[2], rx[2]])
        real_y_axis.set_data([p_real[0], ry[0]], [p_real[1], ry[1]])
        real_y_axis.set_3d_properties([p_real[2], ry[2]])
        real_z_axis.set_data([p_real[0], rz[0]], [p_real[1], rz[1]])
        real_z_axis.set_3d_properties([p_real[2], rz[2]])

        # --- orientation sim ---
        qx, qy, qz, qw = df_sim.loc[frame_idx, ["qx", "qy", "qz", "qw"]]
        R_sim = quat_to_rotmat(qx, qy, qz, qw)
        p_sim = X_sim[frame_idx, :]

        sx, sy, sz = p_sim + axis_len * R_sim[:, 0], p_sim + axis_len * R_sim[:, 1], p_sim + axis_len * R_sim[:, 2]
        sim_x_axis.set_data([p_sim[0], sx[0]], [p_sim[1], sx[1]])
        sim_x_axis.set_3d_properties([p_sim[2], sx[2]])
        sim_y_axis.set_data([p_sim[0], sy[0]], [p_sim[1], sy[1]])
        sim_y_axis.set_3d_properties([p_sim[2], sy[2]])
        sim_z_axis.set_data([p_sim[0], sz[0]], [p_sim[1], sz[1]])
        sim_z_axis.set_3d_properties([p_sim[2], sz[2]])

        # --- update time text ---
        time_text.set_text(f"t = {t_vec[frame_idx]:.2f} s")

        return (sim_line, real_line,
                real_x_axis, real_y_axis, real_z_axis,
                sim_x_axis, sim_y_axis, sim_z_axis)

    ani = FuncAnimation(fig, update, frames=frames, blit=False)

    if gif_filename:
        ani.save(gif_filename, writer="pillow", fps=gif_fps, dpi=100)
        print(f"✅ Saved animation: {gif_filename}")

    return ani

# idsia_mpc/utils/plot_utils.py
"""
Plotting utilities for quadrotor trajectory simulation.
Includes time-series plots, 3D trajectory visualization,
and 3D animation export as GIF.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_positions(t_vec, X_ref, X_m, X_m_bis=None, N=None):
    """Plot position tracking x,y,z."""
    if N is None:
        N = len(t_vec)
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    labels = ["x [m]", "y [m]", "z [m]"]
    for i, ax in enumerate(axs):
        ax.plot(t_vec[:N], X_ref[:N, i], "k--", label="ref")
        ax.plot(t_vec[:N], X_m[:N, i], "b", label="Mellinger")
        if X_m_bis is not None:
            ax.plot(t_vec[:N], X_m_bis[:N, i], "r", label="Repetition")
        ax.set_ylabel(labels[i]); ax.grid(True)
    axs[-1].set_xlabel("Time [s]"); axs[0].legend()
    fig.suptitle("Position Tracking")
    return fig


def plot_velocities(t_vec, V_ref, X_m, X_m_bis=None, N=None):
    """Plot velocity tracking vx,vy,vz."""
    if N is None:
        N = len(t_vec)
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    labels = ["vx [m/s]", "vy [m/s]", "vz [m/s]"]
    for i, ax in enumerate(axs):
        ax.plot(t_vec[:N], V_ref[:N, i], "k--", label="ref")
        ax.plot(t_vec[:N], X_m[:N, 3+i], "b", label="Mellinger")
        if X_m_bis is not None:
            ax.plot(t_vec[:N], X_m_bis[:N, 3+i], "r", label="Repetition")
        ax.set_ylabel(labels[i]); ax.grid(True)
    axs[-1].set_xlabel("Time [s]"); axs[0].legend()
    fig.suptitle("Velocity Tracking")
    return fig


def plot_angular_rates(t_vec, W_ref, X_m, X_m_bis=None, N=None):
    """Plot angular rates wx,wy,wz."""
    if N is None:
        N = len(t_vec)
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    labels = ["wx [rad/s]", "wy [rad/s]", "wz [rad/s]"]
    for i, ax in enumerate(axs):
        ax.plot(t_vec[:N], W_ref[:N, i], "k--", label="ref")
        ax.plot(t_vec[:N], X_m[:N, 10+i], "b", label="Mellinger")
        if X_m_bis is not None:
            ax.plot(t_vec[:N], X_m_bis[:N, 10+i], "r", label="Repetition")
        ax.set_ylabel(labels[i]); ax.grid(True)
    axs[-1].set_xlabel("Time [s]"); axs[0].legend()
    fig.suptitle("Angular Rate Tracking")
    return fig


def plot_position_errors(t_vec, errors_m):
    """Plot position tracking errors ex,ey,ez."""
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    labels = ["ex [m]", "ey [m]", "ez [m]"]
    for i, ax in enumerate(axs):
        ax.plot(t_vec, errors_m[:, i], "b", label="Mellinger")
        ax.set_ylabel(labels[i]); ax.grid(True)
    axs[-1].set_xlabel("Time [s]"); axs[0].legend()
    axs[-1].set_xlim([0, 1])
    fig.suptitle("Position Errors")
    return fig


def plot_euler_angles(t_vec, Euler_ref, X_m, X_m_bis=None, N=None):
    """Plot Euler angles roll, pitch, yaw."""
    if N is None:
        N = len(t_vec)
    Euler_m = np.array([X_m[i, 6:10] for i in range(len(t_vec))])
    if X_m_bis is not None:# quat
        Euler_bis = np.array([X_m_bis[i, 6:10] for i in range(len(t_vec))])  # quat
    # You’ll call quat_to_euler outside and pass arrays if preferred
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    labels = ["roll [rad]", "pitch [rad]", "yaw [rad]"]
    for i, ax in enumerate(axs):
        ax.plot(t_vec[:N], Euler_ref[:N, i], "k--", label="ref")
        ax.plot(t_vec[:N], Euler_m[:N, i], "b", label="Mellinger")
        if X_m_bis is not None:
            ax.plot(t_vec[:N], Euler_bis[:N, i], "r", label="Repetition")
        ax.set_ylabel(labels[i]); ax.grid(True)
    axs[-1].set_xlabel("Time [s]"); axs[0].legend()
    fig.suptitle("Euler Angle Tracking")
    return fig


def plot_3d_traj(X_ref, traj_type):
    """Simple 3D plot of reference trajectory."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(X_ref[:, 0], X_ref[:, 1], X_ref[:, 2], color="C0")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ax.set_title(traj_type, fontsize=14, fontweight="bold",
                 backgroundcolor="orange", pad=10)
    ax.set_box_aspect([1,1,1])
    ax.view_init(elev=30, azim=45)
    return fig


def plot_multistate_predictions(dfs, h=50, N_start=0, N_end=None):
    """
    Create a 4x3 grid of time-series plots comparing true vs. predicted trajectories
    (h-step ahead) for each state variable.
    """
    if N_end is None:
        N_end = len(dfs["Naive"]) - h

    # --- State order (12 total) ---
    state_names = [
        "x", "y", "z",
        "roll", "pitch", "yaw",
        "vx", "vy", "vz",
        "wx", "wy", "wz"
    ]

    # --- Fancy axis labels ---
    state_labels = [
        [r"$x$ [m]", r"$y$ [m]", r"$z$ [m]"],
        [r'$\varphi$ [rad]', r'$\theta$ [rad]', r'$\psi$ [rad]'],
        [r"$v_x$ [m/s]", r"$v_y$ [m/s]", r"$v_z$ [m/s]"],
        [r"$\omega_x$ [rad/s]", r"$\omega_y$ [rad/s]", r"$\omega_z$ [rad/s]"]
    ]

    # --- Figure setup ---
    fig, axs = plt.subplots(4, 3, figsize=(12, 5), sharex=True, dpi=200)
    t = dfs["Naive"]["t"].values

    for r in range(4):
        for c in range(3):
            idx = r * 3 + c
            if idx >= len(state_names):
                continue

            state = state_names[idx]
            pred_col = f"{state}_pred_h{h}"
            ax = axs[r, c]

            # --- Smoothed signals ---
            neur = dfs["Residual"][pred_col][N_start:N_end].rolling(20, min_periods=1, center=True).mean()
            res  = dfs["Phys+Res"][pred_col][N_start:N_end].rolling(20, min_periods=1, center=True).mean()
            lstm = dfs["LSTM"][pred_col][N_start:N_end].rolling(20, min_periods=1, center=True).mean()
            base = dfs["Naive"][pred_col][N_start:N_end].rolling(20, min_periods=1, center=True).mean()
            phys = dfs["Physics"][pred_col][N_start:N_end].rolling(20, min_periods=1, center=True).mean()
            true = dfs["Naive"][state][N_start + h:N_end + h].rolling(20, min_periods=1, center=True).mean()

            # --- Plot ---
            ax.plot(t[N_start + h:N_end + h], base, '-', color='tab:red', label='Naïve', linewidth=2, alpha=0.2)
            ax.plot(t[N_start + h:N_end + h], phys, '-', color='tab:blue', label='Physics', linewidth=1.2)
            ax.plot(t[N_start + h:N_end + h], neur, '-', color='tab:orange', label='Residual', linewidth=1.2)
            ax.plot(t[N_start + h:N_end + h], res, '-', color='tab:purple', label='Phys+Res', linewidth=1.2)
            ax.plot(t[N_start + h:N_end + h], lstm, '-', color='tab:green', label='LSTM', linewidth=1.2)
            ax.plot(t[N_start + h:N_end + h], true, 'k--', label='GT', linewidth=1.5)

            # --- Aesthetics ---
            ax.set_ylabel(state_labels[r][c], fontsize=14, labelpad=10)
            ax.grid(True, alpha=0.3)

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            # Light grey axis box (same as torque-y)
            for spine in ax.spines.values():
                spine.set_edgecolor("lightgray")

            # make tick numbers bigger
            ax.tick_params(labelsize=13, width=1.2, length=4)

            if (r == 3) and (c == 1):
                ax.set_xlabel("Time [s]", fontsize=14)

    # --- Align y-labels properly ---
    fig.align_ylabels(axs[:, 0])

    # --- Shared legend (like 2nd figure) ---
    handles, labels = [], []
    for ax in axs.flat:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    # fig.legend(handles, labels, loc='upper center', ncols=6,
    #            bbox_to_anchor=(0.5, 1), fontsize=14, frameon=True)

    labelx = -0.17  # axes coords

    for ax in axs.flat:
        ax.yaxis.set_label_coords(labelx, 0.5)

    plt.subplots_adjust(top=0.86, bottom=0.12, hspace=0.25, wspace=0.3)

    # Reorder legend so Baseline is FIRST
    legend_order = ["GT", "Naïve", "Physics", "Residual", "Phys+Res", "LSTM"]
    legend_handles = [handles[labels.index(m)] for m in legend_order]

    fig.legend(
    legend_handles, legend_order,
    loc="upper center",
    ncols=6,
    bbox_to_anchor=(0.5, .99),
    fontsize=14
    )

    plt.savefig('new_lineplots_models.pdf', bbox_inches='tight')
    plt.show()

def plot_multistate_boxplots(df_base, df_lstm, df_neur, df_res,
                             h=50, N_start=0, N_end=None, max_outliers=None):
    """
    Create a 4x3 grid of boxplots showing absolute errors for each model and state.
    Matches the aesthetic of plot_multistate_predictions.

    Args:
        df_base, df_lstm, df_neur, df_res : DataFrames
            Containing columns like '<state>_pred_h<h>'.
        h : int
            Prediction horizon.
        N_start, N_end : int
            Index range for samples.
        max_outliers : int or None
            If set, limits the number of largest error samples kept (useful to reduce extreme tails).
    """

    if N_end is None:
        N_end = len(df_base) - h

    # --- State order (12 total) ---
    state_names = [
        "x", "y", "z",
        "roll", "pitch", "yaw",
        "vx", "vy", "vz",
        "wx", "wy", "wz"
    ]

    # --- Fancy axis labels ---
    state_labels = [
        [r"$x$ [m]", r"$y$ [m]", r"$z$ [m]"],
        [r'$\varphi$ [rad]', r'$\theta$ [rad]', r'$\psi$ [rad]'],
        [r"$v_x$ [m/s]", r"$v_y$ [m/s]", r"$v_z$ [m/s]"],
        [r"$\omega_x$ [rad/s]", r"$\omega_y$ [rad/s]", r"$\omega_z$ [rad/s]"]
    ]

    # --- Model colors ---
    colors = {
        "Naïve": "tab:red",
        "Residual": "tab:purple",
        "Neural": "tab:orange",
        "LSTM": "tab:green",
    }

    fig, axs = plt.subplots(4, 3, figsize=(12, 6), dpi=100)
    axs = axs.flatten()

    for r in range(4):
        for c in range(3):
            idx = r * 3 + c
            if idx >= len(state_names):
                continue

            state = state_names[idx]
            pred_col = f"{state}_pred_h{h}"
            ax = axs[idx]

            # --- Compute absolute errors ---
            true = df_base[state].values[N_start + h:N_end + h]
            base = np.abs(df_base[pred_col].values[N_start:N_end] - true)
            neur = np.abs(df_neur[pred_col].values[N_start:N_end] - true)
            lstm = np.abs(df_lstm[pred_col].values[N_start:N_end] - true)
            res  = np.abs(df_res[pred_col].values[N_start:N_end] - true)

            # --- Optionally limit outliers ---
            if max_outliers is not None:
                def limit_outliers(arr, n=max_outliers):
                    if len(arr) > n:
                        cutoff = np.partition(arr, -n)[-n]  # nth largest
                        arr = np.clip(arr, None, cutoff)
                    return arr
                base, neur, lstm, res = map(limit_outliers, [base, neur, lstm, res])

            # --- Prepare data ---
            data = [base, res, neur, lstm]
            labels = list(colors.keys())

            # --- Boxplot ---
            box = ax.boxplot(data, patch_artist=True, labels=labels,
                             widths=0.55,
                             showfliers=False,
                             medianprops=dict(color='black', linewidth=1.2),
                             boxprops=dict(linewidth=1.1),
                             whiskerprops=dict(linewidth=1.0),
                             capprops=dict(linewidth=1.0),
                             flierprops=dict(marker='.', markersize=2, alpha=0.4))

            # --- Color boxes ---
            for patch, key in zip(box['boxes'], colors.keys()):
                patch.set_facecolor(colors[key])
                patch.set_alpha(0.6)

            # # --- Style ---
            # ax.set_ylabel(state_labels[r][c], fontsize=14, labelpad=10)
            # ax.grid(True, alpha=0.3)
            # ax.tick_params(labelsize=13, width=1.2, length=4)
            # ax.set_xticklabels(labels, rotation=20, fontsize=12)
            # --- Style ---
            ax.set_ylabel(state_labels[r][c], fontsize=14, labelpad=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=13, width=1.2, length=4)

            # REMOVE x labels
            ax.set_xticks([])
            ax.set_xlabel("")


    # --- Shared legend ---
    legend_elements = [
        plt.Line2D([0], [0], color=c, lw=6, label=label, alpha=0.6)
        for label, c in colors.items()
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncols=4,
               bbox_to_anchor=(0.5, .98), fontsize=14, frameon=True)
    fig.align_ylabels()
    plt.subplots_adjust(top=0.86, bottom=0.12, hspace=0.25, wspace=0.4)
    plt.savefig('boxplots_models.pdf')
    plt.show()


def plot_metrics(model_metrics, save_fig=False):
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

    model_styles = {
        "Physics": ('-', 'tab:blue'),
        "Residual": ('-', 'tab:orange'),
        "Phys+Res": ('-', 'tab:purple'),
        "LSTM": ('-', 'tab:green'),
        "Naïve": ('--', 'tab:red'),
    }

    for i, metric in enumerate(metric_names):
        ax = axs[i]
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

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
            values = np.array(list(mm[metric].values()))

            ls, color = model_styles[model_name]
            ax.plot(horizons, values, ls, color=color,
                    linewidth=2, markersize=4, label=model_name)

        ax.set_ylabel(ylabels[i], fontsize=12)
        min_val = min(model_metrics["Naïve"][metric].values())
        max_val = max(model_metrics["Naïve"][metric].values())
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
    if save_fig:
        plt.savefig("../figures/metrics_models.pdf", bbox_inches='tight')

    plt.show()

