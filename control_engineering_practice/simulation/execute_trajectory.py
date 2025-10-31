"""
Main script to simulate quadrotor trajectories with Mellinger controller.
Select trajectory type (multisine, melon, random, points),
simulate, log, and generate plots.
"""

import numpy as np
import jax.numpy as jnp
import pandas as pd
from datetime import datetime
import json
import matplotlib.pyplot as plt

import os, sys

from idsia_mpc.control_engineering_practice.simulation.export_uav_trajectory import export_traj_for_uav_trajectories

# find the folder that contains "idsia_mpc"
current_dir = os.path.dirname(os.path.abspath(__file__))
while True:
    if os.path.exists(os.path.join(current_dir, "idsia_mpc")):
        PROJECT_ROOT = current_dir
        break
    parent = os.path.dirname(current_dir)
    if parent == current_dir:  # reached filesystem root
        raise RuntimeError("Could not find project root containing 'idsia_mpc'")
    current_dir = parent

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("Using project root:", PROJECT_ROOT)

from idsia_mpc.quadrotor_sys import quad_dynamics
from idsia_mpc.quadrotor_ctrl import quad_controller_mellinger
from idsia_mpc.quadrotor_traj import trajectory_from_csv
from idsia_mpc.new.quadrorotor_traj_plan import (
    trajectory_plan_multi_axis_excitation,
    trajectory_plan_melon,
    trajectory_plan_random_points,
    trajectory_plan_excitation_square,
    trajectory_plan_excitation_square_facing_motion
)

from idsia_mpc.control_engineering_practice.traj_utils import get_setpoint, get_ref_arrays
from idsia_mpc.control_engineering_practice.quat_utils import quat_to_euler
from idsia_mpc.control_engineering_practice.plot_utils import (
    plot_positions, plot_velocities, plot_angular_rates,
    plot_position_errors, plot_euler_angles, plot_3d_traj, animate_trajectory
)
from idsia_mpc.control_engineering_practice.log_utils import log_to_dataframe

# -------------------------------------------------------------
# Settings
# -------------------------------------------------------------
save_animation = False
save_trajectory = False
save_dataframe = True
traj_type = "square"   # "multisine", "melon", "random", "points", "square"

sys_params = {
    "g": 9.81,
    "m": 0.032,
    "J": np.diag([1.43e-6, 1.43e-6, 2.89e-6]),
    "thrust_to_weight": 2.0,
    "max_torque": jnp.array([1e-4, 1e-4, 3e-5]),
}

ctrl_params_mellinger = {
    "Kp_pos": jnp.array([1.5, 1.5, 3.0]),
    "Kd_vel": jnp.array([2.0, 2.0, 2.5]),
    "Kp_att": jnp.array([8.0, 8.0, 3.0]),
    "Kp_rate": jnp.array([4.8e-5, 4.8e-5, 1.4e-5]),
}

traj_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# -------------------------------------------------------------
# Select trajectory
# -------------------------------------------------------------
if traj_type == "multisine":
    config = dict(center=[0,0,1.25], amplitudes=(.5,.5,.5),
                  freqs_start=(.1,.1,.1), freqs_end=(.5,.5,.5),
                  duration=60.0, sampling_frequency=100.0)
    traj = trajectory_plan_multi_axis_excitation(**config)

elif traj_type == "melon":
    config = dict(center=[0,0,1.25], start_radii=(.75,.75), end_radii=(.75,.75),
                  omega_circle_start=2.5, omega_circle_end=3.5,
                  omega_plane=0.4, plane_axis="x",
                  duration=60.0, sampling_frequency=100.0)
    traj = trajectory_plan_melon(**config)

elif traj_type == "random":
    config = dict(center=[0,0,1.5], box_size=(1.,1.,0.5), n_points=52,
                  duration=60.0, sampling_frequency=100.0, seed=1)
    traj = trajectory_plan_random_points(**config)

elif traj_type == "points":
    # traj = trajectory_from_csv("../data/real/poly_spline_trajectories/random_1848.csv")
    traj = trajectory_from_csv("../data/real/poly_spline_trajectories/figure8.csv")

elif traj_type == "square":
    traj = trajectory_plan_excitation_square(center=(0, 0, 0.5), side=1.0, z_step=1.0)
    # traj = trajectory_plan_excitation_square_facing_motion(center=(0, 0, 0.5), side=1.0, z_step=1.0)
else:
    raise ValueError(f"Unknown trajectory type {traj_type}")

# duration
T_final = traj.duration if traj_type == "points" else float(traj.t[-1])

# -------------------------------------------------------------
# Simulation loop
# -------------------------------------------------------------
dt = 0.01
steps = int(T_final / dt)
t_vec = np.arange(steps) * dt

sp0 = get_setpoint(traj, 0.0, traj_type)
x_m = np.array(sp0.to_state())
X_m, U_m, ctrl_state_m = [x_m.copy()], [], None

for t in t_vec:
    sp = get_setpoint(traj, float(t), traj_type)
    u, ctrl_state_m = quad_controller_mellinger(x_m, sp, sys_params, ctrl_params_mellinger, ctrl_state_m)
    dx = quad_dynamics(x_m, np.array(u), sys_params)
    x_m = x_m + dt * np.array(dx)
    X_m.append(x_m.copy()); U_m.append(np.array(u))

X_m, U_m = np.array(X_m), np.array(U_m)

# -------------------------------------------------------------
# References and errors
# -------------------------------------------------------------
X_ref, V_ref, W_ref, Euler_ref = get_ref_arrays(traj, t_vec, traj_type, quat_to_euler)
errors_m = X_m[:len(X_ref),:3] - X_ref
rmse_m = np.sqrt(np.mean(errors_m**2, axis=0))
print("RMSE Mellinger [x,y,z]:", rmse_m)

# -------------------------------------------------------------
# Logging
# -------------------------------------------------------------
if save_dataframe:
    df = log_to_dataframe(X_m, U_m, t_vec, traj, traj_type, quat_to_euler)
    df.to_parquet(f"../data/sim/new/experiment_{traj_type}_{traj_id}.parquet", index=False)

# -------------------------------------------------------------
# Plots
# -------------------------------------------------------------
fig1 = plot_positions(t_vec, X_ref, X_m)
fig2 = plot_velocities(t_vec, V_ref, X_m)
fig3 = plot_angular_rates(t_vec, W_ref, X_m)
fig4 = plot_position_errors(t_vec, errors_m)
fig5 = plot_euler_angles(t_vec, Euler_ref, X_m)

# Show all plots
fig1.show(), fig2.show(), fig3.show(), fig4.show(), fig5.show()

plot_3d_traj(X_ref, traj_type)

if save_animation:
    animate_trajectory(X_m, X_ref, t_vec, traj_type, traj_id, dt, T_final)


if save_trajectory:
    print(len(traj.t))
    export_traj_for_uav_trajectories(traj, filename=f"traj_{traj_type}_{traj_id}.csv")

    # -------------------------------------------------------------
    # Compare exported trajectory vs original
    # -------------------------------------------------------------
    import pandas as pd

    # Load exported trajectory (CSV with header)
    df = pd.read_csv(f"traj_{traj_type}_{traj_id}.csv")
    X_export = df[["x","y","z"]].values

    # Original reference positions
    X_ref = np.array(traj.pos)

    # 3D plot
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection="3d")

    # plot original
    ax.plot(X_ref[:,0], X_ref[:,1], X_ref[:,2], "k--", label="original traj")

    # plot exported
    ax.plot(X_export[:,0], X_export[:,1], X_export[:,2], "r-o", alpha=0.7, label="exported txt")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Exported vs Original Trajectory", fontsize=12)
    ax.legend()

    ax.set_box_aspect([1,1,1])   # equal aspect
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.show()
