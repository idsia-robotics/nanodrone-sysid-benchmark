# idsia_mpc/utils/log_utils.py
"""
Logging utilities: convert sim and reference into a DataFrame.
"""

import pandas as pd
import numpy as np
from idsia_mpc.control_engineering_practice.traj_utils import get_setpoint

def _to_float(x):
    """Convert JAX/NumPy arrays to plain Python float(s)."""
    if hasattr(x, "item"):   # scalar JAX/NumPy array
        return float(x.item())
    if isinstance(x, (list, tuple, np.ndarray)):
        return [float(v) for v in x]
    return float(x)


def log_to_dataframe(X_m, U_m, t_vec, traj, traj_type, quat_to_euler_fn):
    """
    Build DataFrame with simulated states, reference, and control inputs.
    Ensures all values are Python floats for parquet export.
    """
    rows = []
    for k, t in enumerate(t_vec):
        # --- simulated state ---
        x, y, z = map(float, X_m[k, 0:3])
        vx, vy, vz = map(float, X_m[k, 3:6])
        qx, qy, qz, qw = map(float, X_m[k, 6:10])
        wx, wy, wz = map(float, X_m[k, 10:13])

        roll, pitch, yaw = map(float, quat_to_euler_fn([qx, qy, qz, qw]))

        # --- reference ---
        sp = get_setpoint(traj, float(t), traj_type)
        xr, yr, zr = map(float, sp.pos)
        vxr, vyr, vzr = map(float, sp.vel)
        qxr, qyr, qzr, qwr = map(float, sp.orientation)
        wxr, wyr, wzr = map(float, sp.omega)
        roll_r, pitch_r, yaw_r = map(float, quat_to_euler_fn([qxr, qyr, qzr, qwr]))

        if k < len(U_m):
            thrust, torque_roll, torque_pitch, torque_yaw = map(float, U_m[k])
        else:
            thrust, torque_roll, torque_pitch, torque_yaw = (np.nan,)*4

        rows.append([
            float(t),
            x, y, z,
            qx, qy, qz, qw,
            yaw, pitch, roll,
            vx, vy, vz,
            wx, wy, wz,
            xr, yr, zr,
            qxr, qyr, qzr, qwr,
            yaw_r, pitch_r, roll_r,
            vxr, vyr, vzr,
            wxr, wyr, wzr,
            thrust, torque_roll, torque_pitch, torque_yaw
        ])

    cols = [
        "t", "x", "y", "z", "qx", "qy", "qz", "qw",
        "yaw", "pitch", "roll",
        "vx", "vy", "vz", "wx", "wy", "wz",
        "x_r", "y_r", "z_r", "qx_r", "qy_r", "qz_r", "qw_r",
        "yaw_r", "pitch_r", "roll_r",
        "vx_r", "vy_r", "vz_r", "wx_r", "wy_r", "wz_r",
        "thrust", "torque_roll", "torque_pitch", "torque_yaw"
    ]
    return pd.DataFrame(rows, columns=cols)