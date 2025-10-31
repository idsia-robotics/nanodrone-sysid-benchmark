# idsia_mpc/utils/traj_utils.py

"""
Trajectory utility functions.
This module provides a unified interface for evaluating different trajectory types
(e.g. precomputed splines vs polynomial CSVs).
"""

from idsia_mpc.new.quadrorotor_traj_plan import trajectory_point_eval
from idsia_mpc.quadrotor_traj import trajectory_eval


def get_setpoint(traj, t, traj_type):
    """
    Return a Setpoint at time t for the given trajectory type.

    Args:
        traj : trajectory object (from generator or CSV)
        t (float) : time in seconds
        traj_type (str) : "multisine", "melon", "random", or "points"

    Returns:
        Setpoint
    """
    if traj_type in ["multisine", "melon", "random", "square"]:
        return trajectory_point_eval(traj, t)
    elif traj_type == "points":
        return trajectory_eval(traj, t)
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")


def get_ref_arrays(traj, t_vec, traj_type, quat_to_euler_fn):
    """
    Evaluate reference arrays (position, velocity, omega, Euler angles).

    Args:
        traj : trajectory object
        t_vec (array) : array of times
        traj_type (str) : "multisine", "melon", "random", or "points"
        quat_to_euler_fn : function to convert quaternion -> Euler

    Returns:
        X_ref : np.ndarray (N,3) position
        V_ref : np.ndarray (N,3) velocity
        W_ref : np.ndarray (N,3) angular rates
        Euler_ref : np.ndarray (N,3) [roll, pitch, yaw]
    """
    X_ref, V_ref, W_ref, Euler_ref = [], [], [], []

    for t in t_vec:
        sp = get_setpoint(traj, t, traj_type)
        X_ref.append(sp.pos)
        V_ref.append(sp.vel)
        W_ref.append(sp.omega)
        Euler_ref.append(quat_to_euler_fn(sp.orientation))

    import numpy as np
    return np.array(X_ref), np.array(V_ref), np.array(W_ref), np.array(Euler_ref)
