import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import os, sys


from simulator.utils.quat import *

def quad_dynamics(x, u, params):
    """
    Quaternion-based quadrotor dynamics (continuous time).
    State x: [pos(3), vel(3), quat(4), omega(3)]
    Normalized control u: [T, τ_φ, τ_θ, τ_ψ] in [0,1] x [-1,1]^3
    """
    # Extract parameters
    m = params["m"]                     # [kg]
    J = params["J"]                     # [kg·m²]
    g = params["g"]                     # [m/s²]
    thrust_to_weight = params["thrust_to_weight"]
    max_torque = params["max_torque"]   # [Nm]
    
    wind_force = params.get("wind_force", np.zeros(3))   # [N]

    # Unpack state
    pos = x[0:3]
    vel = x[3:6]
    quat = x[6:10]
    omega = x[10:13]

    # Saturate control
    T_norm = jnp.clip(u[0], 0.0, 1.0)
    tau_norm = jnp.clip(u[1:], -1.0, 1.0)

    # Convert to actual thrust and torques
    T_max = thrust_to_weight * m * g
    T = T_norm * T_max                 # [N]
    tau = tau_norm * max_torque        # [Nm]

    # Translational dynamics
    thrust_world = quat_rotate(quat, jnp.array([0.0, 0.0, T]))
    acc = (thrust_world + wind_force - jnp.array([0.0, 0.0, m * g])) / m

    # omega_cross = np.cross(omega, J @ omega)
    # print("omega_cross_z:", omega_cross[2])
    # print("tau_z:", tau[2])
    # print("tau_minus_cross_z:", tau[2] - omega_cross[2])
    # print("omega_dot_z calculated:", (tau[2] - omega_cross[2]) / J[2, 2])
    # Rotational dynamics
    omega_dot = jnp.linalg.solve(J, tau - jnp.cross(omega, J @ omega))

    # Quaternion kinematics
    quat_dot = quat_derivative(quat, omega)

    # Assemble derivative
    dx = jnp.concatenate([vel, acc, quat_dot, omega_dot])
    return dx

def quad_data_to_dataframe(X, U, dt):
    """
    Convert trajectory data to a pandas DataFrame for easy plotting.
    
    Parameters:
    - X: ndarray (T+1, 13), state trajectory [x, y, z, vx, vy, vz, qx, qy, qz, qw, wx, wy, wz]
    - U: ndarray (T, 4), control trajectory [T, τ_φ, τ_θ, τ_ψ]
    - dt: float, time step
    
    Returns:
    - DataFrame with time, state and control variables
    """
    T = X.shape[0]
    time = np.linspace(0, T * dt, T)

    state_columns = ['x', 'y', 'z', 'vx', 'vy', 'vz', 
                     'qx', 'qy', 'qz', 'qw', 'wx', 'wy', 'wz']
    control_columns = ['T', 'tau_phi', 'tau_theta', 'tau_psi']

    df_state = pd.DataFrame(X, columns=state_columns)
    df_state.insert(0, 't', time)

    # Extract all quaternions from X
    quats = X[:, 6:10]  # shape (T, 4)
    euler_angles = jax.vmap(quat_to_euler)(quats)  # shape (T, 3)
    yaw, pitch, roll = euler_angles.T
    df_state['yaw'] = np.asarray(yaw)
    df_state['pitch'] = np.asarray(pitch)
    df_state['roll'] = np.asarray(roll)

    df = df_state
    
    if U is not None:
        df_control = pd.DataFrame(U, columns=control_columns)
        df_control.insert(0, 't', time[:-1]) # controls are one step shorter

        # Merge with suffixes so both state and control values are accessible
        df = pd.merge(df, df_control, on='t', how='left')

    return df
