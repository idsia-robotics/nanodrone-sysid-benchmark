import test_torch
from idsia_mpc.control_engineering_practice.quat_utils import quat_to_euler_torch

def state_quat_to_euler(x: torch.Tensor) -> torch.Tensor:
    """
    Replace quaternion (indices 6:10) with Euler angles in the state vector.
    """
    pos_vel = x[:, :6]
    quat = x[:, 6:10]
    euler = quat_to_euler_torch(quat)
    omega = x[:, 10:]
    return torch.cat([pos_vel, euler, omega], dim=1)
