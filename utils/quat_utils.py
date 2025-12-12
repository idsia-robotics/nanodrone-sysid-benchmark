# idsia_mpc/utils/quat_utils.py

"""
Quaternion utilities.
"""

import numpy as np
import torch

def quat_to_euler(q):
    """
    Convert quaternion(s) to Euler angles (roll, pitch, yaw).
    Ensures angles are continuous by applying np.unwrap.

    Args:
        q : array-like of shape (4,) or (N,4) [x,y,z,w]

    Returns:
        ndarray of shape (3,) or (N,3) [roll, pitch, yaw]
    """
    q = np.atleast_2d(q)
    x, y, z, w = q.T

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    angles = np.vstack([roll, pitch, yaw]).T

    if angles.shape[0] > 1:
        for i in range(3):
            angles[:, i] = np.unwrap(angles[:, i])

    return angles[0] if angles.shape[0] == 1 else angles

def quat_to_euler_torch(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion(s) to Euler angles (roll, pitch, yaw) in radians.
    Differentiable and GPU-compatible.

    Args:
        q : torch.Tensor of shape (..., 4) in (x, y, z, w) format.

    Returns:
        torch.Tensor of shape (..., 3): [roll, pitch, yaw]
    """
    # Ensure shape [..., 4]
    assert q.shape[-1] == 4, "Input quaternion must have last dimension 4 (x,y,z,w)"
    x, y, z, w = q.unbind(-1)

    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    sinp_clamped = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp_clamped)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)


def quat_xyzw_to_wxyz(q):
    # (x,y,z,w) → (w,x,y,z)
    return torch.cat([q[..., 3:], q[..., :3]], dim=-1)

def quat_wxyz_to_xyzw(q):
    # (w,x,y,z) → (x,y,z,w)
    return torch.cat([q[..., 1:], q[..., :1]], dim=-1)

def quat_to_so3_log(q_xyzw):
    """Torch: (x,y,z,w) → axis-angle r"""
    q_wxyz = quat_xyzw_to_wxyz(q_xyzw)
    r = quaternion_to_axis_angle(q_wxyz)
    return r

def so3_log_to_quat(r):
    """Torch: axis-angle r → quaternion (x,y,z,w)"""
    q_wxyz = axis_angle_to_quaternion(r)
    q_xyzw = quat_wxyz_to_xyzw(q_wxyz)
    return q_xyzw

def so3_log_to_quat_np(r):
    """
    r: (N,3) rotation vectors
    returns q_xyzw (N,4) = (x,y,z,w)
    """
    rot = R.from_rotvec(r)
    q_wxyz = rot.as_quat()              # (x,y,z,w) ??? NO!
    # scipy uses (x,y,z,w)
    # but we want the same convention as the model: (x,y,z,w)
    return q_wxyz  # already in (x,y,z,w)

def quat_to_euler_np(q_xyzw):
    """
    q: (N,4) in (x,y,z,w)
    Returns euler angles (N,3) in radians
    """
    # scipy expects (x,y,z,w), we're already using that
    rot = R.from_quat(q_xyzw)
    euler = rot.as_euler('xyz', degrees=False)
    return euler


