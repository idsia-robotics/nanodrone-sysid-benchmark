import math

import torch
import torch.nn as nn
from pytorch3d.transforms import so3_exp_map, so3_log_map, so3_relative_angle

class QuadStateMSELoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # The model must implement quad_state_error(pred[i], target[i])

    def forward(self, pred, target):
        batch_errors = []
        for i in range(pred.shape[0]):
            e = self.model.quad_state_error(pred[i], target[i])  # [12] error state
            batch_errors.append(e ** 2)
        batch_errors = torch.stack(batch_errors)  # [B, 12]
        return torch.mean(batch_errors)

class ScaledMSELoss(nn.Module):
    def __init__(self, scale_vector, eps=1e-6):
        super().__init__()
        self.scale = torch.tensor(scale_vector).float()
        self.eps = eps

    def forward(self, pred, target):
        # pred, target: [batch, D]
        scale = self.scale.to(pred.device).unsqueeze(0)  # [1, D]
        pred_scaled = pred / (scale + self.eps)
        target_scaled = target / (scale + self.eps)
        return torch.mean((pred_scaled - target_scaled) ** 2)

class WeightedMSELoss(nn.Module):
    """
    Exponentially weighted MSE loss for multi-step prediction.

    Each time step h in the prediction horizon is weighted as:
        w_h = exp(-lambda_ * (h - 1))
    so that early steps contribute more to the total loss.

    Args:
        lambda_ (float): exponential decay factor (default: 0.05)
    """
    def __init__(self, lambda_=0.05):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, pred, true):
        """
        Args:
            pred: (B, N, D) predicted state sequence
            true: (B, N, D) true state sequence
        Returns:
            scalar loss (torch.Tensor)
        """
        assert pred.shape == true.shape, "pred and true must have same shape"
        N = pred.size(1)

        # Compute exponential weights over horizon
        weights = torch.exp(-self.lambda_ * torch.arange(N, device=pred.device))
        weights = weights / weights.sum()  # normalize to 1

        # Weighted MSE
        loss = ((pred - true) ** 2 * weights.view(1, N, 1)).mean()
        return loss

import torch
import torch.nn as nn

# --------------------------------------------------
# SO(3) exponential + logarithm maps
# --------------------------------------------------

def clamp_rotvec(phi, max_angle=math.pi - 1e-3, eps=1e-8):
    """
    Clamp rotation vectors so that ||phi|| <= max_angle.
    This prevents the network from producing insane angles that
    break exp/log numerics.
    """
    angle = torch.norm(phi, dim=-1, keepdim=True)       # (...,1)
    scale = torch.clamp(angle, max=max_angle) / (angle + eps)
    return phi * scale

def so3_exp(phi, eps=1e-8):
    """
    phi: (..., 3)
    returns R: (..., 3, 3)
    """
    angle = torch.norm(phi, dim=-1, keepdim=True)     # (...,1)
    axis = phi / (angle + eps)                        # (...,3)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]

    ca = torch.cos(angle)[..., 0]                     # (...,)
    sa = torch.sin(angle)[..., 0]
    C  = 1 - ca

    # Allocate output
    R = torch.zeros(phi.shape[:-1] + (3,3), device=phi.device, dtype=phi.dtype)

    R[..., 0,0] = ca + x*x*C
    R[..., 0,1] = x*y*C - z*sa
    R[..., 0,2] = x*z*C + y*sa

    R[..., 1,0] = y*x*C + z*sa
    R[..., 1,1] = ca + y*y*C
    R[..., 1,2] = y*z*C - x*sa

    R[..., 2,0] = z*x*C - y*sa
    R[..., 2,1] = z*y*C + x*sa
    R[..., 2,2] = ca + z*z*C

    return R


def so3_log(R, eps=1e-8):
    """
    Log map from rotation matrix to axis-angle.
    R: (..., 3, 3)
    returns phi: (..., 3)
    """
    # Ensure numerical symmetry (optional but helps)
    # R = 0.5 * (R + R.transpose(-1, -2))

    trace = R[..., 0,0] + R[..., 1,1] + R[..., 2,2]
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)

    theta = torch.acos(cos_theta)                  # (...,)

    wx = R[..., 2,1] - R[..., 1,2]
    wy = R[..., 0,2] - R[..., 2,0]
    wz = R[..., 1,0] - R[..., 0,1]
    w = torch.stack([wx, wy, wz], dim=-1)          # (...,3)

    sin_theta = torch.sin(theta)

    phi = torch.zeros_like(w)

    small = theta < 1e-4
    large = ~small

    # Small-angle approximation: log(R) ≈ 0.5 * w
    if small.any():
        phi[small] = 0.5 * w[small]

    # Normal case
    if large.any():
        scale = (theta[large] / (2.0 * sin_theta[large] + eps)).unsqueeze(-1)
        phi[large] = scale * w[large]

    return phi


# --------------------------------------------------
# Weighted SO(3)-aware multi-step loss
# --------------------------------------------------

class WeightedGeodesicLoss(nn.Module):
    """
    Multi-step loss:
      - MSE on p(3), v(3), omega(3)
      - geodesic SO(3) loss on r_log(3)
      - exponentially decaying weights over horizon

    pred, true: (B, N, 12)
    state = [p (3), v (3), r_log (3), omega (3)]
    """

    def __init__(self, lambda_=0.05, w_pos=1.0, w_vel=0.1, w_omega=10.0, w_rot=10.0):
        super().__init__()
        self.lambda_ = lambda_
        self.w_pos = w_pos
        self.w_vel = w_vel
        self.w_rot = w_rot
        self.w_omega = w_omega

    def forward(self, pred, true):
        assert pred.shape == true.shape
        B, N, D = pred.shape
        assert D == 12, "Expected state dim = 12"

        # --- Exponential weights over horizon ---
        weights = torch.exp(-self.lambda_ * torch.arange(N, device=pred.device))
        # weights = weights / weights.sum()            # (N,)
        W = weights.view(1, N, 1)                    # (1,N,1)

        # --- Split components ---
        p_pred, v_pred, r_pred, w_pred = pred[..., :3], pred[..., 3:6], pred[..., 6:9], pred[..., 9:12]
        p_gt,   v_gt,   r_gt,   w_gt   = true[..., :3], true[..., 3:6], true[..., 6:9], true[..., 9:12]

        # --- Clamp rotation vectors to avoid crazy angles ---
        r_pred = clamp_rotvec(r_pred)
        # r_gt   = clamp_rotvec(r_gt)

        # --- MSE terms ---
        loss_p = ((p_pred - p_gt)**2 * W).mean()
        loss_v = ((v_pred - v_gt)**2 * W).mean()
        loss_w = ((w_pred - w_gt)**2 * W).mean()

        # --- SO(3) geodesic rotation loss ---
        # R_pred = so3_exp(r_pred)            # (B,N,3,3)
        # R_gt   = so3_exp(r_gt)
        # R_err  = torch.matmul(R_pred.transpose(-1, -2), R_gt)
        # phi_err = so3_log(R_err)            # (B,N,3)
        # loss_R = ((phi_err**2) * W).mean()
        #
        # Convert rotation vectors -> rotation matrices
        # ------------------------------------------
        # ROTATION LOSS (SO3, PyTorch3D)
        # Must flatten to (B*N, 3)
        # ------------------------------------------
        # Flatten rotvecs: (B, N, 3) -> (B*N, 3)
        r_pred_flat = r_pred.reshape(B * N, 3)
        r_gt_flat = r_gt.reshape(B * N, 3)

        # Compute rotation matrices: (B*N, 3, 3)
        R_pred_flat = so3_exp_map(r_pred_flat)
        R_gt_flat = so3_exp_map(r_gt_flat)

        # Geodesic angle: (B*N,)
        rot_err_flat = so3_relative_angle(R_pred_flat, R_gt_flat)

        # Reshape back to (B, N)
        rot_err = rot_err_flat.view(B, N)

        # Weighted squared loss
        loss_R = ((rot_err ** 2) * W.squeeze(-1)).mean()

        total = self.w_pos * loss_p + self.w_vel * loss_v + self.w_omega * loss_w + self.w_rot * loss_R

        # print("p:", loss_p.item(),
        #       "v:", loss_v.item(),
        #       "w:", self.w_omega * loss_w.item(),
        #       "R:", self.w_rot * loss_R.item())

        # # Optional: debug guard
        # if torch.isnan(total):
        #     print("NaN in loss: ",
        #           "p:", loss_p.item(),
        #           "v:", loss_v.item(),
        #           "w:", loss_w.item(),
        #           "R:", loss_R.item())
        #     raise RuntimeError("NaN in WeightedGeodesicLoss")

        return total

