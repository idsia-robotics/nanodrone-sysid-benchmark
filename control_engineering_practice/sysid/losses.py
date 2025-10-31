import torch
import torch.nn as nn

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

import torch
import torch.nn as nn

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

