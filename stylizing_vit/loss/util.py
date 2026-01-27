"""
xAILab Bamberg
University of Bamberg

@description:
Utility functions for loss calculations.
"""

import torch
from typing import Tuple


def calc_mean_std(
    feat: torch.Tensor, epsilon: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the mean and standard deviation of the feature tensor.

    Args:
        feat (torch.Tensor): Input feature tensor of shape (N, C, H, W).
        epsilon (float): Small value added to the variance to avoid divide-by-zero.

    Returns:
        (torch.Tensor, torch.Tensor): Mean and standard deviation of the feature tensor.
    """
    size = feat.size()
    assert len(size) == 4, "Input tensor must have 4 dimensions (N, C, H, W)"

    N, C = feat.shape[:2]

    # Compute variance and mean in a stable way
    feat_var, feat_mean = torch.var_mean(feat.view(N, C, -1), dim=2, correction=0)

    # Ensure numerical stability
    feat_var = torch.clamp(feat_var, min=epsilon)

    # Compute standard deviation and reshape mean and std
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat_mean.view(N, C, 1, 1)

    return feat_mean, feat_std
