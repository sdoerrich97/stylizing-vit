"""
xAILab Bamberg
University of Bamberg

@description:
Anatomical loss.
"""

# Import packages
import torch
import torch.nn.functional as F

# Import own scripts
from stylizing_vit.loss.util import calc_mean_std


def compute_anatomical_loss(Z: torch.Tensor, Z_stylized: torch.Tensor) -> torch.Tensor:
    """
    Compute the loss for the anatomical features between the original
    and processed feature embeddings.

    Args:
        Z (torch.Tensor): Feature embeddings of the original image.
        Z_stylized (torch.Tensor): Feature embeddings of the stylized image.

    Returns:
        loss (torch.Tensor): anatomical loss.
    """

    assert Z.size() == Z_stylized.size()
    assert Z.requires_grad is False

    # Normalize the feature embeddings by mean and std along the channel dimension
    Z_mean, Z_std = calc_mean_std(Z)
    Z_stylized_mean, Z_stylized_std = calc_mean_std(Z_stylized)

    Z = (Z - Z_mean) / Z_std
    Z_stylized = (Z_stylized - Z_stylized_mean) / Z_stylized_std

    # Calculate the mean squared error loss between the original
    # and processed feature embeddings
    total_loss = F.mse_loss(Z, Z_stylized)

    return total_loss
