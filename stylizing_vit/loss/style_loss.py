"""
xAILab Bamberg
University of Bamberg

@description:
Style Loss.
"""

# Import packages
import torch
import torch.nn.functional as F

# Import own scripts
from stylizing_vit.loss.util import calc_mean_std


def compute_style_loss(Z: torch.Tensor, Z_stylized: torch.Tensor) -> torch.Tensor:
    """
    Compute the loss for the style features between the original
    and processed feature embeddings.

    Args:
        Z (torch.Tensor): Feature embeddings of the original image.
        Z_stylized (torch.Tensor): Feature embeddings of the stylized image.

    Returns:
        loss (torch.Tensor): Style loss.
    """

    assert Z.size() == Z_stylized.size()
    assert Z.requires_grad is False

    # Calculate the mean and std values
    Z_mean, Z_std = calc_mean_std(Z)
    Z_stylized_mean, Z_stylized_std = calc_mean_std(Z_stylized)

    # Calculate the loss between the mean values of the style feature embeddings
    mean_style_loss = F.mse_loss(Z_mean, Z_stylized_mean)

    # Calculate the loss between the std values of the style feature embeddings
    std_style_loss = F.mse_loss(Z_std, Z_stylized_std)

    # Add the current loss to the total loss
    total_loss = mean_style_loss + std_style_loss

    return total_loss
