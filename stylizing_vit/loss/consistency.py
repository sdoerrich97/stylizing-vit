"""
xAILab Bamberg
University of Bamberg

@description:
Feature consistency loss.
"""

# Import packages
import torch
import torch.nn.functional as F


def compute_consistency_loss(Z: torch.Tensor, Z_stylized: torch.Tensor) -> torch.Tensor:
    """
    Compute the consistency loss between the original and processed feature embeddings.

    Args:
        Z (torch.Tensor): Original feature embeddings.
        Z_stylized (torch.Tensor): Processed feature embeddings.

    Returns:
        loss (torch.Tensor): Consistency loss.
    """

    assert Z.size() == Z_stylized.size()
    assert Z.requires_grad is False

    total_loss = F.mse_loss(Z, Z_stylized)

    return total_loss
