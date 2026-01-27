"""
xAILab Bamberg
University of Bamberg

@description:
Identity loss.
"""

# Import packages
import torch
import torch.nn.functional as F


def compute_identity_loss(X: torch.Tensor, X_recon: torch.Tensor) -> torch.Tensor:
    """
    Compute the identity loss between the original and reconstructed input.

    Args:
        X (torch.Tensor): Original input.
        X_recon (torch.Tensor): Reconstructed input.

    Returns:
        loss (torch.Tensor): Reconstruction loss.
    """

    assert X.size() == X_recon.size()
    assert X.requires_grad is False

    total_loss = F.mse_loss(X, X_recon)

    return total_loss
