"""
xAILab Bamberg
University of Bamberg

@description:
Reconstruction Metrics.
"""

# Import packages
import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure


def compute_mse(X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the mean-squared-error (MSE) for the given input tensors.

    Args:
        X1 (torch.Tensor): First input tensor of shape [B, C, H, W] or [C, H, W].
        X2 (torch.Tensor): Second input tensor of shape [B, C, H, W] or [C, H, W].

    Returns:
        torch.Tensor: MSE value.
    """

    # Check the inputs
    assert X1.shape == X2.shape, "Input tensors must have the same shape"
    assert X1.dim() in [3, 4], "Input tensors must be 3D or 4D"
    assert X2.dim() in [3, 4], "Input tensors must be 3D or 4D"

    # Calculate the mse value for each image-pair between the batches
    return F.mse_loss(X1, X2)


def compute_psnr(
    X1: torch.Tensor, X2: torch.Tensor, epsilon: float = 1e-10
) -> torch.Tensor:
    """
    Calculate the peak-signal-to-noise-ratio (PSNR) for the given original image (X1)
    and its reconstruction (X2).

    Args:
        X1 (torch.Tensor): Original input tensor of shape [B, C, H, W] or [C, H, W]
        X2 (torch.Tensor): Reconstructed input tensor of shape [B, C, H, W] or [C, H, W]

    Returns:
        torch.Tensor: PSNR value.
    """

    # Check the inputs
    assert X1.shape == X2.shape, "Input tensors must have the same shape"
    assert X1.dim() in [3, 4], "Input tensors must be 3D or 4D"
    assert X2.dim() in [3, 4], "Input tensors must be 3D or 4D"

    # If the input is a single image, add a batch dimension
    if X1.dim() == 3:
        X1 = X1.unsqueeze(0)
        X2 = X2.unsqueeze(0)

    mse = torch.mean((X1 - X2) ** 2, dim=(1, 2, 3))
    max_val = max(X1.max(), X2.max())  # Dynamically determine MAX

    # Avoid division by zero
    psnr_values = 10 * torch.log10((max_val**2) / (mse + epsilon))

    return psnr_values.mean()  # Average PSNR over batch


def compute_ssim(X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the structural-similarity-index-measure (SSIM) for the given original
    image (X1) and its reconstruction (X2).

    Args:
        X1 (torch.Tensor): Original input tensor.
        X2 (torch.Tensor): Reconstructed input tensor.

    Returns:
        torch.Tensor: SSIM value.
    """

    # Check the inputs
    assert X1.shape == X2.shape, "Input tensors must have the same shape"
    assert X1.dim() in [3, 4], "Input tensors must be 3D or 4D"
    assert X2.dim() in [3, 4], "Input tensors must be 3D or 4D"

    return structural_similarity_index_measure(X2, X1, reduction="elementwise_mean")
