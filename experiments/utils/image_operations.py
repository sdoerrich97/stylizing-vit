"""
xAILab Bamberg
University of Bamberg

@description:
Image Operation utilities.
"""

# Import packages
import torch
from torchvision.transforms import functional as F
from typing import Tuple


def normalize_image(image: torch.Tensor, mean: list,
                    std: list) -> torch.Tensor:
    """
    Normalize an image.

    Args:
        image (torch.Tensor): The image tensor or batch of tensors to normalize.
        mean (list): The mean for normalization.
        std (list): The standard deviation for normalization.

    Returns:
        torch.Tensor: The normalized image tensor or batch of tensors.
    """
    device = image.device  # Get the device of the input tensor

    if len(image.shape) == 3:
        # Single image
        mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        std = torch.tensor(std).view(-1, 1, 1).to(device)
    else:
        # Batch of images
        mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
        std = torch.tensor(std).view(1, -1, 1, 1).to(device)

    return (image - mean) / std


def denormalize_image(image: torch.Tensor, mean: list, std: list) -> torch.Tensor:
    """
    Denormalize an image.

    Args:
        image (torch.Tensor): The normalized image tensor or batch of tensors.
        mean (list): The mean used for normalization.
        std (list): The standard deviation used for normalization.

    Returns:
        torch.Tensor: The denormalized image tensor or batch of tensors.
    """
    device = image.device  # Get the device of the input tensor

    if len(image.shape) == 3:
        # If the input is a single image
        mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        std = torch.tensor(std).view(-1, 1, 1).to(device)

    else:
        mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
        std = torch.tensor(std).view(1, -1, 1, 1).to(device)

    return image * std + mean


def resize_image(X: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """
    Resize (a batch of) images to the desired size.

    Args:
        X (torch.Tensor): (Batch of) images to be resized.
        size (Tuple[int, int]): Tuple of integers containing the desired size
        (height, width).

    Returns:
        torch.Tensor: (Batch of) resized images.
    """

    # Single image
    if len(X.shape) == 3:
        return F.resize(X, size)

    # Batch of images
    elif len(X.shape) == 4:
        return torch.stack([F.resize(img, size) for img in X])

    else:
        raise ValueError("Input must be a 3D or 4D tensor")
