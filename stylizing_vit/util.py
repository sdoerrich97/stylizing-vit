"""
xAILab Bamberg
University of Bamberg

@description:
Utilities.
"""

# Import packages
import torch
from torchvision.transforms.v2 import functional as F


def resize_image(inpt: torch.Tensor, size: int, fill: int = 0) -> torch.Tensor:
    """
    Resize the input image while retaining the original aspect ratio.

    Args:
        inpt (torch.Tensor): The input image.
        size (int): The target size for the shorter side of the image.
        fill (int): The fill value for padding.

    Returns:
        torch.Tensor: The transformed image.
    """

    # Convert input to tensor
    inpt_tensor = F.to_image(inpt)

    # Get the original dimensions
    h, w = inpt_tensor.shape[-2:]

    if h == size and w == size:
        # If the image is already the target size, return it as is
        return inpt_tensor

    # Determine the new size and padding
    if h == w:
        # If the image is already square, just resize it
        resized_inpt = F.resize(inpt_tensor, (size, size))
    else:
        # Resize the longer edge to the target size
        if h > w:
            new_h, new_w = size, int(size * w / h)
        else:
            new_h, new_w = int(size * h / w), size

        resized_inpt = F.resize(inpt_tensor, (new_h, new_w))

        # Calculate padding
        pad_h = (size - new_h) // 2
        pad_w = (size - new_w) // 2
        padding = [pad_w, pad_h, size - new_w - pad_w, size - new_h - pad_h]

        # Apply padding
        resized_inpt = F.pad(resized_inpt, padding, fill=fill)

    return resized_inpt


def normalize_image(image: torch.Tensor, mean: list, std: list) -> torch.Tensor:
    """
    Normalize an image tensor of shape (C, H, W) or (B, C, H, W).

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
    Denormalize an image tensor of shape (C, H, W) or (B, C, H, W).

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


def patchify_image(X: torch.Tensor, in_channel: int, patch_size: int) -> torch.Tensor:
    """
    Split each image of the input batch into patches.

    Args:
        in_channel (int): Channel dimension of the input.
        X (torch.Tensor): Input batch of images of shape: (B, C, H, W).
        patch_size (int): Patch size.

    Returns:
        torch.Tensor: Patchified batch of images of shape: (B, N, P * P * C).
    """

    assert X.shape[2] == X.shape[3] and X.shape[2] % patch_size == 0

    h = w = X.shape[2] // patch_size
    X = X.reshape(shape=(X.shape[0], in_channel, h, patch_size, w, patch_size))

    X = torch.einsum("nchpwq->nhwpqc", X)
    X = X.reshape(shape=(X.shape[0], h * w, patch_size**2 * in_channel))

    return X


def unpatchify_image(X: torch.Tensor, patch_size: int, in_channel: int) -> torch.Tensor:
    """
    Unpatchify an input into the original image shape.

    Args:
        X (torch.Tensor): Patchified batch of images of shape: [B, N, L]
        or [B, N, C, P, P]
        patch_size (int): Patch size.
        in_channel (int): Channel dimension of the input.

    Returns:
        torch.Tensor: Original image-sized batch of images of shape: [B, C, H, W]
    """

    h = w = int(X.shape[1] ** 0.5)
    assert h * w == X.shape[1]

    if len(X.shape) == 3:
        X = X.reshape(shape=(X.shape[0], h, w, patch_size, patch_size, in_channel))

    elif len(X.shape) == 5:
        X = X.reshape(shape=(X.shape[0], h, w, X.shape[3], X.shape[4], X.shape[2]))

    X = torch.einsum("nhwpqc->nchpwq", X)
    X = X.reshape(shape=(X.shape[0], in_channel, h * patch_size, h * patch_size))

    return X
