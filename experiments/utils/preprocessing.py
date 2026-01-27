"""
xAILab Bamberg
University of Bamberg

@description:
Preprocessing utilities.
"""

# Imports
import torch
from PIL import Image
from torchvision.transforms.v2 import functional as F
from typing import Any


class ResizeWhileRetainAspectRatio(torch.nn.Module):
    """
    Custom transform to resize an image while retaining its aspect ratio.
    """

    def __init__(self, size: int, fill: int = 0):
        """
        Initialize the transform.

        Args:
            size (int): The target size for the shorter side of the image.
            fill (int): The fill value for padding.
        """
        super().__init__()
        self.size = size
        self.fill = fill

    def forward(self, inpt: Any) -> Any:
        """
        Transform the input image by resizing and padding while retaining aspect ratio.

        Args:
            inpt (Any): The input image.

        Returns:
            Any: The transformed image.
        """
        # Check if the input is a PIL Image
        is_pil_image = isinstance(inpt, Image.Image)

        # Convert input to tensor
        inpt_tensor = F.to_image(inpt)

        # Get the original dimensions
        h, w = inpt_tensor.shape[-2:]

        # Determine the new size and padding
        if h == w:
            # If the image is already square, just resize it
            resized_inpt = F.resize(inpt_tensor, (self.size, self.size))
        else:
            # Resize the longer edge to the target size
            if h > w:
                new_h, new_w = self.size, int(self.size * w / h)
            else:
                new_h, new_w = int(self.size * h / w), self.size

            resized_inpt = F.resize(inpt_tensor, (new_h, new_w))

            # Calculate padding
            pad_h = (self.size - new_h) // 2
            pad_w = (self.size - new_w) // 2
            padding = [
                pad_w,
                pad_h,
                self.size - new_w - pad_w,
                self.size - new_h - pad_h,
            ]

            # Apply padding
            resized_inpt = F.pad(resized_inpt, padding, fill=self.fill)

        # Convert back to PIL Image if the input was a PIL Image
        if is_pil_image:
            resized_inpt = F.to_pil_image(resized_inpt)

        return resized_inpt
