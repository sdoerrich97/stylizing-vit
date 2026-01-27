"""
xAILab Bamberg
University of Bamberg
"""

# Import packages
import torch
from dataclasses import dataclass

from stylizing_vit.loss.identity import compute_identity_loss
from stylizing_vit.loss.consistency import compute_consistency_loss
from stylizing_vit.loss.anatomical_loss import compute_anatomical_loss
from stylizing_vit.loss.style_loss import compute_style_loss


@dataclass
class Loss:
    identity: torch.Tensor = torch.tensor(0.0)
    consistency: torch.Tensor = torch.tensor(0.0)
    anatomical: torch.Tensor = torch.tensor(0.0)
    style: torch.Tensor = torch.tensor(0.0)


__all__ = [
    "compute_identity_loss",
    "compute_consistency_loss",
    "compute_anatomical_loss",
    "compute_style_loss",
    "Loss",
]
