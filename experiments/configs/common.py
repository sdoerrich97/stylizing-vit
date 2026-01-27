"""
xAILab Bamberg
University of Bamberg

@description:
Common Configurations.
"""

# Import packages
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CommonConfiguration:
    dataset: str = "camelyon17wilds"
    data_path: str = "./data/"
    backbone: str = "base"
    input_size: int = 224
    batch_size: int = 64
    seed: int = 265017005
    use_cuda: bool = True
    num_workers: int = 4


@dataclass
class Optimizer:
    name: str = "AdamW"
    lr: float = 1e-3
    params: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 1e-2,
            "amsgrad": False,
        }
    )


@dataclass
class LRScheduler:
    name: str = "cosine"
    params: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "decay_epochs": 90,
            "decay_milestones": (90, 180, 270),
            "cooldown_epochs": 0,
            "patience_epochs": 10,
            "decay_rate": 0.1,
            "min_lr": 0,
            "warmup_lr": 1e-5,
            "warmup_epochs": 0,
            "warmup_prefix": False,
            "noise": None,
            "noise_pct": 0.67,
            "noise_std": 1.0,
            "noise_seed": 42,
            "cycle_mul": 1.0,
            "cycle_decay": 0.1,
            "cycle_limit": 1,
            "k_decay": 1.0,
            "plateau_mode": "max",
            "step_on_epochs": True,
            "updates_per_epoch": 0,
        }
    )
