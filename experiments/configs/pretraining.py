"""
xAILab Bamberg
University of Bamberg

@description:
Pretraining configuration.
"""

# Import packages
from dataclasses import dataclass, field

# Import own scripts
from experiments.configs.common import CommonConfiguration, Optimizer, LRScheduler


@dataclass
class PretrainingConfiguration(CommonConfiguration):
    checkpoint_path: str = "./checkpoints/"
    epochs: int = 50
    batch_size: int = 64
    max_gpu_batch_size: int = 64
    effective_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    optimizer: Optimizer = field(default_factory=Optimizer)
    lr_scheduler: LRScheduler = field(default_factory=LRScheduler)
    lambda_identity: float = 70.0
    lambda_consistency: float = 1.0
    lambda_anatomical: float = 7.0
    lambda_style: float = 10.0
