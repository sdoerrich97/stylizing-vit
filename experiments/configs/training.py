"""
xAILab Bamberg
University of Bamberg

@description:
Training configuration.
"""

# Import packages
from dataclasses import dataclass, field

# Import own scripts
from experiments.configs.common import CommonConfiguration, Optimizer, LRScheduler


@dataclass
class GeneralTrainingConfiguration(CommonConfiguration):
    input_path: str = "./input_path"
    output_path: str = "./output_path"
    classifier: str = "densenet121"
    augmentation: str = "noAugment"
    portion_augmented_samples: float = 0.33


@dataclass
class TrainingConfiguration(GeneralTrainingConfiguration):
    input_path: str = "./checkpoints/pretrain"
    output_path: str = "./checkpoints/train"
    epochs: int = 100
    early_stopping: int = 10
    optimizer: Optimizer = field(default_factory=Optimizer)
    lr_scheduler: LRScheduler = field(default_factory=LRScheduler)


@dataclass
class InferenceConfig(GeneralTrainingConfiguration):
    input_path: str = "./checkpoints/train"
    output_path: str = "./results/train"
