"""
xAILab Bamberg
University of Bamberg

@description:
Experiment configuration.
"""

# Import packages
from dataclasses import dataclass

# Import own scripts
from experiments.configs.common import CommonConfiguration


@dataclass
class GeneralExperimentConfiguration(CommonConfiguration):
    input_path: str = "./checkpoints/pretrain"  # Path to the pretrained model weights
    output_path: str = "./results/experiments/"  # Path to where the results are stored
    nr_images_to_save: int = 20


@dataclass
class ReconstructionConfig(GeneralExperimentConfiguration):
    output_path: str = "./results/reconstruction/"


@dataclass
class StyleTransferConfig(GeneralExperimentConfiguration):
    output_path: str = "./results/style_transfer/"
    batch_size_to_save: int = 10000  # Number of images to save per batch as .npz files
