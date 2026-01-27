"""
xAILab Bamberg
University of Bamberg

@description:
Helper functions for the reproducibility.

@references:
Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
"""

# Import packages
import torch
import random
import numpy as np


def worker_seed(worker_id):
    """
    Set the seed for the current worker.

    Args:
        worker_id (int): The id of the worker.
    """

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def random_seed(seed_value, use_cuda):
    """
    Set the random seed for numpy, pytorch, python.random, pytorch GPU vars and
    dataloaders.

    Args:
        seed_value (int): The random seed value to set.
        use_cuda (bool): If True, the random seed for the GPU will be set

    Returns:
        (torch.Generator): The generator for the dataloader workers.
    """

    # Set the seed for numpy
    np.random.seed(seed_value)

    # Set the seed for pytorch
    torch.manual_seed(seed_value)

    # Set the seed for python.random
    random.seed(seed_value)

    # Set the seed for pytorch GPU variables
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True, warn_only=True)

    # Set the seed for pytorch dataloader workers
    g = torch.Generator()
    g.manual_seed(seed_value)

    return g
