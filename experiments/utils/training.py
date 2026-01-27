"""
xAILab Bamberg
University of Bamberg

@description:
Helper functions for training.
"""

# Import packages
import os
import torch.nn as nn
from accelerate import Accelerator
from typing import Tuple


def calculate_passed_time(start_time: float, end_time: float) -> Tuple[int, int, float]:
    """
    Calculate the passed time.

    Args:
        start_time (float): Start time.
        end_time (float): End time.

    Returns:
        (int, int, float): total duration in hours, minutes and seconds
    """

    # Calculate the duration
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # Return the duration in hours, minutes and seconds
    return int(hours), int(minutes), seconds


def save_model(
    accelerator: Accelerator,
    model: nn.Module,
    val_loss: float,
    best_val_loss: float,
    save_name: str,
    save_path: str,
) -> float:
    """
    Save the model based on the current epoch and validation loss.

    Args:
        accelerator (Accelerator): The accelerator instance.
        model (torch.nn.Module): The model to be saved.
        val_loss (float): The current validation loss.
        best_val_loss (float): The best validation loss so far.
        save_name (str): The name of the model to save.
        save_path (str): The base path to save the model.

    Returns:
        (float): The best validation loss so far.
    """

    # Check if the save_path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Let all processes finish before saving the model
    accelerator.wait_for_everyone()

    # Unwrap the model
    unwrapped_model = accelerator.unwrap_model(model)

    # Save the best model depending on the total validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        accelerator.save(
            unwrapped_model.state_dict(),
            os.path.join(save_path, f"{save_name}.pth"),
        )

    return best_val_loss
