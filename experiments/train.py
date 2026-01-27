"""
xAILab Bamberg
University of Bamberg

@description:
Training.
"""

# Import packages
import os
import torch
import time
import timm
import argparse
import numpy as np
import torch.nn as nn
from accelerate.utils import tqdm
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch import Generator
from torch.optim import Optimizer
from torchvision.transforms import v2, AutoAugmentPolicy
from accelerate import Accelerator
from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler, create_scheduler_v2
from typing import Tuple
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from medmnistc.augmentation import AugMedMNISTC
from medmnistc.corruptions.registry import CORRUPTIONS_DS

# Import own scripts
from stylizing_vit.model import StylizingViT
from experiments.data import (
    create_dataset,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    NUM_CLASSES,
)
from experiments.utils import (
    ResizeWhileRetainAspectRatio,
    random_seed,
    worker_seed,
    calculate_passed_time,
    save_model,
)
from experiments.configs import TrainingConfiguration


def prepare_the_dataloader(
    cfg: dataclass, g: Generator, **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare the dataloader for the training and validation datasets.

    Args:
        cfg (dataclass): The pretraining configuration.
        g (Generator): The random number generator.
        **kwargs: Additional keyword arguments to pass to the dataset.

    Returns:
        (tuple): Tuple containing the training and validation dataloader.
    """

    transforms = [ResizeWhileRetainAspectRatio(size=cfg.input_size)]

    # Apply the augmentation strategy to the training set
    if "grayScale" in cfg.augmentation:
        transforms.append(v2.Grayscale(num_output_channels=3))

    elif "colorJitter" in cfg.augmentation:
        transforms.append(
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        )

    elif "autoAugment" in cfg.augmentation:
        transforms.append(v2.AutoAugment(policy=AutoAugmentPolicy.IMAGENET))

    elif "randAugment" in cfg.augmentation:
        transforms.append(v2.RandAugment(num_ops=2, magnitude=9))

    elif "trivialAugment" in cfg.augmentation:
        transforms.append(v2.TrivialAugmentWide(num_magnitude_bins=31))

    elif "augMix" in cfg.augmentation:
        transforms.append(
            v2.AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0)
        )

    elif "targetedAugment" in cfg.augmentation:
        if cfg.dataset in ["camelyon17wilds", "epistr"]:
            transforms.append(AugMedMNISTC(CORRUPTIONS_DS["pathmnist"]))

        elif "fitzpatrick17k" in cfg.dataset or "ddi" in cfg.dataset:
            transforms.append(AugMedMNISTC(CORRUPTIONS_DS["dermamnist"]))

        else:
            raise ValueError(
                f"Targeted augmentation is not available for dataset {cfg.dataset}."
            )

    transform_train = v2.Compose(
        transforms
        + [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=NORMALIZATION_MEAN[cfg.dataset], std=NORMALIZATION_STD[cfg.dataset]
            ),
        ]
    )

    transform_val = v2.Compose(
        [
            ResizeWhileRetainAspectRatio(size=cfg.input_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=NORMALIZATION_MEAN[cfg.dataset], std=NORMALIZATION_STD[cfg.dataset]
            ),
        ]
    )

    # Load the dataset
    train_set = create_dataset(
        dataset_name=cfg.dataset,
        data_path=cfg.data_path,
        split="train",
        transform=transform_train,
        **kwargs,
    )

    val_set = create_dataset(
        dataset_name=cfg.dataset,
        data_path=cfg.data_path,
        split="val",
        transform=transform_val,
        **kwargs,
    )

    # Initialize the dataloader
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        worker_init_fn=worker_seed,
        generator=g,
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        worker_init_fn=worker_seed,
        generator=g,
    )

    # Return the dataloader
    return train_loader, val_loader


def create_the_optimizer_and_lr_scheduler(
    model: nn.Module, cfg: dataclass
) -> Tuple[Optimizer, CosineLRScheduler]:
    """
    Create the optimizer and the learning rate scheduler.

    Args:
        model (nn.Module): The model to optimize.
        cfg (dataclass): The pretraining configuration.

    Returns:
        (tuple): Tuple containing the optimizer and the learning rate scheduler.
    """

    # Initialize the optimizer
    optimizer = create_optimizer_v2(
        model,
        opt=cfg.optimizer.name,
        lr=cfg.optimizer.lr,
        **cfg.optimizer.params,
    )

    # Initialize the lr scheduler
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer=optimizer,
        sched=cfg.lr_scheduler.name,
        num_epochs=cfg.epochs,
        **cfg.lr_scheduler.params,
    )

    # Upate the number of epochs in the configuration
    cfg.epochs = num_epochs

    # Return the optimizer and lr scheduler
    return optimizer, lr_scheduler


def main(cfg: dataclass, **kwargs) -> None:
    """
    Main function to start the training.

    Args:
        cfg (dataclass): The training configuration.
        **kwargs: Additional keyword arguments to pass to the dataset.
    """

    # Specify the output path
    if not os.path.exists(cfg.output_path):
        os.makedirs(cfg.output_path)

    # Depending on the augmentation strategy, specify the run name
    if "styleTransfer" in cfg.augmentation:
        apply_style_mixing = True
        augmentation_str = f"{cfg.augmentation}-stylizingVit-{cfg.backbone}"
    else:
        apply_style_mixing = False
        augmentation_str = f"{cfg.augmentation}"

    # Depending on the portion of augmented samples, specify the run name
    if cfg.augmentation in [
        "noAugment",
        "resizedCrop",
        "grayScale",
        "colorJitter",
        "trivialAugment",
        "targetedAugment",
        "autoAugment",
        "randAugment",
        "augMix",
        "cutOut",
        "randomErasing",
    ]:
        portion_augmented_samples_str = ""
    else:
        portion_augmented_samples_str = (
            f"{int(cfg.portion_augmented_samples * 100)}percent_augmented_samples-"
        )

    # Specify the run name
    run_name = f"{cfg.dataset}-{cfg.classifier}-{augmentation_str}-" \
               f"{portion_augmented_samples_str}{cfg.seed}"

    # Initialize the accelerator and run tracker
    accelerator = Accelerator(log_with="wandb")
    accelerator.init_trackers(
        project_name="stylizing-vit-training",
        config=cfg,
        init_kwargs={"wandb": {"entity": "ofu-xai", "name": run_name}},
    )

    # Set the random seed
    accelerator.print(f"Set the seed as: {cfg.seed}.")
    g = random_seed(seed_value=cfg.seed, use_cuda=cfg.use_cuda)

    # Load the dataset
    accelerator.print(
        f"Load the dataset {cfg.dataset} from {cfg.data_path} or download from "
        f"the internet."
    )
    train_loader, val_loader = prepare_the_dataloader(cfg=cfg, g=g, **kwargs)

    # Load the classifier model
    accelerator.print(f"Load the classifier with backbone {cfg.classifier}")
    model = timm.create_model(
        cfg.classifier, pretrained=False, num_classes=NUM_CLASSES[cfg.dataset]
    )
    model.requires_grad_(True)

    # If the Stylizing ViT shall be used
    if apply_style_mixing:
        # Load the model
        backbone_weights = os.path.join(
            cfg.input_path, f"{cfg.dataset}-stylizingVit-{cfg.backbone}.pth"
        )
        accelerator.print(
            f"Load the pretrained model {os.path.basename(backbone_weights)} from "
            f"{cfg.input_path}."
        )
        stylizing_vit = StylizingViT(backbone=cfg.backbone, train=False)

        stylizing_vit.load_state_dict(
            torch.load(backbone_weights, weights_only=True, map_location="cpu"),
            strict=False,
        )
        stylizing_vit.requires_grad_(False)
        stylizing_vit.eval()

        # Prepare the resize-cropping
        if "resizedCrop" in cfg.augmentation:
            resize_crop = v2.RandomResizedCrop(size=(cfg.input_size, cfg.input_size))
        else:
            resize_crop = None

        # Prepare cutout or random erasing if specified
        if "cutOut" in cfg.augmentation:
            cutout_or_erasing = v2.RandomErasing(
                p=0.5, scale=(0.02, 0.5), ratio=(1.0, 1.0), value=0.0
            )

        elif "randomErasing" in cfg.augmentation:
            cutout_or_erasing = v2.RandomErasing(
                p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value="random"
            )

        else:
            cutout_or_erasing = None

    else:
        stylizing_vit = None
        resize_crop = None
        cutout_or_erasing = None

    accelerator.print("Initialize the optimizer and lr scheduler.")
    optimizer, lr_scheduler = create_the_optimizer_and_lr_scheduler(
        model=model, cfg=cfg
    )

    accelerator.print("Initialize the loss function.")
    loss_criterion = nn.CrossEntropyLoss()
    prediction = nn.Softmax(dim=1)

    # Distribute to the specified device(s)
    accelerator.print("Distribute to the specified device(s)")
    (
        model,
        loss_criterion,
        optimizer,
        lr_scheduler,
        train_loader,
        val_loader,
        stylizing_vit,
    ) = accelerator.prepare(
        model,
        loss_criterion,
        optimizer,
        lr_scheduler,
        train_loader,
        val_loader,
        stylizing_vit,
    )

    # Initialize some helpers
    accelerator.print("Initialize some helper variables.")
    best_val_loss = np.inf  # Best validation loss
    epochs_no_improve = 0  # Counter for epochs without improvement

    nr_batches_per_epoch_train = len(train_loader)
    nr_batches_per_epoch_val = len(val_loader)

    # Start the training
    accelerator.print("Start the training.")
    start_time = time.time()
    for epoch in range(cfg.epochs):
        # Stop the time for the epoch
        start_time_epoch = time.time()

        # =================================================
        # Training Loop
        # =================================================
        # Set the progress bar
        pbar_train = tqdm(
            total=nr_batches_per_epoch_train,
            bar_format="{l_bar}{bar}",
            ncols=80,
            initial=0,
            position=0,
            leave=False,
        )
        pbar_train.set_description(f"Train [{epoch + 1}/{cfg.epochs}]")

        # Initialize helpers for the current epoch
        num_updates = (
            epoch * nr_batches_per_epoch_train
        )  # Number of total updates for the lr scheduler
        total_loss_train = 0.0  # Total train loss for the current epoch
        y_target_train, y_predicted_train = torch.tensor([]), torch.tensor([])

        # Set the model to training mode
        model.train()

        # Run the training loop
        for i, (x, y) in enumerate(train_loader):
            # Apply the style mixing
            if apply_style_mixing:
                # Get the number of samples to augment
                num_samples_to_augment = max(
                    1, int(cfg.portion_augmented_samples * len(x))
                )

                # Select the required number of samples to augment
                idx_samples_to_augment = torch.randperm(len(x))[:num_samples_to_augment]
                x_to_augment = x[idx_samples_to_augment].clone()

                # Create the batch of style images
                x_style = x_to_augment.clone().roll(shifts=-1, dims=0)

                # Apply Random Cropping to the style images
                if resize_crop is not None:
                    x_style = resize_crop(x_style)

                # Run the style infusion for the given batch of anatomy images
                # and style images
                x_infused = stylizing_vit(x_to_augment, x_style)

                # Replace the selected samples with the augmented samples
                x[idx_samples_to_augment] = x_infused

            # Apply cutout or random erasing to the images
            if cutout_or_erasing is not None:
                x = torch.stack([cutout_or_erasing(x) for x in x])

            # Run the forward pass of the classifer
            outputs = model(x)

            # Compute the loss
            loss = loss_criterion(outputs, y)

            # Backward pass, weight and lr scheduler update
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step_update(
                num_updates=num_updates
            )  # Good practice to update timm scheduler twice
            optimizer.zero_grad()

            # Accumulate the loss per batch and store the predictions
            total_loss_train += loss.item()
            y_target_train = torch.cat([y_target_train, y.cpu()])
            y_predicted_train = torch.cat(
                [y_predicted_train, prediction(outputs).detach().cpu()], dim=0
            )

            # Update the progress bar
            pbar_train.update(1)

        # Update the lr scheduler the second time
        lr_scheduler.step(epoch + 1)

        # Average the loss across the batches and compute the accuracy and AUC
        total_loss_train /= nr_batches_per_epoch_train

        # Convert tensors to numpy arrays
        y_target_train_np = y_target_train.numpy()
        y_predicted_train_np = y_predicted_train.numpy()

        # Compute performance metrics
        if NUM_CLASSES[cfg.dataset] == 2:
            accuracy_train = accuracy_score(
                y_target_train_np, y_predicted_train_np[:, -1] > 0.5
            )  # Threshold at 0.5
            balanced_accuracy_train = balanced_accuracy_score(
                y_target_train_np, y_predicted_train_np[:, -1] > 0.5
            )  # Threshold at 0.5
            roc_auc_train = roc_auc_score(
                y_target_train_np, y_predicted_train_np[:, -1]
            )
        else:
            accuracy_train = accuracy_score(
                y_target_train_np, np.argmax(y_predicted_train_np, axis=1)
            )
            balanced_accuracy_train = balanced_accuracy_score(
                y_target_train_np, np.argmax(y_predicted_train_np, axis=1)
            )
            roc_auc_train = roc_auc_score(
                y_target_train_np, y_predicted_train_np, multi_class="ovr"
            )

        # Log the metrics
        accelerator.log(
            {
                "train/total_loss": total_loss_train,
                "train/accuracy": accuracy_train,
                "train/balanced_accuracy": balanced_accuracy_train,
                "train/auc": roc_auc_train,
            },
            step=epoch + 1,
        )

        # Log the results
        accelerator.print(f"Train Loss [{epoch + 1}/{cfg.epochs}]: {total_loss_train}")
        accelerator.print(
            f"Train Accuracy [{epoch + 1}/{cfg.epochs}]: {accuracy_train}"
        )
        accelerator.print(
            f"Train Balanced Accuracy [{epoch + 1}/{cfg.epochs}]: "
            f"{balanced_accuracy_train}"
        )
        accelerator.print(f"Train AUC [{epoch + 1}/{cfg.epochs}]: {roc_auc_train}")

        # =================================================
        # Validation Loop
        # =================================================
        # Set the progress bar
        pbar_val = tqdm(
            total=nr_batches_per_epoch_val,
            bar_format="{l_bar}{bar}",
            ncols=80,
            initial=0,
            position=0,
            leave=False,
        )
        pbar_val.set_description(f"Val [{epoch + 1}/{cfg.epochs}]")

        # Total validation loss for the current epoch
        total_loss_val = 0.0
        y_target_val, y_predicted_val = torch.tensor([]), torch.tensor([])

        # Set the model to evaluation mode
        model.eval()

        # Run the validation loop
        with torch.no_grad():
            for i, (x_val, y_val) in enumerate(val_loader):
                # Run the forward pass
                outputs_val = model(x_val)

                # Compute the loss and perform backpropagation
                loss_val = loss_criterion(outputs_val, y_val)

                # Accumulate the loss per batch and store the predictions
                total_loss_val += loss_val.item()
                y_target_val = torch.cat([y_target_val, y_val.cpu()])
                y_predicted_val = torch.cat(
                    [y_predicted_val, prediction(outputs_val).cpu()], dim=0
                )

                # Update the progress bar
                pbar_val.update(1)

        # Average the loss across the batches and compute the accuracy and AUC
        total_loss_val /= nr_batches_per_epoch_val

        # Convert tensors to numpy arrays
        y_target_val_np = y_target_val.numpy()
        y_predicted_val_np = y_predicted_val.numpy()

        # Compute performance metrics
        if NUM_CLASSES[cfg.dataset] == 2:
            accuracy_val = accuracy_score(
                y_target_val_np, y_predicted_val_np[:, -1] > 0.5
            )  # Threshold at 0.5
            balanced_accuracy_val = balanced_accuracy_score(
                y_target_val_np, y_predicted_val_np[:, -1] > 0.5
            )  # Threshold at 0.5
            roc_auc_val = roc_auc_score(y_target_val_np, y_predicted_val_np[:, -1])
        else:
            accuracy_val = accuracy_score(
                y_target_val_np, np.argmax(y_predicted_val_np, axis=1)
            )
            balanced_accuracy_val = balanced_accuracy_score(
                y_target_val_np, np.argmax(y_predicted_val_np, axis=1)
            )
            roc_auc_val = roc_auc_score(
                y_target_val_np, y_predicted_val_np, multi_class="ovr"
            )

        # Log the metrics
        accelerator.log(
            {
                "val/total_loss": total_loss_val,
                "val/accuracy": accuracy_val,
                "val/balanced_accuracy": balanced_accuracy_val,
                "val/auc": roc_auc_val,
            },
            step=epoch + 1,
        )

        # Log the results
        accelerator.print(f"Val Loss [{epoch + 1}/{cfg.epochs}]: {total_loss_val}")
        accelerator.print(f"Val Accuracy [{epoch + 1}/{cfg.epochs}]: {accuracy_val}")
        accelerator.print(
            f"Val Balanced Accuracy [{epoch + 1}/{cfg.epochs}]: {balanced_accuracy_val}"
        )
        accelerator.print(f"Val AUC [{epoch + 1}/{cfg.epochs}]: {roc_auc_val}")

        # Save the model
        if total_loss_val < best_val_loss:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        best_val_loss = save_model(
            accelerator=accelerator,
            model=model,
            val_loss=total_loss_val,
            best_val_loss=best_val_loss,
            save_name=run_name,
            save_path=cfg.output_path,
        )

        # Stop the time for the epoch
        end_time_epoch = time.time()
        hours_epoch, minutes_epoch, seconds_epoch = calculate_passed_time(
            start_time_epoch, end_time_epoch
        )
        accelerator.print(
            "Elapsed time for epoch: {:0>2}:{:0>2}:{:05.2f}".format(
                hours_epoch, minutes_epoch, seconds_epoch
            )
        )

        # Check for early stopping
        if epochs_no_improve == cfg.early_stopping:
            print("\tEarly stopping!")
            break

    # End the training
    accelerator.end_training()

    # Stop the time for the training
    end_time = time.time()
    hours, minutes, seconds = calculate_passed_time(start_time, end_time)
    accelerator.print(
        "Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(hours, minutes, seconds)
    )


if __name__ == "__main__":
    # Read out the command line parameters.
    parser = argparse.ArgumentParser(description="Training.")

    # Pretraining configurations (optimizer and lr scheduler can not be adjusted as of
    # now via command line arguments)
    parser.add_argument("--dataset", type=str, help="Dataset to use for pretraining.")
    parser.add_argument(
        "--data_path", type=str, help="Path to where the dataset shall be or is stored."
    )
    parser.add_argument(
        "--input_path", type=str, help="Path to where pretrained models are stored."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to where the trained models shall be stored.",
    )
    parser.add_argument(
        "--classifier", type=str, help="Classifier model to use for the training."
    )
    parser.add_argument(
        "--backbone", type=str, help="Backbone model to use for the Stylizing ViT."
    )
    parser.add_argument(
        "--input_size", type=int, help="Which input size to use for the model."
    )

    parser.add_argument(
        "--augmentation",
        type=str,
        help="Augmentation strategy to use for the training.",
    )
    parser.add_argument(
        "--portion_augmented_samples",
        type=float,
        help="Portion of augmented samples to use for the training.",
    )

    parser.add_argument("--epochs", type=int, help="Number of epochs to train for.")
    parser.add_argument(
        "--early_stopping",
        type=int,
        help="Number of epochs to wait for early stopping.",
    )
    parser.add_argument("--batch_size", type=int, help="Batch size for training.")

    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--use_cuda", type=bool, help="Use CUDA for training")
    parser.add_argument(
        "--num_workers", type=int, help="Number of workers for the dataloader"
    )

    args = parser.parse_args()

    # Initialize the configurations
    cfg = TrainingConfiguration()

    # Overwrite the default configs with the given command line arguments
    for arg in vars(args):
        setattr(cfg, arg, getattr(args, arg))

    # Create the configurations
    print(f"Initialize the configurations: {cfg}")

    # Start the training
    main(cfg=cfg)
