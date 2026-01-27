"""
xAILab Bamberg
University of Bamberg

@description:
Pretraining.
"""

# Import packages
import os
import torch
import time
import argparse
import numpy as np
import torch.nn as nn
from accelerate.utils import tqdm
from torch.utils.data import DataLoader
from torch import Generator
from torch.optim import Optimizer
from torchvision.transforms import v2
from accelerate import Accelerator
from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler, create_scheduler_v2
from typing import Tuple, Union, Any, Iterable

# Import own scripts
from stylizing_vit.model import StylizingViT
from experiments.data import create_dataset, NORMALIZATION_MEAN, NORMALIZATION_STD
from experiments.utils import (
    ResizeWhileRetainAspectRatio,
    random_seed,
    worker_seed,
    calculate_passed_time,
    save_model,
    denormalize_image,
)
from experiments.configs import PretrainingConfiguration
from experiments.metrics import compute_psnr


def prepare_the_dataloader(
    cfg: PretrainingConfiguration, g: Generator, **kwargs: Any
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare the dataloader for the training and validation datasets.

    Args:
        cfg (PretrainingConfiguration): The pretraining configuration object containing
        dataset settings.
        g (Generator): The random number generator for reproducibility.
        **kwargs (Any): Additional keyword arguments to pass to the dataset creation
        function.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing the training dataloader and
        the validation dataloader.
    """

    # Create the data loading transforms for the training and validation datasets.
    transform_train = v2.Compose(
        [
            ResizeWhileRetainAspectRatio(size=cfg.input_size),
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
        batch_size=cfg.effective_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        worker_init_fn=worker_seed,
        generator=g,
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.effective_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        worker_init_fn=worker_seed,
        generator=g,
    )

    # Return the dataloader
    return train_loader, val_loader


def create_the_optimizer_and_lr_scheduler(
    params_to_optimize: Union[nn.Module, Iterable[nn.Parameter]],
    cfg: PretrainingConfiguration,
) -> Tuple[Optimizer, CosineLRScheduler]:
    """
    Create the optimizer and the learning rate scheduler.

    Args:
        params_to_optimize (Union[nn.Module, Iterable[nn.Parameter]]): The model or
        parameters to optimize.
        cfg (PretrainingConfiguration): The pretraining configuration object containing
        optimizer and scheduler settings.

    Returns:
        Tuple[Optimizer, CosineLRScheduler]: A tuple containing the initialized
        optimizer and the learning rate scheduler.
    """

    # Initialize the optimizer
    optimizer = create_optimizer_v2(
        params_to_optimize,
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


def main(cfg: PretrainingConfiguration, **kwargs: Any) -> None:
    """
    Main function to start the pretraining of the Stylizing ViT model.

    Args:
        cfg (PretrainingConfiguration): The pretraining configuration object.
        **kwargs (Any): Additional keyword arguments to pass to the dataset or other
        components.

    Returns:
        None
    """

    # Specify the output path
    output_path = os.path.join(cfg.checkpoint_path, "pretrain")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    run_name = f"{cfg.dataset}-stylizingVit-{cfg.backbone}"

    # Initialize the accelerator and run tracker
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps, log_with="wandb"
    )
    accelerator.init_trackers(
        project_name="stylizing-vit-pretraining",
        config=cfg,
        init_kwargs={"wandb": {"entity": "ofu-xai", "name": run_name}},
    )

    # Set the random seed
    print(accelerator.device)
    accelerator.print(f"Set the seed as: {cfg.seed}.")
    g = random_seed(seed_value=cfg.seed, use_cuda=cfg.use_cuda)

    # Load the dataset
    accelerator.print(
        f"Load the dataset {cfg.dataset} from {cfg.data_path} or download from the "
        f"internet."
    )
    train_loader, val_loader = prepare_the_dataloader(cfg=cfg, g=g, **kwargs)

    # Load the model
    accelerator.print(f"Load the model with backbone {cfg.backbone}")
    model = StylizingViT(backbone=cfg.backbone, train=True)
    model.encoder.requires_grad_(True)
    model.bottleneck.requires_grad_(True)
    model.post_process_conv.requires_grad_(True)

    # Set vgg_encoder to eval mode and ensure its parameters are not updated
    model.vgg_encoder.requires_grad_(False)
    model.vgg_encoder.eval()

    # Collect parameters to be optimized
    params_to_optimize = (
        list(model.encoder.parameters())
        + list(model.bottleneck.parameters())
        + list(model.post_process_conv.parameters())
    )

    accelerator.print("Initialize the optimizer and lr scheduler.")
    optimizer, lr_scheduler = create_the_optimizer_and_lr_scheduler(
        params_to_optimize=params_to_optimize, cfg=cfg
    )

    # Distribute to the specified device(s)
    accelerator.print("Distribute to the specified device(s)")
    model, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_loader, val_loader
    )

    # Initialize some helpers
    accelerator.print("Initialize some helper variables.")
    best_val_loss = np.inf  # Best validation loss
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
        num_updates = epoch * nr_batches_per_epoch_train
        (
            total_loss_train,
            total_identity_loss_train,
            total_consistency_loss_train,
            total_anatomical_loss_train,
            total_style_loss_train,
        ) = (0.0, 0.0, 0.0, 0.0, 0.0)
        average_psnr_train = 0.0

        # Set the model to train mode
        model.encoder.train()
        model.bottleneck.train()
        model.post_process_conv.train()

        # Run the training loop
        for i, x in enumerate(train_loader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()

                # Check for labeled and unlabeled data
                if isinstance(x, list):
                    x = x[0]

                # Run the forward pass
                loss, x_recon = model(x)

                # Calculate the total loss
                loss_train = (
                    cfg.lambda_identity * loss.identity
                    + cfg.lambda_consistency * loss.consistency
                    + cfg.lambda_anatomical * loss.anatomical
                    + cfg.lambda_style * loss.style
                )

                # Backward pass, weight and lr scheduler update
                accelerator.backward(loss_train)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                lr_scheduler.step_update(
                    num_updates=num_updates
                )  # Good practice to update timm scheduler twice
                optimizer.zero_grad()

                # Accumulate the loss per batch
                total_loss_train += loss_train.item()
                total_identity_loss_train += cfg.lambda_identity * loss.identity.item()
                total_consistency_loss_train += (
                    cfg.lambda_consistency * loss.consistency.item()
                )
                total_anatomical_loss_train += (
                    cfg.lambda_anatomical * loss.anatomical.item()
                )
                total_style_loss_train += cfg.lambda_style * loss.style.item()

                # Calculate the PSNR of the identity reconstructions
                x = denormalize_image(
                    x, NORMALIZATION_MEAN[cfg.dataset], NORMALIZATION_STD[cfg.dataset]
                )
                x_recon = denormalize_image(
                    x_recon,
                    NORMALIZATION_MEAN[cfg.dataset],
                    NORMALIZATION_STD[cfg.dataset],
                )
                average_psnr_train += compute_psnr(x, x_recon).item()

                # Update the progress bar
                pbar_train.update(1)

        # Update the lr scheduler the second time
        lr_scheduler.step(epoch + 1)

        # Average the loss across the batches and log the result
        total_loss_train /= nr_batches_per_epoch_train
        total_identity_loss_train /= nr_batches_per_epoch_train
        total_consistency_loss_train /= nr_batches_per_epoch_train
        total_anatomical_loss_train /= nr_batches_per_epoch_train
        total_style_loss_train /= nr_batches_per_epoch_train
        average_psnr_train /= nr_batches_per_epoch_train

        accelerator.log({"train/total_loss": total_loss_train}, step=epoch + 1)
        accelerator.log(
            {"train/identity_loss": total_identity_loss_train}, step=epoch + 1
        )
        accelerator.log(
            {"train/consistency_loss": total_consistency_loss_train}, step=epoch + 1
        )
        accelerator.log(
            {"train/anatomical_loss": total_anatomical_loss_train}, step=epoch + 1
        )
        accelerator.log({"train/style_loss": total_style_loss_train}, step=epoch + 1)
        accelerator.log({"train/average_psnr": average_psnr_train}, step=epoch + 1)

        accelerator.print(f"Train Loss [{epoch + 1}/{cfg.epochs}]: {total_loss_train}")
        accelerator.print(
            f"Train Identity Loss [{epoch + 1}/{cfg.epochs}]: "
            f"{total_identity_loss_train}"
        )
        accelerator.print(
            f"Train Consistency Loss [{epoch + 1}/{cfg.epochs}]: "
            f"{total_consistency_loss_train}"
        )
        accelerator.print(
            f"Train anatomical Loss [{epoch + 1}/{cfg.epochs}]: "
            f"{total_anatomical_loss_train}"
        )
        accelerator.print(
            f"Train Style Loss [{epoch + 1}/{cfg.epochs}]: {total_style_loss_train}"
        )
        accelerator.print(
            f"Train Average PSNR [{epoch + 1}/{cfg.epochs}]: {average_psnr_train}"
        )

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
        (
            total_loss_val,
            total_identity_loss_val,
            total_consistency_loss_val,
            total_anatomical_loss_val,
            total_style_loss_val,
        ) = (0.0, 0.0, 0.0, 0.0, 0.0)
        average_psnr_val = 0.0

        # Set the model to eval mode
        model.encoder.eval()
        model.bottleneck.eval()
        model.post_process_conv.eval()

        # Run the validation loop
        with torch.no_grad():
            for i, x_val in enumerate(val_loader):
                # Check for labeled and unlabeled data
                if isinstance(x_val, list):
                    x_val = x_val[0]

                # Run the forward pass
                loss, x_recon_val = model(x_val)

                # Calculate the total loss
                loss_val = (
                    cfg.lambda_identity * loss.identity
                    + cfg.lambda_consistency * loss.consistency
                    + cfg.lambda_anatomical * loss.anatomical
                    + cfg.lambda_style * loss.style
                )

                # Accumulate the loss per batch
                total_loss_val += loss_val.item()
                total_identity_loss_val += cfg.lambda_identity * loss.identity.item()
                total_consistency_loss_val += (
                    cfg.lambda_consistency * loss.consistency.item()
                )
                total_anatomical_loss_val += (
                    cfg.lambda_anatomical * loss.anatomical.item()
                )
                total_style_loss_val += cfg.lambda_style * loss.style.item()

                # Calculate the PSNR of the identity reconstructions
                x_val = denormalize_image(
                    x_val,
                    NORMALIZATION_MEAN[cfg.dataset],
                    NORMALIZATION_STD[cfg.dataset],
                )
                x_recon_val = denormalize_image(
                    x_recon_val,
                    NORMALIZATION_MEAN[cfg.dataset],
                    NORMALIZATION_STD[cfg.dataset],
                )
                average_psnr_val += compute_psnr(x_val, x_recon_val).item()

                # Update the progress bar
                pbar_val.update(1)

        # Average the loss across the batches and log the result
        total_loss_val /= nr_batches_per_epoch_val
        total_identity_loss_val /= nr_batches_per_epoch_val
        total_consistency_loss_val /= nr_batches_per_epoch_val
        total_anatomical_loss_val /= nr_batches_per_epoch_val
        total_style_loss_val /= nr_batches_per_epoch_val
        average_psnr_val /= nr_batches_per_epoch_val

        accelerator.log({"val/total_loss": total_loss_val}, step=epoch + 1)
        accelerator.log({"val/identity_loss": total_identity_loss_val}, step=epoch + 1)
        accelerator.log(
            {"val/consistency_loss": total_consistency_loss_val}, step=epoch + 1
        )
        accelerator.log(
            {"val/anatomical_loss": total_anatomical_loss_val}, step=epoch + 1
        )
        accelerator.log({"val/style_loss": total_style_loss_val}, step=epoch + 1)
        accelerator.log({"val/average_psnr": average_psnr_val}, step=epoch + 1)

        accelerator.print(f"Val Loss [{epoch + 1}/{cfg.epochs}]: {total_loss_val}")
        accelerator.print(
            f"Val Identity Loss [{epoch + 1}/{cfg.epochs}]: {total_identity_loss_val}"
        )
        accelerator.print(
            f"Val Consistency Loss [{epoch + 1}/{cfg.epochs}]: "
            f"{total_consistency_loss_val}"
        )
        accelerator.print(
            f"Val anatomical Loss [{epoch + 1}/{cfg.epochs}]: "
            f"{total_anatomical_loss_val}"
        )
        accelerator.print(
            f"Val Style Loss [{epoch + 1}/{cfg.epochs}]: {total_style_loss_val}"
        )
        accelerator.print(
            f"Val Average PSNR [{epoch + 1}/{cfg.epochs}]: {average_psnr_val}"
        )

        # Save the model
        best_val_loss = save_model(
            accelerator=accelerator,
            model=model,
            val_loss=total_loss_val,
            best_val_loss=best_val_loss,
            save_name=run_name,
            save_path=output_path,
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
    parser = argparse.ArgumentParser(description="Pretraining.")

    # Pretraining configurations (optimizer and lr scheduler can not be adjusted as of
    # now via command line arguments)
    parser.add_argument("--dataset", type=str, help="Dataset to use for pretraining.")
    parser.add_argument(
        "--data_path", type=str, help="Path to where the dataset shall be or is stored."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to where the checkpoints shall be stored.",
    )
    parser.add_argument(
        "--backbone", type=str, help="Backbone model to use for the Stylizing ViT."
    )
    parser.add_argument(
        "--input_size", type=int, help="Which input size to use for the model."
    )

    parser.add_argument("--epochs", type=int, help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, help="Batch size for training.")
    parser.add_argument(
        "--max_gpu_batch_size",
        type=int,
        help="Maximum batch size that fits on the GPU.",
    )
    parser.add_argument(
        "--lambda_identity", type=float, help="Weight for the identity loss."
    )
    parser.add_argument(
        "--lambda_consistency", type=float, help="Weight for the consistency loss."
    )
    parser.add_argument(
        "--lambda_anatomical", type=float, help="Weight for the anatomical loss."
    )
    parser.add_argument("--lambda_style", type=float, help="Weight for the style loss.")

    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--use_cuda", type=bool, help="Use CUDA for training")
    parser.add_argument(
        "--num_workers", type=int, help="Number of workers for the dataloader"
    )

    args = parser.parse_args()

    # Initialize the configurations
    cfg = PretrainingConfiguration()

    # Overwrite the default configs with the given command line arguments
    for arg in vars(args):
        setattr(cfg, arg, getattr(args, arg))

    # If the batch size is too big we use gradient accumulation
    if cfg.batch_size > cfg.max_gpu_batch_size:
        cfg.gradient_accumulation_steps = args.batch_size // cfg.max_gpu_batch_size
        cfg.effective_batch_size = cfg.max_gpu_batch_size
    else:
        cfg.gradient_accumulation_steps = 1
        cfg.effective_batch_size = cfg.batch_size

    # Create the configurations
    print(f"Initialize the configurations: {cfg}")

    # Start the pretraining
    main(cfg=cfg)
