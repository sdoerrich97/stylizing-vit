"""
xAILab Bamberg
University of Bamberg

@description:
Style transfer.
"""

# Import packages
import os
import time
import json
import torch
import argparse
from torch import Generator
from accelerate import Accelerator
from accelerate.utils import tqdm
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from typing import Tuple, Dict, Any
import itertools
import numpy as np
from itertools import zip_longest

# Import own scripts
from stylizing_vit.model import StylizingViT
from experiments.data import (
    create_dataset,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    DATASET_SPLITS,
)
from experiments.utils.reproducibility import random_seed, worker_seed
from experiments.utils.training import calculate_passed_time
from experiments.utils.image_operations import denormalize_image
from experiments.utils.preprocessing import ResizeWhileRetainAspectRatio
from experiments.metrics import (
    prepare_generation_evaluation,
    compute_fid,
    compute_lpips,
    compute_artfid,
)
from experiments.configs.experiments import StyleTransferConfig


def prepare_the_dataloader(
    cfg: StyleTransferConfig, g: Generator, **kwargs: Any
) -> Dict[str, Tuple[DataLoader, DataLoader]]:
    """
    Prepare the dataloader for the available splits of the dataset.

    Args:
        cfg (StyleTransferConfig): The style transfer configuration object containing
        dataset settings.
        g (Generator): The random number generator for reproducibility.
        **kwargs (Any): Additional keyword arguments to pass to the dataset creation
        function.

    Returns:
        Dict[str, Tuple[DataLoader, DataLoader]]: A dictionary where keys are split
        combinations (e.g., 'train-train') and values are tuples of dataloaders
        for the two splits.
    """

    # Create the data loading transforms for the training and validation datasets.
    transform = v2.Compose(
        [
            ResizeWhileRetainAspectRatio(size=cfg.input_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=NORMALIZATION_MEAN[cfg.dataset], std=NORMALIZATION_STD[cfg.dataset]
            ),
        ]
    )

    # Get the available dataset splits
    available_datasetsplits = DATASET_SPLITS[cfg.dataset]

    # Now create a dataloader dictionary for each combination of splits
    # (e.g., train-train, train-val, val-train, val-val, ...)
    dataloaders = {}
    for split1, split2 in itertools.product(available_datasetsplits, repeat=2):
        if f"{split1}-{split2}" in dataloaders:
            continue

        dataset1 = create_dataset(
            dataset_name=cfg.dataset,
            data_path=cfg.data_path,
            split=split1,
            transform=transform,
            **kwargs,
        )
        dataset2 = create_dataset(
            dataset_name=cfg.dataset,
            data_path=cfg.data_path,
            split=split2,
            transform=transform,
            **kwargs,
        )

        dataloaders[f"{split1}-{split2}"] = (
            DataLoader(
                dataset=dataset1,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                worker_init_fn=worker_seed,
                generator=g,
            ),
            DataLoader(
                dataset=dataset2,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                worker_init_fn=worker_seed,
                generator=g,
            ),
        )

    # Return the dataloader
    return dataloaders


def main(cfg: StyleTransferConfig, **kwargs: Any) -> None:
    """
    Main function to evaluate the style mixing.

    Args:
        cfg (StyleTransferConfig): The style transfer configuration object.
        **kwargs (Any): Additional keyword arguments to pass to the dataset or other
        components.

    Returns:
        None
    """

    # Specify where to save the images
    output_path = os.path.join(
        cfg.output_path, cfg.dataset, f"stylizingVit-{cfg.backbone}"
    )
    json_path = os.path.join(cfg.output_path, f"evaluation_results_{cfg.dataset}.json")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    accelerator = Accelerator()

    # Specify the JSON file to store the evaluation results
    if not os.path.exists(json_path):
        header = {
            "_comment": "This JSON file contains metrics for style transfer methods. "
            "The structure is as follows:",
            "_structure": {
                "backbone": {
                    "split": {
                        "FID": "float",
                        "LPIPS": "float",
                        "ArtFID": "float",
                    }
                }
            },
            "data": {},
        }
        with open(json_path, "w") as json_file:
            json.dump(header, json_file, indent=4)

    # Load the existing content of the JSON file
    with open(json_path, "r") as json_file:
        metrics_data = json.load(json_file)

    # Ensure 'data' key exists in metrics_data
    if "data" not in metrics_data:
        metrics_data["data"] = {}

    # Set the random seed
    accelerator.print(f"Set the seed as: {cfg.seed}.")
    g = random_seed(seed_value=cfg.seed, use_cuda=cfg.use_cuda)

    # Load the model
    backbone_weights = os.path.join(
        cfg.input_path, f"{cfg.dataset}-stylizingVit-{cfg.backbone}.pth"
    )
    accelerator.print(
        f"Load the pretrained model {os.path.basename(backbone_weights)} "
        f"from {cfg.input_path}."
    )
    model = StylizingViT(backbone=cfg.backbone, train=False)
    model.load_state_dict(
        torch.load(backbone_weights, weights_only=True, map_location="cpu"),
        strict=False,
    )
    model.requires_grad_(False)
    model.eval()

    # Load the dataloaderss
    accelerator.print(
        f"Load the dataset {cfg.dataset} from {cfg.data_path} or download from the "
        f"internet."
    )
    dataloaders = prepare_the_dataloader(cfg=cfg, g=g, **kwargs)

    # Distribute to the specified device(s)
    accelerator.print("Distribute to the specified device(s)")
    model = accelerator.prepare(model)

    # Generate and store augmented images for each split
    accelerator.print(
        "Generate and store the augmented images as well as calculate the evaluation "
        "metrics for each split."
    )
    start_time = time.time()

    with torch.no_grad():
        for split, (dataloader1, dataloader2) in dataloaders.items():
            # Stop the time for the current split
            start_time_split = time.time()

            # Create the output path for the current split
            split_path = os.path.join(output_path, split)
            if not os.path.exists(split_path):
                os.makedirs(split_path)

            # Set the progress bar
            max_num_batches = min(len(dataloader1), len(dataloader2))
            pbar_train = tqdm(
                total=max_num_batches,
                bar_format="{l_bar}{bar}",
                ncols=80,
                initial=0,
                position=0,
                leave=False,
            )
            pbar_train.set_description(f"{split} split")

            # Initialize counters and lists to store images
            img_counter = 0
            batch_counter = 0
            (
                anatomy_images,
                style_images,
                stylized_images,
                labels_anatomy_images,
                labels_style_images,
            ) = ([], [], [], [], [])

            for i, (batch1, batch2) in enumerate(
                zip_longest(dataloader1, dataloader2, fillvalue=None)
            ):
                # Handle the case when one dataloader is exhausted
                if batch1 is None or batch2 is None or \
                   img_counter >= cfg.nr_images_to_save:
                    break

                # Get the images and labels
                X1, Y1 = batch1
                X2, Y2 = batch2

                # Handle the case if one batch is larger than the other
                if X1.size(0) != X2.size(0):
                    break

                # If both dataloaders produce the same samples,
                # e.g. for train-train or val-val, roll the style batch
                X2 = X2.clone().roll(shifts=-1, dims=0)

                # Send the batches to the same device as the model
                X1 = X1.to(accelerator.device)
                X2 = X2.to(accelerator.device)

                # Apply the style mixing to the original samples
                X_stylized = model(X1, X2)
                X_stylized_reverse = model(X2, X1)

                # Denormalize the images from value range [-1, 1]
                # to the value range [0, 1]
                X1 = denormalize_image(
                    X1, NORMALIZATION_MEAN[cfg.dataset], NORMALIZATION_STD[cfg.dataset]
                )
                X2 = denormalize_image(
                    X2, NORMALIZATION_MEAN[cfg.dataset], NORMALIZATION_STD[cfg.dataset]
                )
                X_stylized = denormalize_image(
                    X_stylized,
                    NORMALIZATION_MEAN[cfg.dataset],
                    NORMALIZATION_STD[cfg.dataset],
                )
                X_stylized_reverse = denormalize_image(
                    X_stylized_reverse,
                    NORMALIZATION_MEAN[cfg.dataset],
                    NORMALIZATION_STD[cfg.dataset],
                )

                # Save the images
                for j in range(X1.size(0)):
                    if img_counter < cfg.nr_images_to_save:
                        save_image(
                            X1[j],
                            os.path.join(
                                split_path,
                                f"{split}_sample{i * cfg.batch_size + j}_anatomy.png",
                            ),
                        )
                        save_image(
                            X2[j],
                            os.path.join(
                                split_path,
                                f"{split}_sample{i * cfg.batch_size + j}_style.png",
                            ),
                        )
                        save_image(
                            X_stylized[j],
                            os.path.join(
                                split_path,
                                f"{split}_sample{i * cfg.batch_size + j}_stylized.png",
                            ),
                        )
                        save_image(
                            X_stylized_reverse[j],
                            os.path.join(
                                split_path,
                                f"{split}_sample{i * cfg.batch_size + j}_stylized_reverse.png",
                            ),
                        )

                    anatomy_images.append(
                        X1[j]
                        .mul(255)
                        .add_(0.5)
                        .clamp_(0, 255)
                        .permute(1, 2, 0)
                        .to("cpu", torch.uint8)
                        .numpy()
                    )  # Save as numpy array in value range [0, 255]
                    style_images.append(
                        X2[j]
                        .mul(255)
                        .add_(0.5)
                        .clamp_(0, 255)
                        .permute(1, 2, 0)
                        .to("cpu", torch.uint8)
                        .numpy()
                    )  # Save as numpy array in value range [0, 255]
                    stylized_images.append(
                        X_stylized[j]
                        .mul(255)
                        .add_(0.5)
                        .clamp_(0, 255)
                        .permute(1, 2, 0)
                        .to("cpu", torch.uint8)
                        .numpy()
                    )  # Save as numpy array in value range [0, 255]
                    labels_anatomy_images.append(Y1[j].item())
                    labels_style_images.append(Y2[j].item())
                    img_counter += 1

                # Save the images in batches
                if len(anatomy_images) >= cfg.batch_size_to_save:
                    batch_counter += 1
                    path_anatomy_images = os.path.join(
                        split_path, f"{split}_anatomy_batch{batch_counter}.npz"
                    )
                    path_style_images = os.path.join(
                        split_path, f"{split}_style_batch{batch_counter}.npz"
                    )
                    path_stylized_images = os.path.join(
                        split_path, f"{split}_stylized_batch{batch_counter}.npz"
                    )

                    np.savez(
                        path_anatomy_images,
                        images=np.array(anatomy_images),
                        labels=np.array(labels_anatomy_images),
                    )
                    np.savez(
                        path_style_images,
                        images=np.array(style_images),
                        labels=np.array(labels_style_images),
                    )
                    np.savez(
                        path_stylized_images,
                        images=np.array(stylized_images),
                        labels=np.array(labels_anatomy_images),
                    )

                    # Clear the lists after saving
                    (
                        anatomy_images,
                        style_images,
                        stylized_images,
                        labels_anatomy_images,
                        labels_style_images,
                    ) = ([], [], [], [], [])

                # Update the progress bar
                pbar_train.update(1)

            # Save any remaining images
            if anatomy_images:
                batch_counter += 1
                path_anatomy_images = os.path.join(
                    split_path, f"{split}_anatomy_batch{batch_counter}.npz"
                )
                path_style_images = os.path.join(
                    split_path, f"{split}_style_batch{batch_counter}.npz"
                )
                path_stylized_images = os.path.join(
                    split_path, f"{split}_stylized_batch{batch_counter}.npz"
                )

                np.savez(
                    path_anatomy_images,
                    images=np.array(anatomy_images),
                    labels=np.array(labels_anatomy_images),
                )
                np.savez(
                    path_style_images,
                    images=np.array(style_images),
                    labels=np.array(labels_style_images),
                )
                np.savez(
                    path_stylized_images,
                    images=np.array(stylized_images),
                    labels=np.array(labels_anatomy_images),
                )

            # Calcualte the evaluation metrics for the current split
            accelerator.print(
                f"Calculate the evaluation metrics for the {split} split."
            )

            # Collect the paths to the saved images
            all_paths_anatomy_images = sorted(
                [
                    os.path.join(split_path, f)
                    for f in os.listdir(split_path)
                    if f.startswith(f"{split}_anatomy_batch") and f.endswith(".npz")
                ]
            )
            all_paths_style_images = sorted(
                [
                    os.path.join(split_path, f)
                    for f in os.listdir(split_path)
                    if f.startswith(f"{split}_style_batch") and f.endswith(".npz")
                ]
            )
            all_paths_stylized_images = sorted(
                [
                    os.path.join(split_path, f)
                    for f in os.listdir(split_path)
                    if f.startswith(f"{split}_stylized_batch") and f.endswith(".npz")
                ]
            )

            # Calculate FID between style and stylized images
            accelerator.print("\tCalculate FID between style and stylized images.")
            fid_model, fid_dataloader_original, fid_dataloader_generated = (
                prepare_generation_evaluation(
                    "fid",
                    all_paths_style_images,
                    all_paths_stylized_images,
                    cfg.batch_size,
                    num_workers=2,
                )
            )
            fid_model, fid_dataloader_original, fid_dataloader_generated = (
                accelerator.prepare(
                    fid_model, fid_dataloader_original, fid_dataloader_generated
                )
            )
            fid_value = compute_fid(
                fid_model, fid_dataloader_original, fid_dataloader_generated
            )

            # Calculate LPIPS between content and stylized images
            accelerator.print("\tCalculate LPIPS between content and stylized images.")
            lpips_model, lpips_dataloader_original, lpips_dataloader_generated = (
                prepare_generation_evaluation(
                    "lpips",
                    all_paths_anatomy_images,
                    all_paths_stylized_images,
                    cfg.batch_size,
                    num_workers=2,
                )
            )
            lpips_model, lpips_dataloader_original, lpips_dataloader_generated = (
                accelerator.prepare(
                    lpips_model, lpips_dataloader_original, lpips_dataloader_generated
                )
            )
            lpips_value = compute_lpips(
                lpips_model, lpips_dataloader_original, lpips_dataloader_generated
            )

            # Calculate ArtFID value
            accelerator.print("\tCalculate ArtFID value.")
            artfid_value = compute_artfid(fid_value, lpips_value)

            # Log the metrics
            metrics = {
                "FID": float(fid_value),
                "LPIPS": float(lpips_value),
                "ArtFID": float(artfid_value),
            }
            accelerator.print(
                f"Metrics for {split} split: FID: {fid_value}, LPIPS: {lpips_value}, \n"
                f"ArtFID: {artfid_value}"
            )

            # Update the JSON file with the current metrics
            if f"stylizingVit-{cfg.backbone}" not in metrics_data["data"]:
                metrics_data["data"][f"stylizingVit-{cfg.backbone}"] = {}

            metrics_data["data"][f"stylizingVit-{cfg.backbone}"][split] = metrics

            # Stop the time for the current split
            end_time_split = time.time()
            hours_split, minutes_split, seconds_split = calculate_passed_time(
                start_time_split, end_time_split
            )
            accelerator.print(
                "Elapsed time for split: {:0>2}:{:0>2}:{:05.2f}".format(
                    hours_split, minutes_split, seconds_split
                )
            )

    # Save the updated JSON file
    with open(json_path, "w") as json_file:
        json.dump(metrics_data, json_file, indent=4)

    # Stop the time for the whole process
    end_time = time.time()
    hours, minutes, seconds = calculate_passed_time(start_time, end_time)
    accelerator.print(
        "Elapsed time for generating and storing the images: "
        "{:0>2}:{:0>2}:{:05.2f}".format(
            hours, minutes, seconds
        )
    )


if __name__ == "__main__":
    # Read out the command line parameters.
    parser = argparse.ArgumentParser(description="Style transfer")
    parser.add_argument(
        "--dataset", type=str, default="camelyon17wilds", help="Dataset name"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/",
        help="Path to where the dataset is stored or shall be downloaded to",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="./checkpoints/pretrain",
        help="Path where the Stylizing ViT model checkpoints are stored",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./results/style_transfer",
        help="Path to where the images shall be stored",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="base",
        help="Name of the backbone to use for the StyleMixer",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="Height and width of the input images",
    )
    parser.add_argument(
        "--resize_crop",
        type=bool,
        default=False,
        help="Apply random resized crop to the style images",
    )
    parser.add_argument(
        "--nr_images_to_save", type=int, default=20, help="Number of images to save"
    )
    parser.add_argument(
        "--batch_size_to_save",
        type=int,
        default=10000,
        help="Number of images to save per batch as .npz files",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Number of images to process"
    )
    parser.add_argument(
        "--seed", type=int, default=265017005, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--use_cuda", type=bool, default=True, help="Use CUDA for training"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of workers for the dataloader",
    )
    args = parser.parse_args()

    # Initialize the configurations
    cfg = StyleTransferConfig()

    # Overwrite the default configs with the given command line arguments
    for arg in vars(args):
        setattr(cfg, arg, getattr(args, arg))

    # Create the configurations
    print(f"Initialize the configurations: {cfg}")

    # Start the evaluation
    main(cfg=cfg)
