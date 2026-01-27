"""
xAILab Bamberg
University of Bamberg

@description:
Metrics to evaluate the quality of the stylized images.

@references:
StyleID: https://github.com/jiwoogit/StyleID
art-fid: https://github.com/matthias-wright/art-fid
HistoGAN: https://github.com/mahmoudnafifi/HistoGAN
"""

# Import packages
import timm
import glob
import numpy as np
import os
import torch
import lpips
from scipy import linalg
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from typing import Any, Callable, Optional

# Import own packages
from experiments.utils.reproducibility import worker_seed
from experiments.utils.preprocessing import ResizeWhileRetainAspectRatio


ALLOWED_IMAGE_EXTENSIONS = ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]


class NumpyDataset(Dataset):
    def __init__(
        self,
        npz_file_path,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """
        Initialize the NumpyDataset.

        Args:
            npz_file_path (str or list): Path to a single .npz file or a list of paths
            to multiple .npz files containing the images.
            transform (callable, optional): Optional transform to be applied on a
            sample.
            target_transform (callable, optional): Optional transform to be applied on
            the target.
        """
        super().__init__()

        # Convert single path to a list
        if isinstance(npz_file_path, str):
            npz_file_path = [npz_file_path]

        self.images = []
        self.targets = []

        # Load data from all .npz files
        for file_path in npz_file_path:
            data = np.load(file_path)
            self.images.append(data["images"])
            self.targets.append(data["labels"])

        # Concatenate data from all files
        self.images = np.concatenate(self.images, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx: int) -> Any:
        """
        Get a sample and its respective label from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: Image and target.
        """
        image, target = self.images[idx], self.targets[idx]

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target


def get_image_paths(path, sort=False):
    """
    Returns the paths of the images in the specified directory, filtered by allowed
    file extensions.

    Args:
        path (str): Path to image directory.
        sort (bool): Sort paths alphanumerically.

    Returns:
        list: List of image paths with allowed file extensions.
    """
    paths = []
    for extension in ALLOWED_IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(path, f"*.{extension}")))
    if sort:
        paths.sort()
    return paths


def compute_activation_statistics(model: torch.nn.Module, dataloader: DataLoader):
    """
    Computes the activation statistics used by the FID.

    Args:
        model (torch.nn.Module): Model for computing activations.
        dataloader (DataLoader): Dataloader for the images.

    Returns:
        tuple: mean of activations, covariance of activations
    """

    # Set the model to evaluation mode
    model.eval()

    # Compute the activations
    act = []
    with torch.no_grad():
        for batch in dataloader:
            # Check for labeled and unlabeled data
            if isinstance(batch, list) or isinstance(batch, tuple):
                batch = batch[0]

            features = model.forward_features(batch)
            features = model.forward_head(features, pre_logits=True)
            act.append(features.cpu().numpy())

    # Compute the statistics
    act = np.concatenate(act, axis=0)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)

    # Return the statistics
    return mu, sigma


def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Numpy implementation of the Frechet Distance.

    Args:
        mu1 (np.ndarray): Sample mean of activations of original images.
        mu2 (np.ndarray): Sample mean of activations of generated images.
        sigma1 (np.ndarray): Covariance matrix of activations of original images.
        sigma2 (np.ndarray): Covariance matrix of activations of generated images.
        eps (float): Epsilon for numerical stability.

    Returns:
        float: FID value.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def compute_fid(
    model: torch.nn.Module,
    dataloader_original: DataLoader,
    dataloader_generated: DataLoader,
):
    """
    Computes the FID for the given paths.

    Args:
        model (torch.nn.Module): Model for computing activations.
        dataloader_original (DataLoader): Dataloader for the original images.
        dataloader_generated (DataLoader): Dataloader for the generated images.

    Returns:
        float: FID value.
    """

    # Set the model to evaluation mode
    mu1, sigma1 = compute_activation_statistics(model, dataloader_original)
    mu2, sigma2 = compute_activation_statistics(model, dataloader_generated)

    fid_value = compute_frechet_distance(mu1, sigma1, mu2, sigma2)

    return fid_value


def compute_lpips(
    model: torch.nn.Module,
    dataloader_original: DataLoader,
    dataloader_generated: DataLoader,
):
    """
    Compute the LPIPS (Learned Perceptual Image Patch Similarity) between the original
    and generated images.

    Args:
        model (torch.nn.Module): Model for computing activations.
        dataloader_original (DataLoader): Dataloader for the original images.
        dataloader_generated (DataLoader): Dataloader for the generated images.

    Returns:
        float: The LPIPS value.
    """

    lpips_sum = None
    N = 0

    for i, (batch_original, batch_generated) in enumerate(
        zip(dataloader_original, dataloader_generated)
    ):
        with torch.no_grad():
            # Check for labeled and unlabeled data
            if isinstance(batch_original, list) or isinstance(batch_original, tuple):
                batch_original = batch_original[0]

            if isinstance(batch_generated, list) or isinstance(batch_generated, tuple):
                batch_generated = batch_generated[0]

            batch_lpips = model(batch_original, batch_generated)
            N += batch_original.shape[0]

            if i == 0:
                lpips_sum = torch.sum(batch_lpips)
            else:
                lpips_sum += torch.sum(batch_lpips)

    return lpips_sum.item() / N


def compute_artfid(fid_value: float, lpips_value: float) -> float:
    """
    Compute the ArtFID (Artistic Fréchet Inception Distance) between the original and
    generated images.

    Args:
        fid_value (float): The FID value.
        lpips_value (float): The LPIPS value.

    Returns:
        float: The ArtFID value.
    """

    # Compute ArtFID
    return (lpips_value + 1) * (fid_value + 1)


def prepare_generation_evaluation(
    metric: str,
    original_images_npz: str,
    generated_images_npz: str,
    batch_size: int,
    num_workers=2,
    generator=None,
):
    """
    Prepares the model and dataloaders for the given metric.

    Args:
        metric (str): The metric to prepare for ('fid', 'lpips').
        original_images_npz (str): Path to the original images .npz file.
        generated_images_npz (str): Path to the generated images .npz file.
        batch_size (int): Batch size for the dataloader.
        num_workers (int): Number of threads for data loading.
        generator (torch.Generator): Generator for seeding the dataloader.

    Returns:
        tuple: The model and dataloaders for the original and generated images.
    """

    # Initialize the model and the transforms (each metric requires different resizing)
    if metric == "fid":
        model = timm.create_model("inception_v3", pretrained=True)
        transform = v2.Compose(
            [
                ResizeWhileRetainAspectRatio(size=299),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    elif metric == "lpips":
        model = lpips.LPIPS(net="alex")
        transform = v2.Compose(
            [
                ResizeWhileRetainAspectRatio(size=512),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    elif metric == "ssim":
        model = None
        transform = v2.Compose(
            [
                ResizeWhileRetainAspectRatio(size=224),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Create datasets and dataloaders
    dataset_original = NumpyDataset(original_images_npz, transform=transform)
    dataloader_original = DataLoader(
        dataset_original,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        worker_init_fn=worker_seed,
        generator=generator,
    )

    dataset_generated = NumpyDataset(generated_images_npz, transform=transform)
    dataloader_generated = DataLoader(
        dataset_generated,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        worker_init_fn=worker_seed,
        generator=generator,
    )

    # Return the model and dataloaders
    return model, dataloader_original, dataloader_generated
