"""
xAILab Bamberg
University of Bamberg

@description:
Dataset factory.
"""

# Import packages
import os
from torchvision.datasets.vision import VisionDataset
from typing import Callable, Optional

# Import own scripts
from experiments.data.camelyon import Camelyon17WILDS
from experiments.data.epistr import EpitheliumStroma
from experiments.data.fitzpatrick import Fitzpatrick17k


class CustomDataset(VisionDataset):
    def __init__(
        self,
        dataset_name: str,
        data_path: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        **kwargs,
    ):

        # Load the dataset
        if dataset_name.lower() == "camelyon17wilds":
            # Create the root directory to the data if it does not exist
            root_dir = create_dataset_directory(data_path, dataset_name)

            # Load the data with the default dataset splits
            self.dataset = Camelyon17WILDS(
                root_dir=root_dir,
                split=split,
                transform=transform,
                target_transform=target_transform,
            )

        elif dataset_name.lower() == "epistr":
            # Create the root directory to the data if it does not exist
            root_dir = create_dataset_directory(data_path, dataset_name)

            # Load the data with dataset splits across datasets (NKI, VGH, IHC)
            # Train: NKI / Val: VGH / Test: IHC
            self.dataset = EpitheliumStroma(
                root_dir=root_dir,
                split=split,
                transform=transform,
                target_transform=target_transform,
            )

        elif "fitzpatrick17k" in dataset_name.lower():
            # Create the root directory to the data if it does not exist
            root_dir = create_dataset_directory(data_path, "fitzpatrick17k")

            # Load the data with dataset splits across skin tones
            if dataset_name.lower() == "fitzpatrick17k-12_34_56":
                # Train: 1,2 / Val: 3,4 / Test: 5,6
                self.dataset = Fitzpatrick17k(
                    root_dir=root_dir,
                    split=split,
                    transform=transform,
                    target_transform=target_transform,
                    train_categories=(1, 2),
                    val_categories=(3, 4),
                    test_categories=(5, 6),
                )
            else:
                # Default: Train: 1,2 / Val: 3,4 / Test: 5,6
                self.dataset = Fitzpatrick17k(
                    root_dir=root_dir,
                    split=split,
                    transform=transform,
                    target_transform=target_transform,
                    train_categories=kwargs.get("train_categories", (1, 2)),
                    val_categories=kwargs.get("val_categories", (3, 4)),
                    test_categories=kwargs.get("test_categories", (5, 6)),
                )

        else:
            raise ValueError(f"Dataset {dataset_name} is not supported.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def create_dataset_directory(data_path: str, dataset_name: str) -> str:
    """
    Create a directory for the dataset.

    Args:
        data_path (str): Path to store/load the dataset.
        dataset_name (str): Name of the dataset.

    Returns:
        str: Path to the dataset directory.
    """

    # Create the root directory to the data if it does not exist
    root_dir = os.path.join(data_path, dataset_name)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    return root_dir


def create_dataset(
    dataset_name: str,
    data_path: str,
    split: str = "train",
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    **kwargs,
) -> CustomDataset:
    """
    Create a dataset with specified split using the CustomDataset class.

    Args:
        dataset_name (str): Name of the dataset.
        data_path (str): Path to store/load the dataset.
        split (str): Which split to load ('train', 'val', 'test').
        transform (callable, optional): A transform to apply to the data.
        target_transform (callable, optional): A transform to apply to the target.
        kwargs: Additional keyword arguments for the dataset.

    Returns:
        dataset: The requested dataset split or all splits as a dictionary.
    """

    return CustomDataset(
        dataset_name, data_path, split, transform, target_transform, **kwargs
    )
