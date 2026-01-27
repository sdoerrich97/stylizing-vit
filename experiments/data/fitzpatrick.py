"""
xAILab Bamberg
University of Bamberg

@description:
Dataset class for the Fitzpatrick17k dataset.

@references:
- Paper:
    - Matthew Groh, et al. "Evaluating deep neural networks trained on clinical images in
        dermatology with the fitzpatrick 17k dataset". In Proceedings of the IEEE/CVF Conference
        on Computer Vision and Pattern Recognition (pp. 1820-1828). 2021.
    - Matthew Groh, et al. "Towards transparency in dermatology image datasets with skin tone
        annotations by experts, crowds, and an algorithm". Proceedings of the ACM on Human-Computer
        Interaction (6):CSCW1, 1-26. 2022.
- Data: https://github.com/mattgroh/fitzpatrick17k
"""

import os
import pandas as pd
from torchvision.datasets.vision import VisionDataset
from PIL import Image
from typing import Any, Callable, Optional, Tuple
from sklearn.model_selection import train_test_split


class Fitzpatrick17k(VisionDataset):
    """
    Dataset class for the Fitzpatrick17k dataset.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the Fitzpatrick17k dataset.

        Args:
            root (str): Root directory of the dataset.
            split (str): Split of the dataset (train, val, test).
            transform (torchvision.transforms, optional): Transformations to apply to
            the dataset. Defaults to None.
            target_transform (torchvision.transforms, optional): Transformations to
            apply to the target. Defaults to None.
            kwargs: Additional keyword arguments.
                -  train_categories (tuple): Categories to use for training.
                -  val_categories (tuple): Categories to use for validation.
                -  test_categories (tuple): Categories to use for testing.
                -  num_classes (int): Number of classes for the classification task.
        """
        super(Fitzpatrick17k, self).__init__(
            root_dir, transform=transform, target_transform=target_transform
        )
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self.all_categories = (
            1,
            2,
            3,
            4,
            5,
            6,
        )  # All skin categories of the Fitzpatrick17k dataset)
        self.all_classification_tasks = [
            "label",
            "nine_partition_label",
            "three_partition_label",
        ]  # All classification tasks of the Fitzpatrick17k dataset

        assert split in [
            "train",
            "val",
            "test",
        ], "Split must be either 'train', 'val', or 'test'."
        assert (
            kwargs.get("train_categories", None) is not None
        ), "Train categories must be specified."
        assert (
            kwargs.get("val_categories", None) is not None
        ), "Validation categories must be specified."
        assert (
            kwargs.get("test_categories", None) is not None
        ), "Test categories must be specified."

        self.train_categories = self._convert_to_tuple(
            kwargs.get("train_categories", (1, 2))
        )  # Default value for train categories
        self.val_categories = self._convert_to_tuple(
            kwargs.get("val_categories", (3, 4))
        )  # Default value for val categories
        self.test_categories = self._convert_to_tuple(
            kwargs.get("test_categories", (5, 6))
        )  # Default value for test categories
        self.num_classes = kwargs.get(
            "num_classes", 3
        )  # Default classification task with 3 classes

        # Set the classification task based on the number of classes
        if self.num_classes == 3:
            self.classification_task = "three_partition_label"
        elif self.num_classes == 9:
            self.classification_task = "nine_partition_label"
        else:
            self.classification_task = "label"

        # Load the metadata
        metadata_path = os.path.join(root_dir, "fitzpatrick17k.csv")
        self.metadata = pd.read_csv(metadata_path)

        # Filter out images with skin category -1
        self.metadata = self.metadata[self.metadata["fitzpatrick_scale"] != -1]

        # Create a mapping from labels to integers
        self.label_to_int = {
            label: idx
            for idx, label in enumerate(
                self.metadata[self.classification_task].unique()
            )
        }

        # Split the samples and apply stratified split if a skin category is listed in
        # multiple splits
        self.metadata = self.apply_stratified_split()

        # Load the images and labels
        self.images = self.metadata["md5hash"].tolist()
        self.labels = [
            self.label_to_int[label]
            for label in self.metadata[self.classification_task].tolist()
        ]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img_path = os.path.join(self.root, "data", f"{self.images[idx]}.jpg")
        img = Image.open(img_path).convert("RGB")
        target = self.labels[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _convert_to_tuple(self, categories):
        """
        Convert categories to a tuple if they are given as an int, list, or tuple.
        Raise an error if they are given as something else.

        Args:
            categories (int, list, tuple): Categories to convert.

        Returns:
            tuple: Converted categories.
        """
        if isinstance(categories, int):
            return (categories,)
        elif isinstance(categories, (list, tuple)):
            return tuple(categories)
        else:
            raise TypeError("Categories must be an int, list, or tuple.")

    def apply_stratified_split(self) -> pd.DataFrame:
        """
        Apply a stratified split to the metadata if a skin category is listed in
        multiple splits.

        Returns:
            pd.DataFrame: The stratified split metadata.
        """
        # Extract labels for stratification
        labels = self.metadata[self.classification_task].tolist()

        # Determine which categories are in multiple splits
        train_val_categories = set(self.train_categories).intersection(
            self.val_categories
        )
        train_test_categories = set(self.train_categories).intersection(
            self.test_categories
        )
        val_test_categories = set(self.val_categories).intersection(
            self.test_categories
        )
        all_splits_categories = train_val_categories.intersection(self.test_categories)

        if all_splits_categories:
            # Split the metadata into training+validation and test
            train_val_metadata, test_metadata = train_test_split(
                self.metadata,
                test_size=0.2,
                random_state=0,
                stratify=labels,
            )

            # Extract labels for the training+validation set
            train_val_labels = train_val_metadata[self.classification_task].tolist()

            # Split the training+validation set into training and validation
            train_metadata, val_metadata = train_test_split(
                train_val_metadata,
                test_size=0.1,
                random_state=0,
                stratify=train_val_labels,
            )

        elif train_val_categories:
            # Split the metadata into training and validation
            train_metadata, val_metadata = train_test_split(
                self.metadata,
                test_size=0.2,
                random_state=0,
                stratify=labels,
            )
            test_metadata = self.metadata[
                self.metadata["fitzpatrick_scale"].isin(self.test_categories)
            ]

        elif train_test_categories:
            # Split the metadata into training and test
            train_metadata, test_metadata = train_test_split(
                self.metadata,
                test_size=0.2,
                random_state=0,
                stratify=labels,
            )
            val_metadata = self.metadata[
                self.metadata["fitzpatrick_scale"].isin(self.val_categories)
            ]

        elif val_test_categories:
            # Split the metadata into validation and test
            val_metadata, test_metadata = train_test_split(
                self.metadata,
                test_size=0.2,
                random_state=0,
                stratify=labels,
            )
            train_metadata = self.metadata[
                self.metadata["fitzpatrick_scale"].isin(self.train_categories)
            ]

        else:
            # Split the metadata into training, validation, and test for no overlapping
            # categories
            train_metadata = self.metadata[
                self.metadata["fitzpatrick_scale"].isin(self.train_categories)
            ]
            val_metadata = self.metadata[
                self.metadata["fitzpatrick_scale"].isin(self.val_categories)
            ]
            test_metadata = self.metadata[
                self.metadata["fitzpatrick_scale"].isin(self.test_categories)
            ]

        # Return the requested split
        if self.split == "train":
            return train_metadata
        elif self.split == "val":
            return val_metadata
        else:
            return test_metadata
