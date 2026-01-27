"""
xAILab Bamberg
University of Bamberg

@description:
Dataset class for the aggregated Epithelium-Stroma dataset, consisting of the publicly
available NKI, VGH, and IHC datasets.
The NKI and VGH datasets comprise H&E stained breast cancer tissue images,
while the IHC dataset consists of IHC-stained colorectal cancer tissue images.

@references:
- Paper:
    - NKI and VGH:
        - Andrew H. Beck, et al. "Systematic analysis of breast cancer morphology
            uncovers stromal features associated with survival". Science Translational
            Medicine: 3(108). 2011.
    - IHC:
        - Nina Linder, et al. "Identification of tumor epithelium and stroma in tissue
            microarrays using texture analysis". Diagnostic Pathology: 7. 2012.
- Data:
    - Chenxin Li et al. "Domain generalization on medical imaging classification
        using episodic training with task augmentation". Computers in Biology and
        Medicine: 141. 2022.
        The dataset is available on Google Drive (https://github.com/chenxinli001/Task-Aug):
        https://drive.google.com/file/d/1YeFcs2yeJmxCFI3puQKUZuac13La1BpW/view?usp=sharing
"""

import os
from torchvision.datasets import ImageFolder
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import ConcatDataset
from typing import Any, Callable, Optional, Tuple


class EpitheliumStroma(VisionDataset):
    """
    Dataset class for the aggregated Epithelium-Stroma dataset,
    consisting of the publicly available NKI, VGH, and IHC datasets.
    The NKI and VGH datasets comprise H&E stained breast cancer tissue images,
    while the IHC dataset consists of IHC-stained colorectal cancer tissue images.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """
        Initialize the Camelyon17-WILDS dataset.

        Args:
            root (str): Root directory of the dataset.
            split (str): Split of the dataset (train, val, test).
            transform (torchvision.transforms, optional): Transformations to apply to
                the dataset. Defaults to None.
            target_transform (torchvision.transforms, optional): Transformations to
                apply to the target. Defaults to None.
        """
        super(EpitheliumStroma, self).__init__(
            root_dir, transform=transform, target_transform=target_transform
        )
        self.transform = transform
        self.target_transform = target_transform

        # Load the image paths and labels
        self.dataset = self.load_data(root_dir, split)

    def load_data(self, root_dir: str, split: str) -> VisionDataset:
        """
        Load the image paths and labels for the specified dataset type.

        Args:
            root_dir (str): Root directory of the dataset.
            split (str): Dataset split (train, val, test) which determines which
                dataset to load of (NKI, VGH, IHC).

        Returns:
            Dataset: Combined dataset for the specified split.
        """

        # Determine the dataset to load for the current split
        if split == "train":
            dataset_type = "NKI"

        elif split == "val":
            dataset_type = "VGH"

        elif split == "test":
            dataset_type = "IHC"

        else:
            raise ValueError(f"Invalid split: {split}")

        original_train_dir = os.path.join(root_dir, dataset_type, "train")
        original_test_dir = os.path.join(root_dir, dataset_type, "test")

        original_train_dataset = ImageFolder(original_train_dir)
        original_test_dataset = ImageFolder(original_test_dir)

        return ConcatDataset([original_train_dataset, original_test_dataset])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img, target = self.dataset[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
