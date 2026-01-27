"""
xAILab Bamberg
University of Bamberg

@description:
Dataset class for the Camelyon17-WILDS dataset.

@references:
- Paper:
    - WILDS: Pang Wei Koh, et al. "WILDS: A Benchmark of in-the-Wild Distribution
        Shifts". International Conference on Machine Learning (ICML). 2021.
    - Camelyon17: Peter Bandi, et al. "From detection of individual metastases to
        classification of lymph node status at the patient level: the CAMELYON17
        challenge". IEEE Transactions on Medical Imaging. 2018.
- Data: https://worksheets.codalab.org/worksheets/0xb44731cc8e8a4265a20146c3887b6b90
- Code: https://github.com/p-lambda/wilds/tree/main
"""

from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Optional, Tuple
from wilds import get_dataset


class Camelyon17WILDS(VisionDataset):
    """
    Dataset class for the Camelyon17-WILDS dataset.
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
        super(Camelyon17WILDS, self).__init__(
            root_dir, transform=transform, target_transform=target_transform
        )
        self.transform = transform
        self.target_transform = target_transform

        full_dataset = get_dataset(
            dataset="camelyon17", download=True, root_dir=root_dir
        )
        self.dataset = full_dataset.get_subset(split)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img, target, _ = self.dataset[idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
