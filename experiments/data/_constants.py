"""
xAILab Bamberg
University of Bamberg

@description:
Dataset Specs:
    - Camelyon17WILDS: default dataset splits
    - Epithelium-Stroma: dataset splits across datasets
        (Train: NKI / Val: VGH / Test: IHC)
    - Fitzpatrick17k: dataset splits across skin tones
        Train: 1,2 / Val: 3,4 / Test: 5,6
"""

# ============================================
# Mean and std calculated from the training sets
# ============================================
NORMALIZATION_MEAN = {
    "camelyon17wilds": (0.7440, 0.5895, 0.7214),
    "epistr": (0.7360, 0.5158, 0.8072),
    "fitzpatrick17k": (0.6219, 0.4917, 0.4478),
    "fitzpatrick17k-12_34_56": (0.6219, 0.4917, 0.4478)
}

NORMALIZATION_STD = {
    "camelyon17wilds": (0.1787, 0.2131, 0.1721),
    "epistr": (0.1948, 0.2434, 0.1438),
    "fitzpatrick17k": (0.2279, 0.1982, 0.1991),
    "fitzpatrick17k-12_34_56": (0.2279, 0.1982, 0.1991)
}

# ============================================
# Number of classes
# ============================================
NUM_CLASSES = {
    "camelyon17wilds": 2,
    "epistr": 2,
    "fitzpatrick17k": 3,
    "fitzpatrick17k-12_34_56": 3,
}

# ============================================
# Available dataset splits
# ============================================
DATASET_SPLITS = {
    # Camelyon17WILDS
    "camelyon17wilds": ["train", "val", "id_val", "test"],

    # Epithelium-Stroma
    "epistr": ["train", "val", "test"],

    # Fitzpatrick17k
    "fitzpatrick17k": ["train", "val", "test"],
    "fitzpatrick17k-12_34_56": ["train", "val", "test"],
}
