"""
xAILab Bamberg
University of Bamberg

@description:
Factory methods for creating StylizingViT models and loading weights.
"""

# Import packages
import os
import torch
from typing import Optional, Union
from huggingface_hub import hf_hub_download

# Import model class
from .model.stylizing_vit import StylizingViT

# Mapping from (backbone, dataset) to the Hugging Face Model Repository ID
MODEL_REPOS = {
    # base models
    ("base", "camelyon17wilds"): "sdoerrich97/stylizing_vit_base_camelyon17wilds",
    ("base", "cholec80"): "sdoerrich97/stylizing_vit_base_cholec80",
    ("base", "ddi"): "sdoerrich97/stylizing_vit_base_ddi_12_34_56",
    ("base", "ddi_12_34_56"): "sdoerrich97/stylizing_vit_base_ddi_12_34_56",
    ("base", "ddi_65_43_21"): "sdoerrich97/stylizing_vit_base_ddi_65_43_21",
    ("base", "epistr"): "sdoerrich97/stylizing_vit_base_epistr",
    ("base", "fitzpatrick17k"): "sdoerrich97/stylizing_vit_base_fitzpatrick17k_12_34_56",
    ("base", "fitzpatrick17k_12_34_56"): "sdoerrich97/stylizing_vit_base_fitzpatrick17k_12_34_56",
    ("base", "fitzpatrick17k_65_43_21"): "sdoerrich97/stylizing_vit_base_fitzpatrick17k_65_43_21",

    # small models
    ("small", "camelyon17wilds"): "sdoerrich97/stylizing_vit_small_camelyon17wilds",
    ("small", "cholec80"): "sdoerrich97/stylizing_vit_small_cholec80",
    ("small", "ddi"): "sdoerrich97/stylizing_vit_small_ddi_12_34_56",
    ("small", "ddi_12_34_56"): "sdoerrich97/stylizing_vit_small_ddi_12_34_56",
    ("small", "ddi_65_43_21"): "sdoerrich97/stylizing_vit_small_ddi_65_43_21",
    ("small", "epistr"): "sdoerrich97/stylizing_vit_small_epistr",
    ("small", "fitzpatrick17k"): "sdoerrich97/stylizing_vit_small_fitzpatrick17k_12_34_56",
    ("small", "fitzpatrick17k_12_34_56"): "sdoerrich97/stylizing_vit_small_fitzpatrick17k_12_34_56",
    ("small", "fitzpatrick17k_65_43_21"): "sdoerrich97/stylizing_vit_small_fitzpatrick17k_65_43_21",

    # tiny models
    ("tiny", "camelyon17wilds"): "sdoerrich97/stylizing_vit_tiny_camelyon17wilds",
    ("tiny", "cholec80"): "sdoerrich97/stylizing_vit_tiny_cholec80",
    ("tiny", "ddi"): "sdoerrich97/stylizing_vit_tiny_ddi_12_34_56",
    ("tiny", "ddi_12_34_56"): "sdoerrich97/stylizing_vit_tiny_ddi_12_34_56",
    ("tiny", "ddi_65_43_21"): "sdoerrich97/stylizing_vit_tiny_ddi_65_43_21",
    ("tiny", "epistr"): "sdoerrich97/stylizing_vit_tiny_epistr",
    ("tiny", "fitzpatrick17k"): "sdoerrich97/stylizing_vit_tiny_fitzpatrick17k_12_34_56",
    ("tiny", "fitzpatrick17k_12_34_56"): "sdoerrich97/stylizing_vit_tiny_fitzpatrick17k_12_34_56",
    ("tiny", "fitzpatrick17k_65_43_21"): "sdoerrich97/stylizing_vit_tiny_fitzpatrick17k_65_43_21",
}


def load_pretrained_weights(
    model: StylizingViT,
    backbone: str,
    dataset: str,
    map_location: Union[str, torch.device] = "cpu"
):
    """
    Load pretrained weights into the model.

    Args:
        model (StylizingViT): The StylizingViT model instance.
        backbone (str): Backbone name (e.g. "base").
        dataset (str): Dataset name (e.g. "camelyon17wilds") or path to local weights.
        map_location (str or torch.device): Device to map weights to. Defaults to "cpu".
    """

    # Check if dataset is a local path
    if os.path.isfile(dataset):
        checkpoint_path = dataset
        print(f"Loading local weights from {checkpoint_path}...")
    else:
        # Determine the repo_id
        key = (backbone, dataset)
        if key not in MODEL_REPOS:
            # Fallback: assume 'dataset' is actually a full HF repo ID if not found in map
            repo_id = dataset
        else:
            repo_id = MODEL_REPOS[key]

        # print(f"Targeting Hugging Face repository: {repo_id}")

        # 1. Download config.json to register the download statistic
        # This is required because we are not using a library with built-in tracking.
        try:
            _ = hf_hub_download(repo_id=repo_id, filename="config.json")
        except Exception:
            # If config.json is missing, just proceed. 
            # It implies no download stats will be tracked, but code shouldn't break.
            pass

        # 2. Download the model weights
        try:
            print(f"Downloading weights for '{dataset}' (backbone: {backbone})...")
            # We prefer .safetensors, but could fallback to .pth if needed
            checkpoint_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        except Exception as e:
            raise ValueError(f"Could not download weights for backbone '{backbone}' "
                             f"and dataset '{dataset}' from repo '{repo_id}'. "
                             f"Ensure the repository exists and contains 'model.safetensors'. "
                             f"Error: {e}")

    # Load state dict
    if checkpoint_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError(
                "The 'safetensors' library is required to load .safetensors files. "
                "Please install it via `pip install safetensors`."
            )

        state_dict = load_file(checkpoint_path)
        # print("Loaded weights using safetensors.")
    else:
        state_dict = torch.load(checkpoint_path, map_location=map_location,
                                weights_only=False)
        # print("Loaded weights using torch.load.")

    # Handle possible state_dict wrapping (e.g. "model", "state_dict" keys)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Remove module. prefix if present (DDP)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict

    # Remove vgg_encoder weights if the model doesn't have a vgg_encoder
    # (e.g. when initialized with train=False)
    if not hasattr(model, "vgg_encoder"):
        keys_to_remove = [k for k in state_dict.keys() if k.startswith("vgg_encoder.")]
        if keys_to_remove:
            # print(f"Removing {len(keys_to_remove)} vgg_encoder keys from state_dict "
            #       "(model initialized with train=False).")
            for k in keys_to_remove:
                del state_dict[k]

    # Load into model
    model.load_state_dict(state_dict, strict=True)
    # print(f"Loaded weights successfully: {msg}")


def create_model(
    backbone: str = "base",
    train: bool = False,
    weights: Optional[str] = None,
    pretrained: Union[bool, str] = False,
    **kwargs
) -> StylizingViT:
    """
    Create a StylizingViT model.

    Args:
        backbone (str): The backbone architecture of the model ("tiny", "small",
            "base", etc.). Defaults to "base".
        train (bool): Whether the model is in training mode. Defaults to False.
        weights (str, optional): Specific weights to load. Can be a dataset name
            (e.g. "camelyon17wilds") or a local file path.
        pretrained (bool or str): If True, loads default weights ("camelyon17wilds").
            If string, serves as an alias for `weights`.
        **kwargs: Additional parameters passed to StylizingViT constructor.

    Returns:
        StylizingViT: The created model.
    """

    model = StylizingViT(backbone=backbone, train=train, **kwargs)

    dataset_to_load = None
    if weights is not None:
        dataset_to_load = weights
    elif isinstance(pretrained, str):
        dataset_to_load = pretrained
    elif pretrained is True:
        # Default dataset
        dataset_to_load = "camelyon17wilds"

    if dataset_to_load:
        load_pretrained_weights(
            model,
            backbone=backbone,
            dataset=dataset_to_load,
        )

    return model
