#!/bin/bash

# xAILab Bamberg
# University of Bamberg
#
# @description:
# Bash script for training.
#
# Usage: bash train.sh

# Activate conda environment if necessary
if [[ "$CONDA_DEFAULT_ENV" != "stylizing-vit" ]]; then
  source activate stylizing-vit
fi

# ======================================================================================================================
# 0. User Configuration
# ======================================================================================================================
# Paths
# ----------------------------------------------------------------------------------------------------------------------
# The project path is automatically determined from the script location but can be overridden.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_PATH="$(dirname "$(dirname "$SCRIPT_DIR")")"

PROJECT_PATH="${PROJECT_PATH:-$DEFAULT_PROJECT_PATH}"
DATA_PATH="${DATA_PATH:-/mnt/data/stylizing-vit/data}"
# Checkpoint and result paths - you can change these base paths
CHECKPOINT_BASE_PATH="${CHECKPOINT_BASE_PATH:-/mnt/data/stylizing-vit/checkpoints}"

# Experiment Specific Paths
INPUT_PATH="${CHECKPOINT_BASE_PATH}/pretrain"
OUTPUT_PATH="${CHECKPOINT_BASE_PATH}/train"

# Device Configuration
# ----------------------------------------------------------------------------------------------------------------------
# Specify GPU devices (e.g., "0" or "0,1") or leave empty for CPU.
# If CUDA_VISIBLE_DEVICES is set, it will be prioritized.
DEVICE_IDS="${CUDA_VISIBLE_DEVICES:-0}"

# ======================================================================================================================
# 1. Environment Setup & Device Selection
# ======================================================================================================================
# Setup project path for imports
if [[ ":$PYTHONPATH:" != *":$PROJECT_PATH:"* ]]; then
  export PYTHONPATH="$PYTHONPATH:$PROJECT_PATH"
  echo "Project path added to PYTHONPATH: $PROJECT_PATH"
else
  echo "Project path already in PYTHONPATH."
fi

# Change to experiments directory for execution
cd "${PROJECT_PATH}/experiments" || exit

# Configure Device
if [[ -z "$DEVICE_IDS" ]]; then
    echo "No GPU devices specified. Using CPU."
    USE_CUDA="False"
    ACCELERATE_ARGS="--cpu"
else
    echo "Using GPU devices: $DEVICE_IDS"
    export CUDA_VISIBLE_DEVICES="$DEVICE_IDS"
    USE_CUDA="True"
    # Using single_gpu.yaml as default config for GPU execution
    ACCELERATE_ARGS="--config_file ${PROJECT_PATH}/experiments/configs/accelerate/single_gpu.yaml"
fi

# ======================================================================================================================
# 2. Experiment Configuration
# ======================================================================================================================
datasets=("camelyon17wilds" "epistr" "fitzpatrick17k-12_34_56")
classifiers=("densenet121")
backbones=("tiny" "small" "base")
input_size=224
augmentations=("styleTransfer" "styleTransfer_resizedCrop" "styleTransfer_grayScale" "styleTransfer_colorJitter" "styleTransfer_trivialAugment" "styleTransfer_targetedAugment")
portion_augmented_samples=(0.1 0.33 0.5 1.0)
epochs=100
early_stopping=10
batch_size=64
seeds=(265017005 145288287 206368124)
num_workers=4

# ======================================================================================================================
# 3. Execution
# ======================================================================================================================
for dataset in "${datasets[@]}"; do
  for classifier in "${classifiers[@]}"; do
    for portion in "${portion_augmented_samples[@]}"; do
      for seed in "${seeds[@]}"; do
        for augmentation in "${augmentations[@]}"; do
          for backbone in "${backbones[@]}"; do
            accelerate launch $ACCELERATE_ARGS ./train.py \
              --dataset $dataset \
              --data_path $DATA_PATH \
              --input_path $INPUT_PATH \
              --output_path $OUTPUT_PATH \
              --classifier $classifier \
              --backbone $backbone \
              --pretrained_stylizing_vit_weights $dataset \
              --input_size $input_size \
              --augmentation $augmentation \
              --portion_augmented_samples $portion \
              --epochs $epochs \
              --early_stopping $early_stopping \
              --batch_size $batch_size \
              --seed $seed \
              --use_cuda $USE_CUDA \
              --num_workers $num_workers
          done
        done
      done
    done
  done
done