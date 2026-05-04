#!/bin/bash
# Training script for CIFAR models
# Usage examples:
#   ./run_training.sh --model vit --dataset cifar10
#   ./run_training.sh --model loopvit --dataset cifar100 --batch_size 64

set -e  # Exit on error

# Default arguments
DATA_ROOT="./data"
BATCH_SIZE=128
EPOCHS=200
LR=5e-4
MODEL="vit"
DATASET="cifar10"
DEVICE="cuda"
OUTPUT_DIR="./outputs"
RESUME=""

# Parse script arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model) MODEL="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --data_root) DATA_ROOT="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --resume) RESUME="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python -m src.CIFAR_trainer \
  --model $MODEL \
  --dataset $DATASET \
  --data_root $DATA_ROOT \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --lr $LR \
  --device $DEVICE \
  --output_dir $OUTPUT_DIR"

# Add resume if specified
if [ -n "$RESUME" ]; then
  CMD="$CMD --resume $RESUME"
fi

echo "Running: $CMD"
eval $CMD
