#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-cfgs/resnet50_v1.yaml}
CHECKPOINT_PATH=${2:-outputs/resnet50_v1/best_model.pt}
OUTPUT_PATH=${3:-outputs/resnet50_v1/submission.csv}

python test.py \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --output_path "${OUTPUT_PATH}"
