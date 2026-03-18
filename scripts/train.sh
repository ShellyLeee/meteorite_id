#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-cfgs/resnet50_v1.yaml}

python train.py --config "${CONFIG_PATH}"
