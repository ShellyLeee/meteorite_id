# meteorite_id

A clean PyTorch baseline for Kaggle **meteorite-identification** (binary image classification).

## Overview

- Task: classify stone images into labels `{0, 1}`
- Model baseline: pretrained ResNet (`resnet18` or `resnet50`)
- Loss: CrossEntropyLoss with 2 logits
- Validation metric for checkpointing/early stopping: **F1**

## Directory

- `cfgs/`: experiment configs
- `common/`: logging, loss, metrics, optimizer/scheduler, utilities
- `datasets/`: dataset class, transforms, dataloader builders
- `models/`: base model + backbones
- `trainers/`: trainer and prediction/submission helpers
- `train.py`: training entrypoint
- `test.py`: inference + submission entrypoint

## Expected Dataset Layout

Dataset paths are config-driven using `data_root` + relative names:

```text
<data_root>/
├── train_labels.csv
├── training_images/
│   ├── 000001.jpg
│   └── ...
├── test_images/
│   ├── 100001.jpg
│   └── ...
└── sample_submission.csv
```

`train_labels.csv` must include columns:

- `id` (image filename)
- `label` (integer class)

`sample_submission.csv` must include `id`. Submission row order follows this file.

## Training

```bash
# Default (uses cfgs/config.yaml)
python train.py

# Specify config file
python train.py --config cfgs/config.yaml
```

Best checkpoint is saved to:

- `<output_dir>/best_model.pt`

## Inference and Submission (Single / CV / Ensemble)

```bash
# By experiment name (auto-resolves config, checkpoint, and output path)
python test.py --exp resnet50/baseline

# By experiment path
python test.py --exp ./outputs/resnet50/baseline

# Specify config directly (sources from config)
python test.py --config cfgs/config.yaml

# Single-checkpoint override (highest priority)
python test.py \
  --config cfgs/config.yaml \
  --checkpoint outputs/resnet50/baseline/best_model.pt \
  --output_path outputs/resnet50/baseline/submission.csv
```

### Prediction Source Resolution Priority

1. `--checkpoint` (single source override)
2. `prediction.sources` (or top-level `prediction_sources`) in config
3. `--exp` auto-discovery:
   - `fold_*/best_model.pt` -> CV ensemble
   - fallback `best_model.pt` -> single model

### Config-driven Ensemble

Use config to manage prediction sources (no hardcoded path in code):

```yaml
prediction:
  threshold: 0.5
  aggregation: mean
  sources:
    - model_name: resnet50
      checkpoint_path: ./outputs/resnet50/cv_exp/fold_0/best_model.pt
      fold: 0
      seed: 42
      weight: 1.0
    - model_name: resnet50
      checkpoint_path: ./outputs/resnet50/cv_exp/fold_1/best_model.pt
      fold: 1
      seed: 42
      weight: 1.0
```

Workflow:

1. Build test dataloader from `test_images`
2. For each source: load model + checkpoint and predict binary probability
3. Aggregate probabilities with weighted mean (`aggregation: mean`)
4. Apply threshold (`threshold`, default `0.5`) to get final labels
5. Map to `sample_submission.csv` ids and save final CSV

## GPU Configuration

Edit `cfgs/config.yaml` to specify GPU device:

```yaml
device: cuda:0  # or cuda:1, cuda:2, etc.
```

Or use environment variable:

```bash
CUDA_VISIBLE_DEVICES=1 python train.py  # Uses GPU 1
```
