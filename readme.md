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

## Inference and Submission

```bash
# Default checkpoint path
python test.py --checkpoint outputs/resnet18_baseline/best_model.pt

# Specify config, checkpoint, and output path
python test.py \
  --config cfgs/config.yaml \
  --checkpoint outputs/resnet18_baseline/best_model.pt \
  --output_path outputs/resnet18_baseline/submission.csv
```

Workflow:

1. Build test dataloader from `test_images`
2. Load trained checkpoint
3. Predict labels for each test image
4. Map predictions to `sample_submission.csv` ids
5. Save final CSV

## GPU Configuration

Edit `cfgs/config.yaml` to specify GPU device:

```yaml
device: cuda:0  # or cuda:1, cuda:2, etc.
```

Or use environment variable:

```bash
CUDA_VISIBLE_DEVICES=1 python train.py  # Uses GPU 1
```
