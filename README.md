# Datasets

## SPair-71k
Can be downloaded from https://cvlab.postech.ac.kr/research/SPair-71k/. It must be unzipped in the ./data folder
Or you can use the script `data/prepare_spair.sh` to download and unzip it.

# Semantic Correspondence Fine-Tuning CLI

This tool provides a command-line interface (CLI) for fine-tuning **DINOv2**, **DINOv3**, and **SAM** (Segment Anything Model) for semantic correspondence tasks using the **SPair-71k** dataset.

## Key Features
* **Multi-Model Support:** Train `DINOv2`, `DINOv3`, and `SAM` architectures.
* **Experiment Tracking:** Built-in integration with **Weights & Biases (WandB)**.
* **Flexible Training:** Configure learning rates, frozen blocks, and epochs via CLI.

---

## Guide

The script is run via `main.py` using the `fine_tune` subcommand.

### 1. Models Training

```bash
python main.py fine_tune \
  --model-name [model name] \
  --weights-path checkpoints/dino_weights.pth  \ 
  --dataset-path /path/to/SPair-71k \
  --epochs [num] \
  --max-iters-per-epoch [num] \
```

* `--weights-path`: Path to the local file you downloaded. (only for Dinov3)

### 2. Using Weights & Biases (WandB)

Track loss curves and metrics online by adding the wandb flags.

```bash
python main.py fine_tune \
  --dataset-path /path/to/SPair-71k \
  --model-name dinov3_vits16 \
  --use-wandb \
  --wandb-project "semantic-correspondence" \
  --wandb-run-name "dinov3-run-01"

```

### 3. Hyperparameters

Customize the training process (learning rate, freezing, batch size).

```bash
python main.py fine_tune \
  --model-name dinov2_vitb14 \
  --num-unfrozen-blocks 4 \
  --lr 5e-5 \
  --batch-size 4 \
  --no-plot \
  --fixed-lr

```

* `--num-unfrozen-blocks 4`: Unfreezes the last 4 layers (default is 2).
* `--no-plot`: Disables the popup visualization windows (useful for servers).
* `--fixed-lr`: Disables learning rate decay.

---

## Command Reference

### Core Arguments

| Argument | Description | Default |
| --- | --- | --- |
| `fine_tune` | **Required.** The command to start training. | N/A |
| `--dataset-path` | Local path to the SPair-71k dataset root. | `./data` |

### Model & Training Options

| Argument | Description | Default |
| --- | --- | --- |
| `--model-name` | Architecture to use. <br>

**Choices:** `dinov2_vits14`, `dinov2_vitb14`, `dinov3_vits16`, `facebook/sam-vit-base`, `facebook/sam-vit-large`, `facebook/sam-vit-huge`| `dinov2_vits14`|

---

## Dataset Setup

Your `--dataset-path` should point to a folder structured like this:

```
/path/to/SPair-71k/
├── JPEGImages/      # Contains category folders (cat, dog, etc.)
└── PairAnnotation/  # Contains train/test split folders
```