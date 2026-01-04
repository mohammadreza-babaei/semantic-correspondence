# Datasets

## SPair-71k
Can be downloaded from https://cvlab.postech.ac.kr/research/SPair-71k/. It must be unzipped in the ./data folder
Or you can use the script `data/prepare_spair.sh` to download and unzip it.

# Semantic Correspondence Fine-Tuning CLI

This tool provides a command-line interface (CLI) for fine-tuning **DINOv2**, **DINOv3**, and **SAM** (Segment Anything Model) for semantic correspondence tasks using the **SPair-71k** dataset.

## Key Features
* **Multi-Model Support:** Train `DINOv2`, `DINOv3`, and `SAM` architectures.
* **Experiment Tracking:** Built-in integration with **Weights and Biases (WandB)**.
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
* `--weights-path`: Path to the local file you downloaded.

### 2. Using Weights and Biases (WandB)
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

### `fine_tune`
Main command for training models.

**Usage:** `python main.py fine_tune [options]`

#### General & Model
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model-name` | str | `dinov2_vits14` | Model variant (`dinov2_vits14`, `dinov2_vitb14`, `dinov3_vits16`, `sam_vit_b`, `tiny_vit` etc.) |
| `--dataset-path` | str | `./data` | Path to SPair-71k dataset |
| `--weights-path` | str | `None` | Path to custom weights (.pth/.safetensors) |
| `--save-path` | str | `None` | Directory to save checkpoints |
| `--resume` | str | `None` | Path to checkpoint to resume training from |
| `--no-save-checkpoints`| flag | `False` | Disable saving checkpoints |

#### Training Configuration
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--epochs` | int | `5` | Number of training epochs |
| `--batch-size` | int | `1` | Batch size |
| `--lr` | float | `1e-3` | Learning rate |
| `--fixed-lr` | flag | `False` | Disable learning rate decay |
| `--num-unfrozen-blocks`| int | `2` | Number of transformer blocks to unfreeze (from end) |
| `--dropout` | float | `0.0` | Dropout rate for unfrozen blocks |
| `--weight-decay` | float | `0.01` | L2 weight decay |
| `--accumulation-steps`| int | `1` | Steps to accumulate gradients |
| `--num-augmentations` | int | `0` | Augmented versions to cache per image |
| `--no-shuffle` | flag | `False` | Disable dataset shuffling |

#### LoRA & Optimization
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--use-lora` | flag | `False` | Enable Low-Rank Adaptation (LoRA) |
| `--lora-rank` | int | `8` | LoRA rank dimension |
| `--lora-alpha` | int | `16` | LoRA scaling factor |
| `--feature-reg` | float | `0.0` | L2 regularization on feature magnitudes |
| `--unfreeze-neck` | flag | `False` | Unfreeze SAM neck layers (SAM only) |

#### Logging (WandB) & Visualization
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--use-wandb` | flag | `False` | Enable Weights & Biases logging |
| `--wandb-project` | str | `semantic_correspondence`| WandB project name |
| `--wandb-run-name` | str | `None` | WandB run name |
| `--log-interval` | int | `50` | Log progress every N batches |
| `--no-plot` | flag | `False` | Disable interactive plotting |

### `eval`
Command for evaluating trained models.

**Usage:** `python main.py eval [options]`

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model-name` | str | **Required** | Model variant (must match checkpoint) |
| `--weights-path` | str | `None` | Path to the .pt checkpoint file |
| `--split` | str | `val` | Dataset split (`test`, `val`) |
| `--alpha` | float | `0.1` | PCK threshold factor |
| `--window-size` | int | `5` | Local refinement window size |
| `--plot-pair` | int | `None` | Visualize prediction for specific pair index |
| `--num-unfrozen-blocks`| int | `2` | Must match training config |
| `--unfreeze-neck` | flag | `False` | Must match training config (SAM only) |

---

## Dataset Setup

Your `--dataset-path` should point to a folder structured like this:

```
/path/to/your/dataset/
├──SPair-71k/
|  ├── JPEGImages/      # Contains category folders (cat, dog, etc.)
|  └── PairAnnotation/  # Contains train/test split folders
└──ap10k/
   ├── annotations/      # Contains category folders (cat, dog, etc.)
   └── data/  # Contains train/test split folders
```