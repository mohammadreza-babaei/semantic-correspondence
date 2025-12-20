# Semantic Correspondence

## Project Overview

This project aims to tackle the task of **semantic correspondence** (finding matching keypoints across images of different instances of the same object category) using modern vision foundation models. The codebase includes implementations for loading the **SPair-71k** benchmark dataset and extracting features from state-of-the-art self-supervised models, specifically **DINOv2**, **DINOv3**, and **SAM (Segment Anything Model)**. It also includes infrastructure for fine-tuning these models using specialized loss functions.

## File Structure & Module Analysis

### Data Handling

* **`src/spair_dataset.py`**: Contains the core PyTorch `Dataset` implementations:
* `SPair71kPairs`: Loads image pairs (source and target) along with their annotations (keypoints, bounding boxes, segmentation masks, viewpoint variations). It handles the mapping between dataset splits (train, val, test) and directory structures.
* `SPair71kImages`: A dataset class for loading individual images and their annotations from SPair-71k.


* **`src/data_loading.ipynb`**: A Jupyter notebook designed to demonstrate loading the dataset and visualizing image pairs with their keypoints. **Note:** It references missing dependencies (see Issues section).

### Feature Extraction

The project provides wrappers for three different foundation models to standardize feature extraction:

* **`src/dinov2_features.py`**:
    * **`DINOv2FeatureExtractor`**: A wrapper around Meta's DINOv2. It handles image preprocessing (resizing to multiples of patch size), feature extraction (extracting patch tokens), and mapping keypoints between image and feature coordinates.
    * **`DINOv2FineTuner`**: A comprehensive class for fine-tuning DINOv2. It implements a training loop that freezes early layers, pre-calculates and caches intermediate features (for efficiency), and trains the last  blocks using a custom loss function.


* **`src/dinov3_features.py`**:
    * **`DINOv3FeatureExtractor`**: A wrapper for the newer **DINOv3** model (released ~2025). It follows the same API as the V2 extractor, loading models like `dinov3_vits16` from PyTorch Hub. It includes logic to handle standard ImageNet normalization and patch sizes (defaulting to 16).


* **`src/sam_features.py`**:
* **`SAMFeatureExtractor`**: A wrapper for the **Segment Anything Model (SAM)** using the HuggingFace `transformers` library. It extracts features specifically from SAM's vision encoder (`get_image_embeddings`), normalizing them for use in correspondence tasks.



### C. Model & Training

* **`src/loss_and_model.py`**:
    * Contains various loss functions adapted from "GeoAware-SC", including `get_corr_map_loss` (End-Point Error) and `self_contrastive_loss`.
* **`DinoV2KeypointDetector`**: A neural network module that places a simple 1x1 Convolutional head on top of DINOv2 features to predict keypoint heatmaps.


* **`src/new_loss.py`**:
* Implements a specialized dense loss function (`loss`). It uses a **Soft-Argmax** approach to differentiably predict matching keypoint coordinates from feature similarity maps and penalizes the L2 distance between predicted and ground-truth target keypoints.

* **`src/pck.py`**:
    - Metric Definition: The function implements PCK@αbbox​, where a prediction is correct if $∥k^−k_{gt}​∥ 2​≤α⋅max(W_{bbox}​,H_{bbox}​)$.

    - Bounding Box: The code assumes the standard SPair-71k format of [xmin, ymin, xmax, ymax]. It calculates the max dimension $(max(w, h))$.

    - Masking: A mask argument is included to handle cases where keypoints are occluded, truncated, or not present in the specific image pair (which is common in SPair-71k).

    - Batch Handling: The code supports both batched $(B, N, 2)$ and single-sample $(N, 2)$ inputs via broadcasting.
