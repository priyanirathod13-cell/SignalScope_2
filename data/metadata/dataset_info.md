# SignalScope Dataset Documentation: GenImage Benchmark Subset

## 1. Dataset Overview

* **Dataset Name**: GenImage Benchmark Subset
* **Primary Reference**: *GenImage: A Million-Scale Dataset for Detecting AI-Generated Image* (Mingyi Jiang, Qiang Zhou, Ninghua Yang, et al., NeurIPS 2023 / IEEE TPAMI).
* **Hosted Source**: [Hugging Face: jhutter2/281_Genimage](https://huggingface.co/datasets/jhutter2/281_Genimage)
* **License**:
  * Repository Subset: MIT License
  * Upstream GenImage Benchmark: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC-BY-NC-SA 4.0)
  * Real Source Images: ImageNet (ILSVRC) non-commercial research use
* **Image Specifications**:
  * Format: PNG (Synthetic) / JPEG (Real)
  * Dimensions: 256 x 256 pixels
  * Channels: 3 (RGB)
  * Bit Depth: 8-bit per channel

---

## 2. Generator Taxonomy and Provenance

The dataset comprises real-world nature photography contrasted against state-of-the-art diffusion generative models:

| Source / Generator | Type | Architecture / Family | Count | Primary Details |
| :--- | :--- | :--- | :--- | :--- |
| **Real ImageNet** | Real | Natural Photography | 2,861 | ImageNet-1K nature/wildlife/scenery photos |
| **ADM** | Synthetic | Guided Diffusion | 968 | Ablated Diffusion Model (*Dhariwal & Nichol, NeurIPS 2021*) |
| **Stable Diffusion v1.5** | Synthetic | Latent Diffusion | 996 | CompVis / RunwayML SD 1.5 text-to-image |
| **Wukong** | Synthetic | Multilingual Diffusion | 958 | Huawei Noah's Ark 100M-scale Chinese-English diffusion |
| **Total** | - | - | **5,783** | **49.5% Real / 50.5% Synthetic** |

---

## 3. Directory Layout

```
SignalScope/
├── data/
│   ├── raw/
│   │   ├── real/                  # All downloaded real images
│   │   └── synthetic/             # All downloaded AI-generated images
│   ├── processed/                 # Placeholder for feature caches or preprocessed tensors
│   ├── splits/
│   │   ├── train/
│   │   │   ├── real/              # Stratified training real set (~70%)
│   │   │   └── synthetic/         # Stratified training synthetic set (~70%)
│   │   ├── validation/
│   │   │   ├── real/              # Stratified validation real set (~15%)
│   │   │   └── synthetic/         # Stratified validation synthetic set (~15%)
│   │   └── test/
│   │       ├── real/              # Stratified test real set (~15%)
│   │       └── synthetic/         # Stratified test synthetic set (~15%)
│   └── metadata/
│       ├── dataset_info.md        # This specification
│       ├── dataset_inventory.csv  # Granular per-image manifest (hashes, labels, generators, splits)
│       └── splits_summary.json    # Exact class and generator split breakdowns
├── src/
│   └── data/
│       ├── __init__.py
│       ├── download.py            # Reproducible download and raw assembly pipeline
│       ├── make_splits.py         # Stratified split generator
│       └── verify_dataset.py      # Health, integrity, and anti-leak verification suite
└── config/
    └── dataset_config.yaml        # Dataset configuration and hyperparameters
```

---

## 4. Stratification & Partitioning Protocol

The dataset is partitioned using stratified sampling across both target labels (`real` vs. `synthetic`) and individual generative architectures (`adm`, `stable_diffusion_v1_5`, `wukong`):
* **Train Split (70%)**: Model training.
* **Validation Split (15%)**: Hyperparameter tuning, early stopping, model checkpointing.
* **Test Split (15%)**: Unbiased evaluation of generalization.

No image appears in more than one partition (zero leakage).
