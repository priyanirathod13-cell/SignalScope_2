# SignalScope V3.1 Dataset Audit & Statistical Parity Report

## 1. Executive Summary
SignalScope V3.1 introduces a redesigned training and evaluation corpus specifically engineered to eradicate the three primary spurious correlations verified in V3:
1. **Aspect-Ratio / Letterbox Padding Shortcut**: In V3, 100% of non-square images requiring neutral grey letterbox borders were Real camera photographs, allowing the network to use padding as a decisive heuristic for Class 0 (`REAL`).
2. **Social Media Lossy Compression Vulnerability**: Standard training images possessed pristine micro-textures that were stripped by WhatsApp/social media recompression, leaving the model defenseless against recompressed AI photographs.
3. **Astrophotography / Dark Scene Semantic Shortcut**: Real astrophotography had no synthetic counterpart in early training, associating dark/starfield scenes with `REAL`.

The V3.1 dataset achieves **exact statistical symmetry** across all three vulnerability vectors.

---

## 2. Statistical Audit Results

### 2.1 Class Balance
- **Total Samples**: 10,500
- **Real Samples**: 5,250 (50.00%)
- **Synthetic Samples**: 5,250 (50.00%)
- **Class Imbalance**: **0.00%** (Target: $< 5.0\%$)

### 2.2 Aspect-Ratio Symmetry & Padding Parity
When images are processed via the aspect-ratio-preserving letterbox pipeline into $224 \times 224$:
- Non-square images (landscape and portrait) receive neutral grey padding (`(128, 128, 128)`).
- Square images receive zero padding.

| Aspect Ratio | Real Count | Real % | Synthetic Count | Synthetic % | Absolute Difference |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Landscape (4:3)** | 1,750 | 33.33% | 1,750 | 33.33% | **0.00%** |
| **Portrait (3:4)** | 1,750 | 33.33% | 1,750 | 33.33% | **0.00%** |
| **Square (1:1)** | 1,750 | 33.33% | 1,750 | 33.33% | **0.00%** |
| **Total** | 5,250 | 100.0% | 5,250 | 100.0% | **0.00%** |

- **Real Images Requiring Padding**: 66.67%
- **Synthetic Images Requiring Padding**: 66.67%
- **Padding Frequency Difference**: **0.00%** (Verified: Letterbox shortcut is mathematically eliminated).

### 2.3 Astrophotography & Night Scene Parity
- **Real Night Sky & Astrophotography**: 450 samples
- **Synthetic Cosmic & Fantasy Diffusion**: 450 samples
- **Ratio**: **1.00 : 1.00** (Perfect semantic balance).

---

## 3. Data Integrity & Leakage Prevention

### 3.1 Group-Aware Partitioning (70% Train / 15% Val / 15% Test)
To prevent cross-crop data leakage, images derived from the same source image are assigned the same `base_group` ID. Partitioning is performed strictly at the group level:

- **Train Split**: 7,350 samples (3,675 Real, 3,675 Synthetic)
- **Validation Split**: 1,572 samples (786 Real, 786 Synthetic)
- **Test Split**: 1,578 samples (789 Real, 789 Synthetic)

### 3.2 Leakage Verification
- **Train-Val SHA-256 Overlap**: 0
- **Train-Test SHA-256 Overlap**: 0
- **Val-Test SHA-256 Overlap**: 0
- **Train-Val Base Group Overlap**: 0
- **Train-Test Base Group Overlap**: 0
- **Val-Test Base Group Overlap**: 0
- **Leakage Status**: **100% Clean / Zero Data Leakage Confirmed**.

---

## 4. Strict Generator Isolation
- **Training & Validation Generators**:
  - Real: ImageNet, Real Night Sky / Astrophotography
  - Synthetic: Stable Diffusion v1.5, ADM Guided Diffusion, Cosmic Fantasy Diffusion
- **Zero-Shot Holdout Benchmark**:
  - **Wukong Generator**: 100% held out. Zero samples present in training, validation, or test CSVs. Reserved exclusively for unseen-generator generalization assessment.
- **User Diagnostic Images**:
  - Excluded from all training and validation data. Evaluated strictly post-training as a regression benchmark.
