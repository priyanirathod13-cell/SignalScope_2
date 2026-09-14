# SignalScope Dataset Validation & Preparation Report

Generated on: 2026-09-13
Pipeline: `src/data/prepare_and_validate.py`

## 1. Executive Summary

The SignalScope dataset preparation and validation pipeline has completed with **zero errors**.
Every image was opened, parsed, verified, and hashed.

* **Total Images Processed**: 5764
* **Real Images**: 2842 (Label 0)
* **Synthetic Images**: 2922 (Label 1)
* **Corrupted / Unusable Images**: 0
* **Duplicate Images**: 0 (0 cross-class, 0 cross-generator)
* **Status**: **READY FOR MODEL TRAINING**

---

## 2. Partition & Leakage Verification

| Split | Real (0) | Synthetic (1) | Total | Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 1989 | 2046 | 4035 | 70.0% |
| **Validation** | 426 | 438 | 864 | 15.0% |
| **Test (Dev)** | 427 | 438 | 865 | 15.0% |
| **Total** | **2842** | **2922** | **5764** | **100.0%** |

* **Disjointness Assertion**:
  * `Train ∩ Validation`: 0 SHA-256 overlaps
  * `Train ∩ Test`: 0 SHA-256 overlaps
  * `Validation ∩ Test`: 0 SHA-256 overlaps
  * **Result: 100% Leakage-Safe**

---

## 3. Generative Architecture Representation

| Generator | Source Dataset | Count | Percent |
| :--- | :--- | :--- | :--- |
| **real_imagenet** | ImageNet-1K | 2842 | 49.3% |
| **stable_diffusion_v1_5** | GenImage-SD15 | 996 | 17.3% |
| **adm** | GenImage-ADM | 968 | 16.8% |
| **wukong** | GenImage-Wukong | 958 | 16.6% |

---

## 4. Resolution & Format Distribution

* **Formats**:
  * PNG: 2922 images
  * JPEG: 2842 images

* **Dimensions**:
  * Width: min=50, max=3648, mean=451.34
  * Height: min=48, max=3200, mean=418.0
  * Most common resolutions:
    - 512x512: 1953 images
    - 256x256: 978 images
    - 500x375: 681 images
    - 500x333: 261 images
    - 375x500: 202 images
