# SignalScope Multi-Dataset Expansion Report

**Generated:** 2026-09-14 21:14:17  
**Status:** COMPLETED & VERIFIED LEAKAGE-FREE  
**Official SIH Held-Out Test Set Used:** **NO** (Preserved untouched)

---

## 1. Dataset Overview & High-Level Summary

| Metric | Value |
|---|:---:|
| **Total Images** | **21,743** |
| **Real Images (Label = 0)** | **10,842 (49.86%)** |
| **Synthetic Images (Label = 1)** | **10,901 (50.14%)** |
| **Datasets Integrated** | **CIFAKE, GenImage** |
| **Distinct Generators Represented** | **6** |
| **Train Set Partition (70%)** | **15,220** |
| **Validation Set Partition (15%)** | **3,261** |
| **Test Set Partition (15%)** | **3,262** |
| **Corrupted Images Quarantined** | **0** |
| **Cross-Split Data Leakage** | **0 (None)** |

---

## 2. Generator & Source Architecture Distribution

| Generator Name | Family / Category | Source Dataset | Image Count | Percentage |
|---|---|---|:---:|:---:|
| `adm` | pixel_diffusion | GenImage | 968 | 4.45% |
| `real_cifar10` | real_camera | CIFAKE | 8,000 | 36.79% |
| `real_imagenet` | real_camera | GenImage | 2,842 | 13.07% |
| `stable_diffusion_v1_4` | latent_diffusion | CIFAKE | 7,979 | 36.7% |
| `stable_diffusion_v1_5` | latent_diffusion | GenImage | 996 | 4.58% |
| `wukong` | multilingual_diffusion | GenImage | 958 | 4.41% |

---

## 3. Split Stratification Breakdown

| Split | Real Photography | Synthetic AI-Generated | Total Samples | Split Percentage |
|---|:---:|:---:|:---:|:---:|
| **Train** | 7,589 | 7,631 | 15,220 | 70.0% |
| **Validation** | 1,626 | 1,635 | 3,261 | 15.0% |
| **Test** | 1,627 | 1,635 | 3,262 | 15.0% |
| **Combined** | **10,842** | **10,901** | **21,743** | **100.0%** |

---

## 4. Diversity Audit

### Real Photography Diversity
* **ImageNet-1K**: Covers natural landscapes, animals, wild plants, household objects, architecture, and food.
* **CIFAR-10**: Covers multi-angle vehicles (airplanes, automobiles, ships, trucks) and domestic/wild animals (birds, cats, deer, dogs, frogs, horses).

### Synthetic Generative Diversity
* **Latent Diffusion**: Stable Diffusion v1.5 (512x512) + Stable Diffusion v1.4 (32x32 upscaled/normalized).
* **Pixel-Space Guided Diffusion**: Ablated Diffusion Model (ADM).
* **Multilingual Diffusion**: Wukong Diffusion model.

---

## 5. Licensing & Provenance

1. **GenImage Benchmark**: Licensed under CC-BY-NC-SA 4.0 / MIT Subset. Academic research use.
2. **CIFAKE Benchmark**: Bird & Lotfi (IEEE Access 2023). Licensed under CC-BY 4.0 / Apache-2.0.
3. **Official SIH Held-Out Test Set**: **NOT ACCESSED, NOT DOWNLOADED, NOT TOUCHED.** Exclusively reserved for jury scoring.

---

## 6. Verification Status

* **SHA-256 Deduplication**: **PASSED**
* **Cross-Split Hash Isolation**: **PASSED (0 overlapping hashes)**
* **Readability & Format Validation**: **PASSED**
* **Label Integrity (`REAL=0`, `SYNTHETIC=1`)**: **PASSED**
