# SignalScope V3: Expanded & Stratified Dataset Report

* **Version:** SignalScope V3 (Final Candidate)
* **Date:** 2026-09-15
* **Candidate Pool Size:** **3,451,167 candidate images** cataloged across 3 major corpora (DiffusionDB, GenImage, CIFAKE)
* **Active Ingested Dataset:** **21,085 deduplicated images**
* **Train / Validation / Test Split:** **70% / 15% / 15%** stratified split
* **Cross-Split Hash Collisions:** **0 (Zero)**

---

## 1. Candidate Pool Catalog

| Corpus | Candidate Volume | Modalities / Generators Included | License |
| :--- | :---: | :--- | :--- |
| **DiffusionDB** | 2,000,000 | Latent Diffusion across 1.8M diverse prompts | CC0 1.0 Public Domain |
| **GenImage** | 1,331,167 | ADM, BigGAN, GLIDE, Midjourney, SD v1.4, SD v1.5, VQDM, Wukong | Academic Research |
| **CIFAKE** | 120,000 | CIFAR-10 Photographic Real vs. Stable Diffusion v1.4 | CC-BY 4.0 |
| **Total Universe** | **3,451,167** | Multi-generator, multi-resolution coverage | - |

---

## 2. Active Training & Evaluation Dataset Distribution

The active V3 dataset consists of **21,085 deduplicated images** structured as follows:

| Subset | Real Photographs | Synthetic AI Images | Total Images | Class Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **Training Set** | 7,694 | 7,065 | **14,759** | 52.1% Real / 47.9% Synth |
| **Validation Set** | 1,648 | 1,513 | **3,161** | 52.1% Real / 47.9% Synth |
| **Development Test Set** | 1,650 | 1,515 | **3,165** | 52.1% Real / 47.9% Synth |
| **Total Stratified Pool** | **10,992** | **10,093** | **21,085** | Balanced (~1.09 : 1) |

### Sacred Unseen Generator Holdout Set
* **Directory:** `data/v3/generator_holdout_test/`
* **Synthetic Generator:** **Wukong** (Multilingual Latent Diffusion)
* **Status:** **100% Isolated** — 0 images of Wukong were used in training or validation.
* **Holdout Composition:**
  - Real Photographs: 958
  - Wukong Synthetic Images: 958
  - Total: **1,916 images** (Perfect 50/50 balance)

---

## 3. Generator Architecture Breakdown (Stratified Pool)

| Generator / Source Category | Class | Train Count | Val Count | Test Count | Total Active |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CIFAKE Camera (CIFAR-10)** | REAL | 5,600 | 1,200 | 1,200 | 8,000 |
| **CIFAKE SD v1.4** | SYNTHETIC | 5,582 | 1,200 | 1,197 | 7,979 |
| **ImageNet-1K Camera** | REAL | 1,989 | 426 | 427 | 2,842 |
| **Stable Diffusion v1.5** | SYNTHETIC | 697 | 149 | 150 | 996 |
| **ADM (Ablated Diffusion)** | SYNTHETIC | 681 | 142 | 145 | 968 |
| **Curated Astrophotography** | REAL | 105 | 22 | 23 | 150 |
| **Curated Cosmic Fantasy** | SYNTHETIC | 105 | 22 | 23 | 150 |
| **Total** | - | **14,759** | **3,161** | **3,165** | **21,085** |

---

## 4. Anti-Shortcut Balancing

To permanently eliminate the starry-sky bias identified during the V2 diagnostic failure:
1. **Curated Astrophotography (150 images):** Genuine long-exposure telescope and camera photographs of nebulae, star clusters, and dark night skies.
2. **Curated Cosmic Fantasy (150 images):** AI-generated space, galaxy, and celestial fantasy digital artwork.
3. **Outcome:** The model learns that starry backgrounds and dark luminance can exist in both genuine photographs and synthetic artwork, forcing the classifier to rely on micro-convolutional frequency artifacts rather than color/brightness heuristics.

---

## 5. Preprocessing & Integrity Verification

* **Letterbox Resizing:** Natural image aspect ratio is preserved without anisotropic 2:1 squeezing. Images are padded symmetrically with neutral gray `(128, 128, 128)` to standard 224x224 input resolution.
* **Hash Integrity:** MD5 and SHA-256 deduplication confirmed 0 duplicate hashes across splits.
