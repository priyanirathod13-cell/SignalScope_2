# SignalScope V3.2 — Dataset Gap Analysis & Forensic Failure Audit

**Audit Date:** September 15, 2026  
**Audited Artifact:** data/v3/splits/train.csv ( = 14,759$) & 
eports/v3_final_candidate/  
**Focus:** Explaining why newly supplied AI-generated test images (Alpine cabin, Instagram screenshots, foggy roads, indoor group portraits) are classified as REAL with 98%–100% confidence.

---

## 1. Generator Representation in Training Data

| Generator / Source Category | Class | Train Samples | % of Train | Primary Resolution | Native Aspect Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 
eal_cifar10 | REAL | 5,600 | 37.94% |  \times 32$ | 1.00 (1:1 Square) |
| stable_diffusion_v1_4 | SYNTHETIC | 5,585 | 37.84% |  \times 32$ | 1.00 (1:1 Square) |
| 
eal_imagenet | REAL | 1,989 | 13.48% | Variable (-800$) | Variable (4:3, 3:4, 16:9) |
| stable_diffusion_v1_5 | SYNTHETIC | 697 | 4.72% |  \times 512$ (or 256) | 1.00 (1:1 Square) |
| dm (Ablated Diffusion) | SYNTHETIC | 678 | 4.59% |  \times 256$ | 1.00 (1:1 Square) |
| 
eal_night_sky (Astro) | REAL | 105 | 0.71% | Variable | Variable |
| synth_cosmic_fantasy | SYNTHETIC | 105 | 0.71% | Variable | Variable |
| **Total Active Pool** | - | **14,759** | **100.0%** | - | - |

---

## 2. Image Domains & Semantic Coverage

* **Over-Indexed Domains:**
  - Tiny  \times 32$ object icons (CIFAKE vehicles, animals, birds): **75.78%** of the entire training set.
  - ImageNet-1K object classifications (mostly animals, artifacts, indoor objects): **22.79%**.
* **Completely Absent / Severely Underrepresented Domains:**
  - **Multi-person full-body / indoor portraits**: < 0.5% (virtually absent).
  - **Atmospheric fog / heavy mist**: < 0.2% (absent in synthetic generators).
  - **Social-media screenshots (Instagram, WhatsApp UI, TikTok)**: **0.00%**.
  - **High-resolution photorealistic modern landscape diffusion (SDXL, Midjourney v5/v6, Flux)**: **0.00%**.

---

## 3. Aspect Ratio & Padding Imbalance

| Dimension | Real Class (%) | Synthetic Class (%) | Discrepancy |
| :--- | :---: | :---: | :---: |
| **Square (1:1)** | 73.2% | **98.5%** | **-25.3%** |
| **Landscape (>1.05)** | 18.5% | 0.8% | **+17.7%** |
| **Portrait (<0.95)** | 8.3% | 0.7% | **+7.6%** |

### Critical Finding:
98.5% of synthetic training images are perfect squares with zero letterbox padding. Meanwhile, the only images in training with non-square aspect ratios (and therefore solid neutral gray padding (128, 128, 128)) were real ImageNet photographs.
The convolutional backbone learned a spurious shortcut: **Presence of Letterbox Padding Margin => REAL (P > 0.98)**.

---

## 4. Semantic Attribute Correlation Analysis

| Attribute | % of REAL | % of SYNTHETIC | Correlation with Class |
| :--- | :---: | :---: | :--- |
| **Letterbox Gray Padding** | 26.8% | **1.5%** | **Strong correlation with REAL** |
| **Multi-Person Human Photos** | ~2.0% | ~0.0% | Confounds human group photos as REAL |
| **Atmospheric Fog / Low Contrast** | ~1.5% | ~0.1% | Confounds low-contrast / foggy scenes as REAL |
| **WhatsApp JPEG Recompression (Q < 75)**| 0.0% | 0.0% | Both classes were clean PNGs or standard JPEGs |
| **Mobile Screenshot Borders / UI** | 0.0% | 0.0% | Completely absent from training set |

---

## 5. Failure Attribution to Newly Failing Images

1. **Alpine Cabin AI (WhatsApp Image 2026-09-15 at 02.56.08.jpeg):**
   - In full letterbox: **REAL (98.38%)** due to 1.14 aspect ratio padding.
   - When cropped to center square or resized directly: **AI-GENERATED (96.56%)**!
   - Root cause: Letterbox padding attenuation + peripheral spatial bias.
2. **Three-Person Indoor Image (WhatsApp Image 2026-09-15 at 02.57.56.jpeg):**
   - 4:3 aspect ratio with 28px top/bottom padding + indoor human portrait domain (underrepresented).
   - In center crop: Jumps from 0.00% to **52.72% Synthetic**.
3. **Instagram Screenshot (WhatsApp Image 2026-09-15 at 02.58.54.jpeg):**
   - 9:19.5 aspect ratio with 60px lateral padding + 80% black Instagram UI.
   - In center crop: Jumps from 0.00% to **98.82% Synthetic (AI-GENERATED)**!
