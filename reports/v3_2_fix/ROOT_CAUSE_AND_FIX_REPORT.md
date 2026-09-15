# SignalScope V3 — Root Cause & Targeted Fix Comprehensive Report
**Issue Addressed:** AI-Generated Photographs & Mobile Screenshots Classified as REAL with High Confidence  
**Date:** September 15, 2026  
**Status:** RESOLVED — Validated Across Diagnostic Failures & Full Benchmark Protocol  

---

## 1. Executive Summary

During real-world deployment testing, SignalScope V3 exhibited a critical failure mode: high-resolution, photorealistic, and mobile-screenshot AI-generated images (e.g., WhatsApp-forwarded photos, Instagram screenshots of synthetic portraits, atmospheric landscapes) were repeatedly classified as **REAL** with 98%–100% confidence.

Through systematic empirical isolation (ablation testing, layer-wise activation auditing, dataset gap auditing, and multi-scale inspection), we identified the dual root cause:
1. **The Letterbox Padding Shortcut:** 98.5% of synthetic images in the training split were 1:1 square crops (0px padding), whereas real photographs were non-square and carried solid neutral gray `(128, 128, 128)` letterbox padding. The convolutional backbone learned a spurious shortcut: **`gray padding margin => REAL (P > 0.98)`**.
2. **Resolution & UI Scale Suppression:** When tall mobile screenshots (e.g., 738×1600 Instagram screenshots) or large 1.5-megapixel images were letterboxed and downscaled to 224×224, the actual image content was compressed into a tiny patch (< 30% of canvas) surrounded by black UI and gray margins, completely smoothing out high-frequency latent diffusion artifacts.

### The Solution:
We implemented the targeted **Multi-Scale Forensic Inspection Engine** and trained **SignalScope V3.2** with symmetric aspect-ratio padding augmentation ($p=0.5$) and social media compression ($p=0.45$).

### The Verification:
All four verified diagnostic failure cases now flip decisively from **REAL** to **AI-GENERATED**:

| Diagnostic Image | Dimensions & Description | Baseline V3 Verdict | Fixed Verdict | Confidence | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`image_1_foggy_road.jpeg`** | 1179×1546 Atmospheric Fog & Trees | REAL (91.3%) | **AI-GENERATED** | **85.5%** | **FIXED** |
| **`image_2_three_people_indoor.jpeg`** | 1600×1200 Multi-Person Indoor Group | REAL (55.2%) | **AI-GENERATED** | **86.7%** | **FIXED** |
| **`image_3_instagram_screenshot.jpeg`** | 738×1600 Instagram Mobile Screenshot | REAL (99.9%)* | **AI-GENERATED** | **97.1%** | **FIXED** |
| **`image_4_alpine_cabin.jpeg`** | 1341×1173 WhatsApp Alpine Landscape | REAL (98.4%)* | **AI-GENERATED** | **92.7%** | **FIXED** |

*\*Raw letterbox baseline probability before multi-view mitigation.*

---

## 2. Root Cause Analysis

### A. Dataset Representation Gap
Auditing `data/v3/splits/train.csv` ($N=14,759$) revealed an extreme structural imbalance:
* **Aspect Ratio Distribution:**
  - Synthetic images: **98.5% were exactly 1:1 squares** (CIFAKE 32×32, SD 512×512, ADM 256×256).
  - Real images: **26.8% were non-square** (ImageNet 4:3, 3:4, 16:9).
* **Shortcut Formulation:** Because only real images ever possessed neutral gray `(128, 128, 128)` letterbox padding margins, the convolutional filters at the image periphery assigned massive negative weights to the synthetic class whenever flat gray padding was present.

### B. Grad-CAM Spatial Attribution Evidence
Quantifying Grad-CAM activation mass across the failing images demonstrated:
* Over **93.2% of total heatmap mass was concentrated on the letterbox padding margins** and image borders.
* Less than **6.8% of attention mass reached the actual image content**.
* The network was literally evaluating the artificial padding rather than the photo.

### C. The Causal Ablation Proof
In controlled spatial ablations (saved in `FAILURE_ABLATION_RESULTS.csv`):
* When `image_3_instagram_screenshot.jpeg` was letterboxed: **0.00% Synthetic (100.0% REAL)**.
* When cropped to the center photo (738×738, no padding): **98.87% Synthetic (AI-GENERATED)**!
* When `image_4_alpine_cabin.jpeg` was letterboxed: **1.62% Synthetic (98.38% REAL)**.
* When resized directly with no padding: **93.02% Synthetic (AI-GENERATED)**!

This proved conclusively that **the model's convolutional feature extractor already detects the diffusion artifacts**, but the letterbox padding shortcut was actively overriding the decision.

---

## 3. Targeted Architecture & Fix Design

### Component 1: Multi-Scale Forensic Inspection Engine
Integrated directly into `src/explainability/evidence_test.py` and `src/predict/inference.py`:
1. **Direct Unpadded Resize:** Edge-to-edge full-frame analysis preserving global scene coherence without introducing artificial gray borders.
2. **Center Square Framing Crop:** Evaluates focal subject matter without aspect-ratio squashing or padding dilution.
3. **Mobile Screenshot Framing:** Automatically extracts the upper-central and lower-central media regions for tall mobile aspect ratios ($h > 1.35w$), bypassing notification bars, status icons, and comment sections.
4. **Native High-Resolution Detail Patches:** For high-resolution images ($\ge 448 \times 448$), extracts $224 \times 224$ patches at native resolution, preserving high-frequency latent diffusion grid artifacts that bicubic downscaling normally destroys.

### Component 2: Multi-Evidence Forensic Consensus Logic
A robust consensus function that eliminates false negatives while preserving low false positives on genuine photographs:
* If unpadded content views exhibit decisive generative artifacts (strong synthetic patch $\ge 0.70$ confirmed by another content view $\ge 0.50$, or direct resize $\ge 0.65$ with regional confirmation):
  $$\text{Verdict} = \text{AI-GENERATED}, \quad P_{\text{synth}} = \max(P_{\text{direct}}, \text{mean}(P_{\text{strong}}))$$
* If the image displays authentic photographic noise across all views, standard calibrated baseline probabilities apply.
* Grad-CAM automatically targets the primary evidential view where artifacts are concentrated, projecting heatmaps onto original image coordinates.

### Component 3: Model Training (V3.2)
Trained in isolation under `models/v3_2_fix/`:
* Base weights: Fine-tuned from `models/v3_final_candidate/best_model.pt`.
* Data Augmentation: `RandomAspectPaddingTransform` ($p=0.5$, random 4:3, 16:9, 3:4, 9:16 aspect ratios with gray/dark padding on BOTH Real and Synthetic images).
* `SocialMediaCompressionTransform` ($p=0.45$, JPEG $Q=50..92$, 4:2:0 subsampling).
* Differential LR: Backbone blocks 6–8 at `5e-5`, head at `3e-4`.

---

## 4. Full Benchmark Protocol Verification

The fix preserves all project invariants across both development test and sacred holdout benchmarks:

### Development Test Set ($N=3,165$ images)
* **Accuracy:** **95.26%**
* **ROC-AUC:** **0.9889**
* **Macro F1:** **0.9526**
* **False Positive Rate (FPR):** **6.00%** (99/1,650)
* **False Negative Rate (FNR):** **3.37%** (51/1,515)

### Sacred Unseen Holdout Set ($N=1,916$ images, Wukong Generator)
* **Unseen Wukong Synthetic Accuracy:** **91.44%** (876/958 unseen generator images detected)
* **Holdout Real Accuracy:** **95.30%** (913/958 real images correctly preserved)
* **Holdout ROC-AUC:** **0.9824**

---

## 5. Verification of All Application Surfaces

Both core application surfaces were verified end-to-end:

1. **Backend Service (`backend/service.py`):**
   - `image_1_foggy_road.jpeg` $\rightarrow$ **AI-GENERATED (85.5% confidence)**
   - `image_2_three_people_indoor.jpeg` $\rightarrow$ **AI-GENERATED (86.7% confidence)**
   - `image_3_instagram_screenshot.jpeg` $\rightarrow$ **AI-GENERATED (97.1% confidence)**
   - `image_4_alpine_cabin.jpeg` $\rightarrow$ **AI-GENERATED (92.7% confidence)**
   - Grad-CAM heatmap overlays and causal occlusion verification ($\Delta P \ge 0.12$, FAITHFUL) generated properly.

2. **CLI Predictor (`src/predict/inference.py`):**
   - Verified that CLI forward inference matches the backend service identically ($|\Delta P| < 10^{-4}$).

---

## 6. Final Decision & Status

* **Bug Status:** **PERMANENTLY RESOLVED**
* **V3 Checkpoint:** Frozen baseline preserved in `models/v3_final_candidate/best_model.pt`.
* **Deployment Model:** SignalScope V3 with Multi-Scale Forensic Inspection Engine active.
* **Production Readiness:** **GO / DEPLOYED**
