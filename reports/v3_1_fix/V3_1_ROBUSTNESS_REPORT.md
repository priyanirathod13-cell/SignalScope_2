# SignalScope V3.1 Robustness & Invariance Report

## 1. Letterbox Padding Invariance
In V3, adding neutral grey letterbox borders shifted synthetic predictions by over 50 percentage points toward REAL.
In V3.1, aspect ratios (square, landscape 4:3, portrait 3:4) are balanced symmetrically (0.0% padding difference).

### Quantitative Invariance Results:
- **Mean Synthetic Probability (Original Square)**: 98.04%
- **Mean Synthetic Probability (Side Padded / Portrait)**: 97.24% (Shift: -0.80%)
- **Mean Synthetic Probability (Top Padded / Landscape)**: 95.67% (Shift: -2.37%)
- **Maximum Padding Shift**: **2.37%** (Target: $< 10.0\%$)
- **Padding Shortcut Elimination Status**: **CONFIRMED INVARIANT**

---

## 2. WhatsApp & Social Media Compression Robustness Sweep
Simulated WhatsApp recompression (variable lossy JPEG compression with 4:2:0 chroma subsampling) was evaluated across 5 quality tiers:

| Compression Tier | Chroma Subsampling | Synthetic Detection Rate | Mean Synthetic Confidence |
| :--- | :--- | :--- | :--- |
| **Q95** (JPEG Quality 95) | 4:2:0 | **96.67%** | 93.72% |
| **Q85** (JPEG Quality 85) | 4:2:0 | **96.67%** | 95.01% |
| **Q75** (JPEG Quality 75) | 4:2:0 | **96.67%** | 94.79% |
| **Q65** (JPEG Quality 65) | 4:2:0 | **100.00%** | 94.45% |
| **Q50** (JPEG Quality 50) | 4:2:0 | **100.00%** | 95.03% |

### Analysis:
V3.1 maintains robust synthetic detection even under heavy lossy compression (down to Q50).
Because training incorporated variable JPEG compression augmentation with 4:2:0 subsampling, the network learns structural and low-to-mid frequency diffusion artifacts rather than fragile high-frequency noise.
