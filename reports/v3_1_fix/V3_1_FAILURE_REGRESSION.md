# SignalScope V3.1 Failure Mode Regression Report

## 1. Executive Summary
This report analyzes the performance of SignalScope V3.1 on the verified failure modes of V3:
1. **WhatsApp-recompressed AI photographs**
2. **Non-square / portrait AI photographs**
3. **Dark / astrophotography-style AI scenes**

In V3, these images were incorrectly classified as REAL with 98% to 100% confidence due to two verified spurious shortcuts:
- Grey letterbox padding was 100% correlated with REAL in V3 training.
- WhatsApp lossy DCT compression stripped high-frequency diffusion artifacts.

---

## 2. Direct Regression on User Diagnostic Failure Images

| Image Filename | Dimensions & Ratio | V3 Classification (Baseline) | V3.1 Classification | V3.1 Calibrated Prob (Synth) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `WhatsApp Image 2026-09-15 at 02.56.08.jpeg` | 1341x1173 (landscape) | REAL (98.2% - 100.0%) | **REAL** | **15.45%** | **[FAIL]** |
| `WhatsApp Image 2026-09-15 at 02.56.40.jpeg` | 1179x1414 (portrait) | REAL (98.2% - 100.0%) | **REAL** | **23.33%** | **[FAIL]** |
| `WhatsApp Image 2026-09-15 at 02.58.54.jpeg` | 738x1600 (portrait) | REAL (98.2% - 100.0%) | **REAL** | **18.78%** | **[FAIL]** |

### Detailed Forensic Logits & Probabilities

- **WhatsApp Image 2026-09-15 at 02.56.08.jpeg**:
  - Dimensions: 1341x1173 (landscape)
  - Raw Logits: `[Real: 0.8937, Synth: -0.9625]`
  - Calibrated Probabilities: Real = 84.55%, Synth = 15.45%
  - Final Prediction: **REAL** (84.55% confidence)

- **WhatsApp Image 2026-09-15 at 02.56.40.jpeg**:
  - Dimensions: 1179x1414 (portrait)
  - Raw Logits: `[Real: 0.8387, Synth: -0.4610]`
  - Calibrated Probabilities: Real = 76.67%, Synth = 23.33%
  - Final Prediction: **REAL** (76.67% confidence)

- **WhatsApp Image 2026-09-15 at 02.58.54.jpeg**:
  - Dimensions: 738x1600 (portrait)
  - Raw Logits: `[Real: 0.8631, Synth: -0.7364]`
  - Calibrated Probabilities: Real = 81.22%, Synth = 18.78%
  - Final Prediction: **REAL** (81.22% confidence)

