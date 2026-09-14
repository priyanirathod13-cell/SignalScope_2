# SignalScope V3 AI Detection Root-Cause Investigation & Audit

**Date:** September 15, 2026  
**System:** SignalScope V3 Final Candidate (`EfficientNet-B0`, 3.16M Parameters)  
**Checkpoint Path:** `models/v3_final_candidate/best_model.pt`  
**Investigation Scope:** Verification of Checkpoint, Direct Inference vs. API, Class Inversion Audit, Preprocessing, Calibration, Decision Threshold, Fallback/Caching Detection, and Model Weight Integrity.  

---

## 1. Problem Description

During live browser testing on the SignalScope web application, users observed that certain AI-generated photographs (downloaded and shared via WhatsApp, e.g. `WhatsApp Image 2026-09-15 at 02.56.08.jpeg`, `02.56.40.jpeg`, and `02.58.54.jpeg`) were classified as **`REAL`** with high confidence ($98.2\%$ to $100.0\%$).

This investigation was conducted to determine with scientific certainty whether this stems from:
* A software/pipeline defect (display inversion, API translation error, wrong checkpoint, preprocessing mismatch, faulty calibration, or silent fallback), or
* Genuine neural network behavior (out-of-distribution domain shift, social media recompression, and aspect-ratio letterbox priors).

---

## 2. Checkpoint Verification

Direct inspection of the active FastAPI backend service confirms:

| Parameter | Expected Specification | Verified Runtime Value | Status |
| :--- | :--- | :--- | :---: |
| **Checkpoint Path** | `models/v3_final_candidate/best_model.pt` | `C:\SignalScope\models\v3_final_candidate\best_model.pt` | **VERIFIED** |
| **File Exists** | `True` | `True` | **VERIFIED** |
| **File Size** | 41.6 MB | 41,667,083 bytes | **VERIFIED** |
| **Model Architecture** | `SignalScopeClassifier (EfficientNet-B0)` | `SignalScopeClassifier (EfficientNet-B0)` | **VERIFIED** |
| **Classifier Head** | `Linear(1280, 2)` | `Linear(in_features=1280, out_features=2, bias=True)` | **VERIFIED** |
| **Number of Output Classes** | 2 | 2 | **VERIFIED** |
| **Class Mapping** | 0 = REAL, 1 = SYNTHETIC | `{0: 'REAL', 1: 'AI-GENERATED'}` | **VERIFIED** |
| **Model Mode** | `model.eval()` (`training == False`) | `model.training == False` | **VERIFIED** |

**Finding:** The active server is loading the authentic V3 trained checkpoint. V1 and V2 checkpoints are untouched in their separate directories.

---

## 3. Class Mapping Audit Across the Stack

The class mapping was traced from dataset storage through to frontend DOM rendering:

1. **`src/training/dataset.py:40`**: `Labels: 0 = Real, 1 = Synthetic`.
2. **Loss Function**: `nn.CrossEntropyLoss` with targets $0 = \text{Real}, 1 = \text{Synthetic}$.
3. **Model Logits Vector**: Index 0 represents $z_{\text{real}}$, Index 1 represents $z_{\text{synth}}$.
4. **`src/explainability/evidence_test.py:31, 116`**: `CLASS_NAMES = {0: "REAL", 1: "AI-GENERATED"}`, `pred_class = 1 if prob_synth > 0.50 else 0`.
5. **`backend/service.py:158`**: `label = "REAL" if pred_class == 0 else "AI-GENERATED"`.
6. **API Response**: `{"label": ..., "probabilities": {"real": ..., "synthetic": ...}}`.
7. **`frontend/app.js:322`**: `const isReal = data.label === 'REAL'`. Displays `REAL PHOTOGRAPHY` if true, `AI-GENERATED` if false.

**Finding:** There is **zero class inversion** anywhere in the system.

---

## 4. Direct PyTorch Model Results (Frontend Completely Bypassed)

Direct PyTorch inference was executed in pure Python, bypassing JavaScript, HTML, and network layers:

| Image | Description | Raw Logit 0 (Real) | Raw Logit 1 (Synth) | Softmax Real | Softmax Synth | Calibrated Real | Calibrated Synth | Final Class | Final Label |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `example_2_stable_diffusion.png` | SD 1.5 Benchmark | -1.926349 | +1.695119 | 2.60% | 97.40% | 2.79% | **97.21%** | 1 | **AI-GENERATED** |
| `example_4_adm_diffusion.png` | ADM Benchmark | -2.949321 | +2.716599 | 0.35% | 99.65% | 0.38% | **99.62%** | 1 | **AI-GENERATED** |
| `WhatsApp ... 02.56.08.jpeg` | User AI Test 1 | +1.988389 | -2.119981 | 98.38% | 1.62% | **98.25%** | 1.75% | 0 | **REAL** |
| `WhatsApp ... 02.56.40.jpeg` | User AI Test 2 | +3.763183 | -3.863288 | 99.95% | 0.05% | **99.94%** | 0.06% | 0 | **REAL** |
| `WhatsApp ... 02.58.54.jpeg` | User AI Test 3 | +5.164462 | -5.478467 | 99.998% | 0.002% | **100.00%** | 0.00% | 0 | **REAL** |

**Crucial Diagnostic Finding:**
* For pristine benchmark AI images (`example_2`, `example_4`), the raw model logits are $z_1 \gg z_0$, producing **`AI-GENERATED`** ($97.2\%$ and $99.6\%$).
* For the user's WhatsApp-shared AI images (`02.56.08`, `02.56.40`, `02.58.54`), the raw PyTorch model logits are $z_0 \gg z_1$ ($+1.99$ to $+5.16$ vs. $-2.12$ to $-5.48$).
* **The direct PyTorch model weights themselves output high Real logits on these images.** This is not a frontend or API bug.

---

## 5. API vs. Direct Model Verification

Submitting the exact same images through `POST /predict` yields exact parity:

### Complete API JSON Output for `example_2_stable_diffusion.png`:
```json
{
  "label": "AI-GENERATED",
  "confidence": 0.9721,
  "confidence_percent": "97.2%",
  "confidence_band": "HIGH CONFIDENCE",
  "confidence_tier_description": "Strong statistical evidence from multi-scale feature representations.",
  "model": "SignalScope V3 EfficientNet-B0",
  "inference_time": 0.285,
  "image_width": 512,
  "image_height": 512,
  "explanation_available": true,
  "probabilities": {
    "real": 0.0279,
    "synthetic": 0.9721
  },
  "target_layer": "backbone.features[8] (Conv2dNormActivation)"
}
```

### Complete API JSON Output for `WhatsApp Image 2026-09-15 at 02.56.40.jpeg`:
```json
{
  "label": "REAL",
  "confidence": 0.9994,
  "confidence_percent": "99.9%",
  "confidence_band": "HIGH CONFIDENCE",
  "confidence_tier_description": "Strong statistical evidence from multi-scale feature representations.",
  "model": "SignalScope V3 EfficientNet-B0",
  "inference_time": 0.312,
  "image_width": 1179,
  "image_height": 1414,
  "explanation_available": true,
  "probabilities": {
    "real": 0.9994,
    "synthetic": 0.0006
  },
  "target_layer": "backbone.features[8] (Conv2dNormActivation)"
}
```

| Source | `example_2_stable_diffusion.png` | `WhatsApp Image ... 02.56.40.jpeg` | Parity Status |
| :--- | :---: | :---: | :---: |
| **Direct PyTorch Model** | P(Synth) = 97.21% $\to$ AI-GENERATED | P(Real) = 99.94% $\to$ REAL | Baseline |
| **FastAPI Backend (`/predict`)**| P(Synth) = 97.21% $\to$ AI-GENERATED | P(Real) = 99.94% $\to$ REAL | **EXACT MATCH** |
| **Frontend UI Rendering** | Displays 97.2% AI-GENERATED | Displays 99.9% REAL | **EXACT MATCH** |

---

## 6. Frontend Response Parsing Audit

In [`frontend/app.js`](file:///C:/SignalScope/frontend/app.js) lines 322–345:
* `const isReal = data.label === 'REAL';`
* Card classes, titles, badges, and progress bar widths use `data.probabilities.synthetic` and `data.probabilities.real` directly.
* There is no intermediate thresholding, inversion, or hardcoded override in the frontend.

---

## 7. Preprocessing Verification

* The uploaded image bytes are received via multipart stream, verified with PIL, converted to RGB, and processed through `LetterboxTransform(224, fill=(128, 128, 128))`.
* ImageNet normalization is applied: Mean $[0.485, 0.456, 0.406]$, Std $[0.229, 0.224, 0.225]$.
* The tensor shape is `[1, 3, 224, 224]`, identical to training.
* The frontend transmits the raw file byte-for-byte; no client-side resizing or re-encoding occurs.

---

## 8. Calibration Verification

* Optimal temperature: $T = 1.0195$ (loaded from `temperature_calibration.json`, Expected Calibration Error $= 0.51\%$).
* **Before Calibration ($T = 1.0$):** $P(\text{Synth}) = 97.3953\%$, $P(\text{Real}) = 2.6047\%$.
* **After Calibration ($T = 1.0195$):** $P(\text{Synth}) = 97.2137\%$, $P(\text{Real}) = 2.7863\%$.
* Because $T > 0$ is a monotonic scaling transformation, it does not alter the argmax or move the $0.50$ decision boundary. Calibration does not cause the misclassifications.

---

## 9. Threshold Verification

* Decision rule in `evidence_test.py:116`:
  $$\text{pred\_class} = 1 \text{ if } P(\text{Synthetic}) > 0.50 \text{ else } 0$$
* For `example_2`: $P(\text{Synth}) = 0.9721 > 0.50 \implies \text{Class } 1$ (`AI-GENERATED`).
* For `WhatsApp 02.56.08`: $P(\text{Synth}) = 0.0175 \le 0.50 \implies \text{Class } 0$ (`REAL`).
* For `WhatsApp 02.56.40`: $P(\text{Synth}) = 0.0006 \le 0.50 \implies \text{Class } 0$ (`REAL`).
* The threshold is not the cause of the issue: the model's raw probability for synthetic is $< 2\%$, far below any reasonable threshold.

---

## 10. Fallback & Caching Investigation

* **Grep Audit:** Searched `backend/`, `frontend/`, and `src/` for silent fallbacks, cached responses, or hardcoded defaults.
* **Results:**
  - Zero hardcoded `REAL` prediction fallbacks exist.
  - If image decoding fails, FastAPI returns HTTP 400 (`Bad Request`).
  - If inference fails, FastAPI returns HTTP 500 (`Internal Server Error`), and the frontend displays an error toast.
  - Every upload writes the raw image to `data/debug_last_upload.png` and executes fresh forward inference.

---

## 11. Root Cause Analysis

```
===================================================================
ROOT CAUSE:
Pipeline is functioning correctly; observed AI -> REAL predictions
are genuine V3 model false negatives.
===================================================================
```

No software bug, class inversion, API translation error, or caching defect exists. The false negatives are driven by three fundamental computer vision and machine learning dynamics:

### A. Training Dataset Aspect-Ratio & Grey Padding Bias
* In the V3 training dataset (14,759 images), **100% of non-square images requiring grey letterbox padding were authentic ImageNet camera photographs** (`real_imagenet`, 1,989 samples, $500 \times 375$).
* In contrast, **100% of synthetic training images were square** (CIFAKE $32 \times 32$, SD 1.4/1.5 $512 \times 512$, ADM $256 \times 256$, Wukong $512 \times 512$).
* During backpropagation across unfrozen blocks 6, 7, and 8, the convolutional filters learned that neutral grey border padding $(128, 128, 128)$ is an overwhelming statistical prior for Class 0 (`REAL`).
* Controlled proof: Taking a square Stable Diffusion image that scores $97.2\%$ AI and artificially adding 28px of neutral grey padding flips its prediction to $99.2\%$ REAL. Non-square portrait AI images ($1179 \times 1414$, $738 \times 1600$) receive this padding and activate the Real class.

### B. Social Media Lossy Recompression (WhatsApp MozJPEG)
* The failure images were all downloaded through WhatsApp (`WhatsApp Image 2026-09-15 at 02.56.xx.jpeg`).
* WhatsApp strips all EXIF metadata and applies aggressive lossy Discrete Cosine Transform (DCT) quantization with 4:2:0 chroma subsampling.
* This lossy compression wipes out the micro-structural generative grid frequencies and high-frequency upsampling peaks that deep CNNs rely on to detect diffusion models, replacing them with camera-like JPEG blocking noise.

### C. Night Sky / Astrophotography Entanglement
* Sample `02.56.40.jpeg` has a very low mean RGB of $[20.9, 22.0, 27.3]$ (deep dark palette).
* In V3, 105 real astrophotography images (`real_night_sky`) were added to resolve the V2 cosmic misclassification. The network learned a strong semantic association between deep shadow/night scenes and Class 0 (`REAL`).

---

## 12. Fix Applied

**Status:** `NO SOFTWARE FIX REQUIRED OR APPLIED`
* No software bug existed to fix.
* Per instructions, no weights were modified, no checkpoints were overwritten, and no artificial heuristics were introduced.

---

## 13. Retest Benchmark Results (5 Known AI + 5 Known Real)

Tested end-to-end against the live application:

| Image Filename | Ground Truth | Real % | Synthetic % | Prediction | Forensic Note |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`example_2_stable_diffusion.png`** | **AI-GENERATED** | 2.8% | **97.2%** | **AI-GENERATED** | Clean square diffusion (Correct) |
| **`example_3_wukong_diffusion.png`** | **AI-GENERATED** | 48.6% | **51.4%** | **AI-GENERATED** | Zero-shot unseen holdout (Correct) |
| **`example_4_adm_diffusion.png`** | **AI-GENERATED** | 0.4% | **99.6%** | **AI-GENERATED** | Guided diffusion (Correct) |
| **`WhatsApp ... 02.56.08.jpeg`** | **AI-GENERATED** | 98.2% | 1.8% | **REAL** | Recompressed non-square (Model FN) |
| **`WhatsApp ... 02.56.40.jpeg`** | **AI-GENERATED** | 99.9% | 0.1% | **REAL** | Recompressed dark scene (Model FN) |
| **`example_1_real_nature.jpg`** | **REAL** | **100.0%** | 0.0% | **REAL** | ImageNet camera photo (Correct) |
| **`real_astro_night_sky_0011.jpg`** | **REAL** | **100.0%** | 0.0% | **REAL** | Deep astrophotography (Correct) |
| **`real_astro_night_sky_0000.jpg`** | **REAL** | **100.0%** | 0.0% | **REAL** | Star clusters & night sky (Correct) |
| **`WhatsApp ... 02.08.20.jpeg`** | **REAL** | **99.9%** | 0.1% | **REAL** | Authentic gym camera photo (Correct) |
| **`WhatsApp ... 02.08.28.jpeg`** | **REAL** | **100.0%** | 0.0% | **REAL** | Authentic polo camera photo (Correct) |

---

## 14. Model Weights & Checkpoint Integrity

* **`models/v3_final_candidate/best_model.pt`**: 100% UNTOUCHED, FROZEN, AND PRESERVED.
* **`models/v1_baseline/best_model.pt`**: 100% UNTOUCHED.
* **`models/v2_expanded_data/best_model.pt`**: 100% UNTOUCHED.
* **Decision Threshold:** Preserved at $\tau = 0.50$.
* **Git Status:** Clean, no commits or pushes made.
