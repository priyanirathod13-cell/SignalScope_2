# SignalScope: Final System Evaluation & Polish Report (Step 7 Complete)

## 1. Executive Summary
**SignalScope** is an end-to-end, production-ready AI image authenticity detection and visual explainability system developed for the **Smart India Hackathon (SIH) 2026**. 

The system leverages a transfer-learning convolutional backbone (**EfficientNet-B0**) paired with **Grad-CAM** gradient-weighted feature attribution to identify generative synthesis signatures and provide human-interpretable visual evidence.

---

## 2. Verified System Architecture

| Component | Technology | Path / Location | Status |
| :--- | :--- | :--- | :--- |
| **Dataset Infrastructure** | 5,764 Balanced Real/Synthetic Images | `data/splits/` (`train`, `val`, `test`) | Verified |
| **Dataset Metadata** | SHA-256 Hashes, Zero-Leakage Split Manifest | `data/metadata/dataset_metadata.csv` | Verified |
| **Model Checkpoint** | Pretrained ImageNet-1K + Custom Linear Head | `models/baseline/best_model.pt` (19.7 MB) | Verified |
| **Explainability Engine** | Grad-CAM + Pure NumPy/PIL Colormapping | `src/explainability/gradcam.py` | Verified |
| **Robustness Suite** | 7 Controlled Real-World Degradation Tests | `src/evaluation/robustness.py` | Verified |
| **Backend API** | High-performance FastAPI Service | `backend/main.py` & `backend/service.py` | Verified |
| **Frontend UI** | Modern Responsive Cybersecurity Dashboard | `frontend/index.html`, `style.css`, `app.js` | Verified |
| **Example Gallery** | 4 Preloaded Forensic Verification Samples | `data/examples/` | Verified |

---

## 3. Official Benchmark Metrics (Development Test Set: 865 Images)

Evaluated strictly on the development test split (427 Real ImageNet, 150 Stable Diffusion v1.5, 145 ADM, 143 Wukong) with zero data leakage:

| Metric | Score | Forensic Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **72.95%** | Overall binary authenticity accuracy |
| **ROC-AUC** | **0.7996** | Area under receiver operating characteristic curve |
| **Macro-F1** | **0.7287** | Balanced harmonic mean across Real and Synthetic classes |
| **Macro-Precision** | **0.7308** | Average precision across classes |
| **Macro-Recall** | **0.7289** | Average recall across classes |
| **Synthetic F1** | **0.7429** | Specific performance on AI-generated images |
| **Real F1** | **0.7146** | Specific performance on authentic camera captures |

### Confusion Matrix
```
               Predicted Real (0)   Predicted Synthetic (1)
Actual Real (0)        293                  134       
Actual Synth (1)       100                  338       
```
* **True Negatives (TN - Real correctly predicted)**: 293
* **False Positives (FP - Real flagged as AI)**: 134
* **False Negatives (FN - AI missed as Real)**: 100
* **True Positives (TP - AI correctly detected)**: 338

### Per-Generator Detection Breakdown
| Generative Category | Test Images | Detection Accuracy | Architecture Notes |
| :--- | :--- | :--- | :--- |
| **Stable Diffusion v1.5** | 150 | **86.67%** | Latent Diffusion Model (text-to-image) |
| **Wukong Diffusion** | 143 | **83.22%** | Large Multilingual Diffusion Model |
| **Real Photography** | 427 | **68.62%** | Authentic natural camera sensor captures |
| **ADM (Ablated Diffusion)** | 145 | **61.38%** | Guided Pixel-Space Diffusion Model |

---

## 4. Live API & Frontend Verification

### Real vs. AI Live Inferences Recorded
1. **Authentic Camera Capture**:
   - Sample: `example_1_real_nature.jpg`
   - Predicted Class: **`REAL`**
   - Confidence: **`88.8%`** (Real: 88.8%, Synthetic: 11.2%)
   - Inference Latency: **`4.480 s`** (4480 ms)
2. **AI-Generated Synthesis**:
   - Sample: `example_2_stable_diffusion.png`
   - Predicted Class: **`AI-GENERATED`**
   - Confidence: **`82.1%`** (Synthetic: 82.1%, Real: 17.9%)
   - Inference Latency: **`1.294 s`** (1294 ms)

### Validation & Edge Cases Passed
* **Unsupported Format**: `.txt` file correctly returned `HTTP 400 Bad Request` (`Unsupported file format`).
* **Corrupt Image Data**: Broken byte stream returned `HTTP 400 Bad Request` (`Corrupted or unreadable image file`).
* **Oversized Upload**: 16 MB payload correctly blocked with `HTTP 400 Bad Request` (`File size exceeds maximum allowed limit of 15 MB`).

---

## 5. Robustness & Generalization Summary

| Evaluation Condition | Accuracy | Macro-F1 | ROC-AUC | Degradation from Baseline (Δ AUC) |
| :--- | :--- | :--- | :--- | :--- |
| **Clean Baseline** | **72.95%** | **0.7287** | **0.7996** | — (Reference) |
| **JPEG Mild (Q=75)** | 73.06% | 0.7281 | 0.8007 | +0.0011 (Negligible) |
| **JPEG Strong (Q=50)** | 69.83% | 0.6935 | 0.7772 | -0.0224 (Moderate) |
| **Resized (50% Down/Up)** | 70.40% | 0.6932 | 0.8076 | +0.0080 (Ranking Preserved) |
| **Screenshot-like** | 68.55% | 0.6810 | 0.7569 | -0.0427 (Largest impact) |
| **Brightness (+10%)** | 72.14% | 0.7205 | 0.7995 | -0.0001 (Photometrically invariant) |
| **Contrast (+15%)** | 72.25% | 0.7217 | 0.7997 | +0.0001 (Photometrically invariant) |

---

## 6. How to Run SignalScope

To run the complete system with the interactive web frontend:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
Open your browser at:
```
http://127.0.0.1:8000/
```
