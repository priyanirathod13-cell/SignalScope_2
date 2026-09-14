# SignalScope V3: Comprehensive Benchmark & Generalization Report

* **Architecture:** EfficientNet-B0 (Blocks 6-8 Fine-Tuned + Linear Head)
* **Checkpoint:** `models/v3_final_candidate/best_model.pt`
* **Training Duration:** 46.03 minutes on CPU (3 epochs)
* **Optimal Temperature (T):** 1.0195
* **Execution Date:** 2026-09-15

---

## 1. Development Test Set Benchmark (3,165 Images)

Evaluated on the held-out development test set comprising 1,650 Real photographs and 1,515 Synthetic images:

| Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **Accuracy** | **96.02%** | Overall correct classification rate |
| **ROC-AUC** | **0.9936** | Area under ROC curve (99.36% discriminability) |
| **Macro F1-Score** | **0.9602** | Balanced harmonic mean of precision and recall |
| **Macro Precision** | **0.9600** | Average precision across both classes |
| **Macro Recall** | **0.9607** | Average recall across both classes |
| **Synthetic F1-Score** | **0.9590** | Harmonic mean on synthetic images |
| **Real F1-Score** | **0.9613** | Harmonic mean on authentic photographs |
| **False Positive Rate (FPR)** | **5.15%** | Rate of Real images falsely flagged as AI (85 / 1,650) |
| **False Negative Rate (FNR)** | **2.71%** | Rate of AI images missed (41 / 1,515) |
| **CPU Inference Latency** | **35.9 ms** | Average per-sample inference time on CPU |

### Confusion Matrix (Test Set)
```
                   Predicted Real (0)   Predicted Synthetic (1)
Actual Real (0)           1,565                   85
Actual Synth (1)             41                1,474
```

### Per-Generator Breakdown
| Generator / Source Category | Ground Truth Class | Test Samples | Accuracy |
| :--- | :---: | :---: | :---: |
| `stable_diffusion_v1_4` | SYNTHETIC | 1,197 | **97.58%** |
| `stable_diffusion_v1_5` | SYNTHETIC | 150 | **97.33%** |
| `adm` (Ablated Diffusion) | SYNTHETIC | 145 | **94.48%** |
| `real_cifar10` | REAL | 1,200 | **94.67%** |
| `real_imagenet` | REAL | 427 | **95.08%** |
| `real_night_sky` (Curated) | REAL | 23 | **100.00%** |
| `synth_cosmic_fantasy` (Curated) | SYNTHETIC | 23 | **100.00%** |

### Per-Source Benchmark
| Dataset Source | Test Samples | Accuracy |
| :--- | :---: | :---: |
| **CIFAKE** (CIFAR Camera + SD 1.4) | 2,397 | **96.12%** |
| **GenImage** (ImageNet + ADM + SD 1.5) | 722 | **95.43%** |
| **Curated Anti-Shortcut** (Astrophotography vs. Cosmic) | 46 | **100.00%** |

---

## 2. Sacred Unseen-Generator Holdout Benchmark (Wukong)

To measure true zero-shot out-of-distribution generalization without any data contamination, **Wukong** was completely isolated from all training and validation procedures.

| Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **Held-Out Generator** | **Wukong** | Multilingual Latent Diffusion |
| **Total Holdout Samples** | **1,916** | 958 Real Photographs + 958 Wukong Synthetic Images |
| **Holdout Accuracy** | **95.15%** | Zero-shot accuracy on completely unseen generator |
| **Holdout ROC-AUC** | **0.9860** | Zero-shot ROC-AUC on unseen generator |
| **Holdout Macro-F1** | **0.9515** | Balanced F1 score on holdout set |
| **Wukong Detection Accuracy** | **95.82%** | **918 out of 958** Wukong images correctly flagged as AI |
| **Real Verification Accuracy** | **94.47%** | **905 out of 958** Real images correctly verified |

### Confusion Matrix (Sacred Holdout Set)
```
                   Predicted Real (0)   Predicted Synthetic (1)
Actual Real (0)             905                   53
Actual Synth (1)             40                  918
```

---

## 3. Version Progression: V1 vs. V2 vs. V3

| Evaluation Dimension | SignalScope V1 (Baseline) | SignalScope V2 (Expanded Data) | SignalScope V3 (Final Candidate) |
| :--- | :---: | :---: | :---: |
| **Training Samples** | 1,180 | 15,220 | **14,759** (Stratified 70/15/15) |
| **Active Dataset** | 1,686 | 21,743 | **21,085** (0 Cross-Split Collisions) |
| **Candidate Universe** | 1,686 | ~25,000 | **3,451,167** |
| **Backbone State** | Fully Frozen | Fully Frozen | **Blocks 6-8 Fine-Tuned** |
| **Preprocessing** | 224x224 Anisotropic Squash | 224x224 Anisotropic Squash | **Aspect-Ratio Letterboxing** |
| **Development Test Accuracy** | 90.50% | 94.67% | **96.02%** |
| **Development Test ROC-AUC** | 0.9520 | 0.9856 | **0.9936** |
| **Macro F1-Score** | 0.9048 | 0.9467 | **0.9602** |
| **False Positive Rate (FPR)**| 8.80% | 5.86% | **5.15%** |
| **Unseen-Generator Holdout** | None (N/A) | None (Leaked in Train) | **Wukong: 95.82% Acc, 0.9860 AUC** |
| **Anti-Shortcut Balancing** | None | None | **Astrophotography vs. Cosmic Art** |
| **Temperature Calibration** | None (T=1.0) | None (T=1.0) | **T=1.0195 (ECE: 0.51%)** |
| **Explainability Faithfulness** | None | Qualitative Only | **Controlled Occlusion ($\Delta p$ Verified)**|
| **Inference Latency (CPU)** | 42.1 ms | 38.4 ms | **35.9 ms** |
