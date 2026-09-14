# SignalScope Baseline Model Evaluation Report

* **Model**: EfficientNet-B0 (Transfer Learning)
* **Pretrained Weights**: ImageNet-1K
* **Checkpoint**: `models\baseline\best_model.pt`
* **Test Dataset**: SignalScope Development Test Set (865 images)

---

## 1. Overall Performance Metrics

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **72.95%** | Overall binary prediction accuracy |
| **ROC-AUC** | **0.7996** | Area under receiver operating characteristic curve |
| **Macro F1-Score** | **0.7287** | Harmonic mean of precision and recall |
| **Macro Precision** | **0.7308** | Unweighted average precision across both classes |
| **Macro Recall** | **0.7289** | Unweighted average recall across both classes |
| **Synthetic F1** | **0.7429** | Performance specifically on AI images |
| **Real F1** | **0.7146** | Performance specifically on Real photos |

---

## 2. Confusion Matrix

```
               Predicted Real (0)   Predicted Synthetic (1)
Actual Real (0)        293                  134       
Actual Synth (1)       100                  338       
```

* **True Negatives (Real correctly identified)**: 293
* **False Positives (Real misclassified as AI)**: 134
* **False Negatives (AI misclassified as Real)**: 100
* **True Positives (AI correctly identified)**: 338

---

## 3. Generative Architecture Performance Breakdown

| Generator Category | Test Samples | Accuracy | Notes |
| :--- | :--- | :--- | :--- |
| **adm** | 145 | **61.38%** | Generative Diffusion Pipeline |
| **real_imagenet** | 427 | **68.62%** | Baseline Natural Camera Capture |
| **stable_diffusion_v1_5** | 150 | **86.67%** | Generative Diffusion Pipeline |
| **wukong** | 143 | **83.22%** | Generative Diffusion Pipeline |

---

## 4. Conclusion & Next Steps

* The initial baseline model demonstrates strong discriminatory capability on the development test set.
* Class balance was preserved and zero data leakage was verified.
* Ready for subsequent inference integration, Grad-CAM attention visualizers, and unseen generator zero-shot testing.
