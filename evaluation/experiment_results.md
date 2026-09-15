# SignalScope Experiment Results

## Overview

This file records the results of the SignalScope model experiments.

The external test set is kept separate from training and is used only to evaluate generalization.

---

# V1 — ResNet18 Baseline

## Training

- Architecture: ResNet18
- Pretrained backbone: Yes
- Backbone training: Frozen except classifier
- Input size: 224 × 224
- Epochs: 10
- Dataset:
  - Real: 1,000
  - AI: 875
- Train/Validation/Test split: 80/10/10

## Internal Test Results

- Accuracy: 92.02%
- Precision: 97.75%
- Recall: 87.00%
- Macro-F1: 92.02%
- ROC-AUC: 0.9744

## Confusion Matrix

```text
[[86,  2],
 [13, 87]]