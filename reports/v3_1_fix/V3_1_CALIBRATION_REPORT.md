# SignalScope V3.1 Temperature Scaling Calibration Report

## 1. Overview
- **Model Checkpoint**: `models/v3_1_fix/best_model.pt`
- **Calibration Split**: `data/v3_1/splits/val.csv` (1,572 validation samples)
- **Method**: Scalar Temperature Scaling via L-BFGS optimization on cross-entropy loss.
- **Optimal Temperature ($T$)**: `1.0923`

---

## 2. Quantitative Calibration Results

| Metric | Uncalibrated ($T=1.0000$) | Calibrated ($T=1.0923$) | Improvement |
| :--- | :--- | :--- | :--- |
| **Negative Log-Likelihood (NLL)** | 0.3348 | **0.3291** | 0.0057 lower |
| **Expected Calibration Error (ECE)** | 4.30% | **3.62%** | **15.9% reduction** |

---

## 3. Operational Confidence Bands
Calibrated probabilities are mapped to the following operational decision bands:

- **$\ge 0.85$**: High-Confidence Synthetic (Autonomous flagging / watermark detection)
- **$0.60 - 0.85$**: Moderate Synthetic (Review recommended)
- **$0.40 - 0.60$**: Ambiguous / Uncertain (Borderline probability, human verification required)
- **$0.15 - 0.40$**: Moderate Real (Likely natural photography)
- **$< 0.15$**: High-Confidence Real (Unambiguous camera capture)

---

## 4. Reliability Diagram Bins

| Bin Interval | Count | Mean Confidence | Observed Accuracy | Calibration Error |
| :--- | :--- | :--- | :--- | :--- |
| 0.0-0.1 | 442 | 2.2% | 2.9% | 0.73% |
| 0.1-0.2 | 89 | 14.2% | 11.2% | 2.96% |
| 0.2-0.3 | 78 | 25.1% | 25.6% | 0.57% |
| 0.3-0.4 | 50 | 34.5% | 32.0% | 2.46% |
| 0.4-0.5 | 62 | 45.5% | 37.1% | 8.45% |
| 0.5-0.6 | 54 | 55.3% | 44.4% | 10.87% |
| 0.6-0.7 | 68 | 64.8% | 57.4% | 7.44% |
| 0.7-0.8 | 83 | 75.1% | 67.5% | 7.61% |
| 0.8-0.9 | 133 | 85.9% | 75.9% | 9.95% |
| 0.9-1.0 | 513 | 97.0% | 94.3% | 2.66% |
