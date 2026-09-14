# SignalScope V3 Temperature Scaling Calibration Report

* **Model:** SignalScope V3 EfficientNet-B0
* **Validation Samples:** 3,161
* **Optimal Temperature (T):** **1.0195**
* **Pre-Calibration ECE:** 0.65%
* **Post-Calibration ECE:** **0.51%** (ECE Improvement: **-0.14%**)

---

## 1. Reliability & Calibration Bins

| Confidence Bin | Samples | Observed Synthetic Frequency | Mean Confidence | Calibration Error |
| :---: | :---: | :---: | :---: | :---: |
| 0.0-0.1 | 1444 | 1.1% | 0.9% | 0.20% |
| 0.1-0.2 | 82 | 14.6% | 13.9% | 0.75% |
| 0.2-0.3 | 54 | 31.5% | 24.6% | 6.90% |
| 0.3-0.4 | 28 | 25.0% | 34.7% | 9.72% |
| 0.4-0.5 | 33 | 39.4% | 43.7% | 4.35% |
| 0.5-0.6 | 26 | 50.0% | 54.7% | 4.74% |
| 0.6-0.7 | 36 | 66.7% | 65.6% | 1.02% |
| 0.7-0.8 | 66 | 75.8% | 75.6% | 0.14% |
| 0.8-0.9 | 82 | 82.9% | 85.4% | 2.50% |
| 0.9-1.0 | 1310 | 98.7% | 98.8% | 0.08% |

---

## 2. Calibrated Confidence Bands

* **HIGH CONFIDENCE (>=90% Synthetic or <=10% Real):** Extremely reliable automated detection.
* **MODERATE CONFIDENCE (70-90% Synthetic or 10-30% Real):** Reliable identification; forensic explainability inspection advised.
* **BORDERLINE / INCONCLUSIVE (30-70%):** Model uncertainty elevated; manual forensic evaluation required.
