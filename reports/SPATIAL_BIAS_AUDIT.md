# SignalScope V3 Spatial Bias & Shortcut Audit Report

**Model Evaluated:** `models/v3_final_candidate/best_model.pt` (Frozen Production V3)  
**Evaluation Protocol:** Quantitative Occlusion & Invariance Perturbation Testing  
**Sample Population:** $N = 120$ balanced evaluation images (60 Real, 60 Synthetic)  
**Calibration Temperature Applied:** $T = 1.0195$  
**Audit Timestamp:** September 15, 2026

---

## 1. Executive Summary & Verdict

### **AUDIT VERDICT: HIGH SPATIAL BIAS**

### Summary of Measured Findings:
1. **Peripheral Dominance over Center (Ratio = 0.59x - 0.69x):** In standard object-centric vision, masking the central subject produces the highest classification shift. In SignalScope V3, the opposite occurs: peripheral regions (Left, Right, Top, Bottom) induce significantly larger probability swings ($|\Delta P| = 9.95\% - 15.57\%$) than the Center ($|\Delta P| = 6.90\%$), yielding an inverted Center-to-Side sensitivity ratio of **0.69x** (and **0.59x** Center-to-Periphery).
2. **Extreme Decision Flip Disparity (up to 4.2x):** Masking peripheral margins inverts the model's decision at dramatically higher frequencies than masking the center:
   - On Synthetic images, occluding the **Bottom** flips the prediction to Real in **28.3%** of cases, and occluding the **Top** flips it in **21.7%** of cases, compared to only **6.7%** for the **Center**.
   - Overall across all images, Bottom occlusion flips **15.8%** of predictions vs. **5.0%** for Center occlusion.
3. **Bilateral Symmetry Maintained (100.0%):** Horizontal reflection yields **100.00% classification consistency** and near-identical sensitivity (Left $|\Delta P| = 10.82\%$, Right $|\Delta P| = 9.09\%$), demonstrating that the spatial bias is non-lateral (primarily vertical and peripheral).

---

## 2. Quantitative Perturbation Measurement Matrix

### 2.1 Mean Absolute Probability Shift ($|\Delta P|$)
Each mask region covers **exactly 25.0% (12,544 pixels)** of the $224 \times 224$ canvas filled with neutral gray `(128, 128, 128)`.

| Evaluation Cohort | Sample Count ($N$) | Center ($|\Delta P|$) | Left ($|\Delta P|$) | Right ($|\Delta P|$) | Top ($|\Delta P|$) | Bottom ($|\Delta P|$) | Lateral Sides Mean | Full Periphery Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall Population** | 120 | **6.90%** | 10.82% | 9.09% | 11.47% | **15.57%** | 9.95% | 11.74% |
| **REAL Photography** | 60 | **3.68%** | 4.89% | 3.81% | 4.15% | **4.67%** | 4.35% | 4.38% |
| **SYNTHETIC Diffusion** | 60 | **10.11%** | 16.76% | 14.36% | 18.78% | **26.48%** | 15.56% | 19.10% |
| **Square (No Padding)** | 93 | **8.63%** | 13.59% | 11.22% | 14.30% | **19.58%** | 12.41% | 14.67% |
| **Non-Square (Letterboxed)** | 27 | **0.92%** | 1.28% | 1.74% | 1.71% | **1.77%** | 1.51% | 1.62% |

---

## 3. Decision Boundary Flip Rates & Sensitivity Ratios

### 3.1 Percentage of Predictions Inverted by 25% Masking
Measures the vulnerability of the decision boundary ($\tau = 0.50$) to regional occlusion:

| Cohort | Center Flip % | Left Flip % | Right Flip % | Top Flip % | Bottom Flip % | Horizontal Flip Consistency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall Population** | **5.0%** | 10.0% | 8.3% | 11.7% | **15.8%** | **100.00%** |
| **REAL Photography** | **3.3%** | 3.3% | 1.7% | 1.7% | **3.3%** | **100.00%** |
| **SYNTHETIC Diffusion** | **6.7%** | 16.7% | 15.0% | 21.7% | **28.3%** | **100.00%** |
| **Square (No Padding)** | **6.5%** | 12.9% | 10.8% | 15.1% | **20.4%** | **100.00%** |
| **Non-Square (Letterboxed)**| **0.0%** | 0.0% | 0.0% | 0.0% | **0.0%** | **100.00%** |

### 3.2 Center-vs-Side & Center-vs-Periphery Sensitivity Ratios
- **Overall Center / Lateral Sides Ratio:** $\frac{6.90\%}{9.95\%} =$ **0.69x**
- **Overall Center / Full Periphery Ratio:** $\frac{6.90\%}{11.74\%} =$ **0.59x**
- **Synthetic Cohort Center / Lateral Sides Ratio:** $\frac{10.11\%}{15.56\%} =$ **0.65x**
- **Synthetic Cohort Center / Bottom Ratio:** $\frac{10.11\%}{26.48\%} =$ **0.38x**
- **Square Images Center / Full Periphery Ratio:** $\frac{8.63\%}{14.67\%} =$ **0.59x**

---

## 4. Grad-CAM Qualitative Cross-Inspection

Inspection of convolutional feature attribution maps from `backbone.features[8]` confirms the quantitative findings:
1. **Peripheral Attribution Clustering:** On synthetic diffusion samples (e.g. SD 1.5 and ADM), Grad-CAM heatmaps routinely exhibit strong activation clusters along the bottom margin and top corners, reflecting high-frequency generative boundary transitions rather than semantic facial/object anatomy.
2. **Bottom-Edge Sensitivity:** The prominent sensitivity to the Bottom region ($|\Delta P| = 26.48\%$ on synthetic images) corresponds to generator-specific background termination artifacts and watermark-adjacent diffusion boundary noise present in training sets.
3. **Padded Border Transitions:** On non-square inputs letterboxed into $224 \times 224$, Grad-CAM heatmaps highlight the interface between neutral grey borders and visual content, corroborating the lateral border shortcut verified during earlier root-cause analysis.

---

## 5. Measured Evidence Explanation & Final Assessment

### Verdict: **HIGH SPATIAL BIAS**

### Empirical Rationale:
1. **Inverted Center-to-Periphery Gradient:** In unbiased computer vision classifiers, the central 25% of the frame accounts for the vast majority of class attribution. In SignalScope V3, peripheral occlusion exerts **1.70x greater overall impact** ($11.74\%$ vs. $6.90\%$) and **1.89x greater impact on synthetic images** ($19.10\%$ vs. $10.11\%$) than central occlusion.
2. **Severe Bottom-Region Reliance:** The bottom 25% of the canvas is the single most influential spatial region in the entire model. Occluding the bottom 25% inverts more than 1 in 4 synthetic predictions (**28.3% flip rate**), a 4.2x amplification compared to center occlusion (**6.7% flip rate**).
3. **Operational Recommendation & Governance:**
   - While heuristic dual-inspection (central crop fusion) was investigated, strict submission governance mandates locking the frozen production V3 letterbox pipeline without ad-hoc post-processing heuristics.
   - Consequently, this High Spatial Bias is fully documented as an inherent, transparent empirical limitation of SignalScope V3.
