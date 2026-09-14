# SignalScope Explainability Framework: Forensic Grad-CAM Interpretability

## 1. Overview
The SignalScope explainability subsystem provides visual attribution maps for the trained deep learning authenticity detector (`SignalScopeClassifier` on `EfficientNet-B0`). By leveraging **Grad-CAM** (Gradient-weighted Class Activation Mapping), the system produces coarse 2D heatmaps indicating which spatial regions of an image contributed most strongly to the model's authenticity decision (0 = REAL, 1 = AI-GENERATED / SYNTHETIC).

---

## 2. Technical Architecture
* **Methodology**: Gradient-weighted Class Activation Mapping (Grad-CAM).
* **Target Layer**: `backbone.features[8]` (`Conv2dNormActivation`), the final convolutional projection block of EfficientNet-B0 (1,280 channels, $7 \times 7$ feature grid for $224 \times 224$ input).
* **Gradient Pooling**: Global Average Pooling over spatial dimensions ($\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}$).
* **Rectified Linear Activation**: $\text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$ isolates features whose presence positively supports the target class score.
* **Upsampling**: Bilinear interpolation from $7 \times 7$ back to the original image resolution $(W_{\text{orig}} \times H_{\text{orig}})$.
* **Colormapping & Blending**: Zero-dependency pure NumPy/Pillow JET colormap (Dark Blue $\to$ Cyan $\to$ Green $\to$ Yellow $\to$ Vivid Red) alpha-blended with the original RGB image ($\alpha = 0.5$).

---

## 3. Responsible AI: Interpretation Guidelines
> [!IMPORTANT]
> **What the Heatmap Actually Shows:**
> - The Grad-CAM heatmap illustrates the **spatial receptive fields in the convolutional feature map that activated most strongly in favor of the model's prediction**.
> - Hot regions (red, orange, yellow) indicate high gradient $\times$ activation weight for the target class logit.
> - Cold regions (blue, dark blue) indicate areas that had minimal or zero positive influence on the target class score.

> [!WARNING]
> **Why the Heatmap is NOT Ground-Truth Proof:**
> 1. **Interpretability Tool, Not a Pixel-Level Mask**: Grad-CAM explains the *model's internal reasoning*, not the absolute physical reality of image synthesis. It does not demarcate exact forged pixels or local diffusion inpainting masks.
> 2. **Coarse Spatial Resolution**: The final convolutional feature map of EfficientNet-B0 operates at a $7 \times 7$ grid. Upsampling to full resolution produces smooth, blob-like regions rather than pixel-sharp boundaries.
> 3. **Correlation vs. Causation**: High attribution in a region means the model attended to features there (e.g., high-frequency brushstrokes, hair strands, text boundaries, or lighting anomalies), not that the entire highlighted object is synthetic.
> 4. **No Replacement for Expert Forensic Analysis**: Heatmaps must be interpreted alongside frequency-domain analysis (e.g. FFT/DCT), metadata checks, and domain-expert inspection.

---

## 4. Known Failure Modes & Limitations
1. **High-Contrast False Salience**: Intrinsic high-frequency edges (such as text, watermarks, fur, foliage, or complex textures) often yield strong convolutional filter activations, which can result in elevated attribution even on authentic photographs.
2. **Global Artifact Diffusion**: Modern diffusion models (Stable Diffusion, ADM, Wukong) inject structural artifacts and high-frequency noise globally throughout the entire canvas. A localized Grad-CAM highlight reflects where the CNN picked up the strongest signal, not that other parts of the image are pristine.
3. **JPEG Compression Shifts**: Heavy compression or re-saving can alter high-frequency DCT coefficients, occasionally causing Grad-CAM focus to shift toward compression macroblock boundaries.
4. **Boundary Uncertainties**: On borderline decisions (confidence $50\% - 60\%$), the gradient magnitude is small, and small perturbations in the image can cause noticeable shifts in the heatmap topology.

---

## 5. Artifact Reference
Sample attribution artifacts stored in this directory:
| Target Category | Sample Image | Predicted Class | Confidence | Overlay Artifact |
| :--- | :--- | :--- | :--- | :--- |
| **Real ImageNet** | `real_imagenet_adm_n01498041_8374.JPEG` | `REAL` | **88.79%** | `real_sample_real_imagenet_adm_n01498041_8374_overlay.png` |
| **Stable Diffusion v1.5** | `synthetic_stable_diffusion_v1_5_012_sdv5_00190.png` | `AI-GENERATED` | **82.13%** | `synthetic_sd15_synthetic_stable_diffusion_v1_5_012_sdv5_00190_overlay.png` |
| **Wukong** | `synthetic_wukong_0_wukong_image8.png` | `AI-GENERATED` | **92.96%** | `synthetic_wukong_synthetic_wukong_0_wukong_image8_overlay.png` |
| **ADM** | `synthetic_adm_122_adm_71.PNG` | `AI-GENERATED` | **64.59%** | `synthetic_adm_synthetic_adm_122_adm_71_overlay.png` |

---

## 6. CLI Usage
To generate an explanation for any image:
```bash
python -m src.explain --image path/to/image.jpg
```
Optional flags:
* `--output-dir reports/explanations` (destination folder)
* `--model models/baseline/best_model.pt` (checkpoint path)
* `--target-class auto` (or `0` for Real, `1` for Synthetic)
* `--colormap jet` (`jet` or `turbo`)
* `--alpha 0.5` (blend ratio: 0.0 = original only, 1.0 = heatmap only)