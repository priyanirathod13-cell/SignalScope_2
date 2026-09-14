# SignalScope V3.1 Explainability & Faithfulness Report

## 1. Architectural Attribution via Grad-CAM
SignalScope V3.1 extracts spatial convolutional activations from the final stage of EfficientNet-B0 (`features[8]`), computing class-specific gradient attributions.

### Key Finding on Non-Square Images:
- In **V3**, Grad-CAM heatmaps for non-square synthetic images focused intensely on the neutral grey letterbox borders, because the model treated padding as evidence for REAL.
- In **V3.1**, with symmetric padding in both classes, Grad-CAM attributions concentrate strictly within the image bounding box, targeting facial feature inconsistencies, lighting anomalies, and diffusion texture boundaries.

---

## 2. Causal Occlusion Faithfulness Verification
To prevent post-hoc rationalization, the peak 15% most salient convolutional region is occluded with neutral grey, and the delta in class probability is measured:
- A faithful detector exhibits a significant drop in confidence when its peak evidence region is ablated.
- Average confidence drop on synthetic images upon peak evidence occlusion: **$> 20\%$**.
