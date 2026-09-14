# SignalScope: Smart India Hackathon (SIH) 2026 Submission Report

## PROJECT
**SignalScope**

## TAGLINE
**Telling Real From Synthetic in the Age of Generative Media**

## PROBLEM
The proliferation of deep learning generative models (such as Latent Diffusion, Guided Diffusion, and large-scale multilingual synthesizers) enables the effortless creation of photorealistic synthetic images. These AI-generated visuals can deceive the human eye, facilitating misinformation, fraud, and identity manipulation. Automated, reliable, and explainable authenticity verification is critically needed to restore trust in digital visual media.

## SOLUTION
SignalScope provides an end-to-end media forensics platform combining:
1. A convolutional neural network detector (**EfficientNet-B0**) fine-tuned on generative diffusion artifact signatures.
2. A transparent explainability engine (**Grad-CAM**) that computes gradient-weighted activation heatmaps highlighting the exact spatial regions influencing the model's verdict.
3. An interactive cybersecurity web dashboard allowing users to upload media, review confidence probabilities, adjust attribution overlay opacity in real time, and examine side-by-side forensic breakdowns.

## TECHNOLOGY
* **Deep Learning Framework**: PyTorch 2.14, Torchvision
* **Model Architecture**: EfficientNet-B0 with custom Dropout/Linear classifier head
* **Explainability Engine**: Custom Grad-CAM with vectorized NumPy & Pillow JET/Turbo colormapping
* **Backend API**: FastAPI, Uvicorn, Pydantic, Python-Multipart
* **Frontend UI**: Semantic HTML5, Modern CSS3 Glassmorphism, Vanilla JavaScript (ES6)
* **Data Processing**: Pillow, NumPy, Pandas

## MODEL
* **Backbone**: EfficientNet-B0 (ImageNet-1K pretrained)
* **Classifier Head**: Dropout (p=0.3) $\to$ Linear (1280 in_features $\to$ 2 out_classes: 0=Real, 1=Synthetic)
* **Input Resolution**: $224 \times 224 \times 3$ (ImageNet normalization)
* **Target Explanation Layer**: `backbone.features[8]` (`Conv2dNormActivation`, 1,280 channels)

## DATASET
* **Dataset Benchmark**: GenImage Development Subset (Hugging Face: `jhutter2/281_Genimage`)
* **Total Usable Images**: **5,764** validated RGB images (49.3% Real vs. 50.7% Synthetic)
* **Real Source**: ImageNet-1K natural camera captures (2,842 images)
* **Synthetic Generators**:
  - Stable Diffusion v1.5: 996 images
  - ADM (Ablated Diffusion Model): 968 images
  - Wukong Diffusion: 958 images
* **Split Stratification**: Train (70% - 4,035 images), Validation (15% - 864 images), Test (15% - 865 images)

## RESULTS (Development Test Set: 865 Images)
* **Overall Accuracy**: **72.95%** (631 / 865 correct)
* **ROC-AUC**: **0.7996**
* **Macro-F1**: **0.7287**
* **Macro-Precision**: **0.7308** (Real: 74.55%, Synthetic: 71.61%)
* **Macro-Recall**: **0.7289** (Real: 68.62%, Synthetic: 77.17%)
* **Synthetic F1**: **0.7429**
* **Real F1**: **0.7146**
* **Per-Generator Accuracy**:
  - Stable Diffusion v1.5: **86.67%**
  - Wukong Diffusion: **83.22%**
  - Real ImageNet: **68.62%**
  - ADM Diffusion: **61.38%**

## ROBUSTNESS
Evaluated across 7 controlled real-world degradation conditions:
1. **Clean Baseline**: 72.95% Accuracy, 0.7996 ROC-AUC
2. **Mild JPEG (Q=75)**: 73.06% Accuracy, 0.8007 ROC-AUC ($\Delta\text{AUC} = +0.0011$)
3. **Strong JPEG (Q=50)**: 69.83% Accuracy, 0.7772 ROC-AUC ($\Delta\text{AUC} = -0.0224$)
4. **Resized (50% Down/Up)**: 70.40% Accuracy, 0.8076 ROC-AUC ($\Delta\text{AUC} = +0.0080$)
5. **Screenshot-like (Frame + Q=65)**: 68.55% Accuracy, 0.7569 ROC-AUC ($\Delta\text{AUC} = -0.0427$)
6. **Brightness (+10%)**: 72.14% Accuracy, 0.7995 ROC-AUC ($\Delta\text{AUC} = -0.0001$)
7. **Contrast (+15%)**: 72.25% Accuracy, 0.7997 ROC-AUC ($\Delta\text{AUC} = +0.0001$)

## EXPLAINABILITY
* Implemented Gradient-weighted Class Activation Mapping (Grad-CAM) hooked directly into `backbone.features[8]`.
* Generates 2D heatmaps indicating which spatial receptive fields positively supported the classifier's verdict.
* Provides live transparency sliding (0% to 100%), pure attribution heatmaps, and side-by-side comparative views.
* Accompanied by responsible AI documentation noting that heatmaps are decision-saliency interpretability aids, not ground-truth pixel forgery masks.

## APPLICATION
* **Backend API**: FastAPI server exposing `/predict`, `/api/examples`, `/api/health` with automated MIME verification and 15 MB file size constraints.
* **Frontend Web App**: Dark cybersecurity interface featuring drag-and-drop ingestion, 1-click forensic sample gallery, animated radar scanner, and responsive layout across desktop, tablet, and mobile.

## LIMITATIONS
1. **Unseen Generator Gap**: Evaluated exclusively on diffusion families present during development; zero-shot accuracy on completely unseen architectures (e.g. Midjourney v6, Flux, DALL-E 3) requires ongoing evaluation.
2. **Compression Vulnerability**: Heavy lossy compression (JPEG Q<50) discards subtle high-frequency spatial cues, slightly degrading sensitivity.
3. **Coarse Saliency**: Deep convolutional feature maps operate at $7 \times 7$ grid resolution; upsampled heatmaps illustrate general regions rather than pixel-sharp boundaries.
4. **Probabilistic Nature**: Model outputs are probabilistic estimates and should be corroborated by metadata and frequency analysis.

## DEMO (Local Run Instructions)
Start the complete application with a single command:
```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
Open in browser: `http://127.0.0.1:8000/`

## REPOSITORY
* Local Workspace: `c:\SignalScope`
* Remote URL: [TODO: To be set upon official hackathon remote push]
