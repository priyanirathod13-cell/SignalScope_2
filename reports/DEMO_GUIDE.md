# SignalScope: Official 3-Minute Hackathon Demo Guide

This guide outlines a seamless, step-by-step 3-minute presentation flow for hackathon judges and evaluators.

---

## 0. Pre-Demo Setup (30 Seconds Before)
Open terminal in `c:\SignalScope` and run:
```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
Open your web browser to:
```
http://127.0.0.1:8000/
```
Confirm the green status indicator at the top right displays: **"Model Online • EfficientNet-B0"**.

---

## Minute 1: Problem, Solution & Authentic Image Demonstration (0:00 - 1:00)

1. **State Problem & Mission**:
   > *"Generative AI models like Stable Diffusion and Midjourney produce photorealistic media that can deceive human visual inspection. SignalScope was developed for SIH 2026 to detect synthetic diffusion patterns in real time and explain exactly which visual cues drove the model's conclusion."*

2. **Test an Authentic Image**:
   * Click the **"Authentic Photo (Nature)"** card in the preloaded sample gallery (or drag-and-drop `data/examples/example_1_real_nature.jpg`).
   * Point out the live scanning sequence (`Image verification` $\to$ `Inference` $\to$ `Grad-CAM`).
   * **Show the Result**:
     - Large **`REAL PHOTOGRAPHY`** badge in emerald green.
     - Confidence score: **`88.8%`** (Inference time: **~200–300 ms**).
     - Point out the class probability distribution: Real 88.8% vs AI 11.2%.

---

## Minute 2: AI-Generated Detection & Grad-CAM Explainability (1:00 - 2:00)

1. **Click "Analyze Another Image"**:
   * Show that the application cleanly resets.

2. **Test a Synthetic AI Image**:
   * Click the **"Stable Diffusion v1.5"** sample card (or drag `data/examples/example_2_stable_diffusion.png`).
   * **Show the Result**:
     - Large **`AI-GENERATED`** badge in neon pink/amber.
     - Confidence score: **`82.1%`** (Synthetic 82.1% vs Real 17.9%).

3. **Demonstrate Interactive Explainability**:
   * Scroll down to the **Grad-CAM Visual Attribution** card.
   * **Adjust the Transparency Slider**: Slide from 0% (original photo) to 50% (overlay) to 100% (pure heatmap).
   * **Switch Tabs**:
     - Click **"Pure Heatmap"** to show high-energy activation regions (vivid red/yellow).
     - Click **"Side-by-Side"** to show Original, Pure Heatmap, and Overlay simultaneously.
   * **Highlight Responsible AI**:
     - Read the interpretability note: *"The heatmap highlights image regions that contributed strongly to the model's prediction. It is an interpretability aid, not proof that those exact pixels are synthetic."*

---

## Minute 3: Robustness, Rigorous Evaluation & Integrity (2:00 - 3:00)

1. **Robustness Testing**:
   > *"In real-world deployment, images are re-saved and compressed on social media. We subjected our model to 7 controlled real-world degradations. While strong JPEG compression (Q=50) drops accuracy by 3.12%, mild compression (Q=75) and photometric adjustments retain over 80% ROC-AUC ranking discrimination."*

2. **Verified Benchmark Numbers**:
   * Development Test Set (865 images, zero data leakage):
     - **Accuracy: 72.95%**
     - **ROC-AUC: 0.7996**
     - **Macro-F1: 0.7287**
     - Stable Diffusion detection rate: **86.67%**
     - Wukong detection rate: **83.22%**

3. **Responsible AI & SIH Guardrails**:
   > *"SignalScope provides a probabilistic estimate, not definitive legal proof. Furthermore, our development strictly adhered to SIH guidelines: the official held-out test set remained completely untouched throughout training and evaluation."*

4. **Conclude & Take Questions**:
   > *"SignalScope is open-source, fully reproducible, and ready for immediate deployment."*
