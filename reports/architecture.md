# SignalScope System Architecture & Technical Specifications

## 1. System Overview
SignalScope is an AI image authenticity detection and interpretability framework designed to classify images as **REAL** or **AI-GENERATED (SYNTHETIC)** while attributing specific spatial features that drove the decision using **Grad-CAM**.

---

## 2. End-to-End Architecture Diagram

```mermaid
flowchart TD
    subgraph Ingestion ["1. INGESTION LAYER"]
        A[Input Image File] --> B{Format & Security Validation}
        B -- "Valid: JPG, PNG, WEBP (<15MB)" --> C[RGB Image Decoding]
        B -- "Invalid / Corrupt / Oversized" --> Err[HTTP 400 Client Error]
    end

    subgraph Preprocessing ["2. PREPROCESSING & NORMALIZATION"]
        C --> D[Resize to 224x224]
        D --> E[ImageNet Normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]]
        E --> F[Tensor Batch: 1 x 3 x 224 x 224]
    end

    subgraph ModelInference ["3. CONVOLUTIONAL DETECTOR (EfficientNet-B0)"]
        F --> G["features[0..7]: Low & Mid-level Feature Extraction"]
        G --> H["features[8]: Conv2dNormActivation (320 -> 1280 channels, 7x7 spatial)"]
        H --> I[Adaptive Global Average Pooling]
        I --> J["Classifier Head: Dropout(0.3) -> Linear(1280, 2)"]
        J --> K[Raw Logits: Real vs Synthetic]
    end

    subgraph Explainability ["4. GRAD-CAM ATTRIBUTION ENGINE"]
        K --> L["Target Class Logit y^c"]
        L --> M["Backward Autograd: dy^c / dA^k"]
        H -. Forward Activations A^k .-> N["Feature Importance Weights: alpha_k = (1/Z) sum(grad)"]
        M -. Gradients .-> N
        N --> O["Rectified Linear Combination: ReLU(sum(alpha_k * A^k))"]
        O --> P["Bilinear Upsampling to Original Resolution (W_orig x H_orig)"]
        P --> Q["Zero-Dependency Colormap: Standard JET / Turbo"]
        Q --> R["Alpha Blending: (1-alpha)*Original + alpha*Heatmap"]
    end

    subgraph Serving ["5. FASTAPI BACKEND API"]
        K --> S[Softmax Probability & Confidence]
        S --> T[JSON Payload Assembly]
        R --> T
        T --> U["REST API Responses: /predict, /api/examples, /api/health"]
    end

    subgraph UserInterface ["6. SIGNALSCOPE FRONTEND DASHBOARD"]
        U --> V[Verdict Card: REAL or AI-GENERATED Badge]
        U --> W[Confidence Metric & Probability Bars]
        U --> X[Interactive Explainability Stage: Live Opacity Slider & Multi-Tab Viewer]
        U --> Y[Responsible AI Interpretation Notice]
    end

    subgraph OfflineValidation ["OFFLINE BENCHMARKING & RIGOR"]
        Z1[(5,764 GenImage Subset)] --> Z2[70/15/15 Stratified Split]
        Z2 --> Z3[Training Pipeline: AdamW + Cosine Annealing]
        Z2 --> Z4["Evaluation Suite (865 Test Images): Accuracy, ROC-AUC, F1"]
        Z2 --> Z5["Robustness Suite: 7 Realistic Corruptions (JPEG, Resize, Screenshot, etc.)"]
    end
```

---

## 3. Detailed Component Breakdown

### 3.1 Input Validation & Ingestion
* **MIME Verification**: Restricts inputs to `image/jpeg`, `image/png`, `image/webp`.
* **Integrity Auditing**: Verifies file headers before full decompression to block truncated or malformed streams.
* **Payload Boundary**: Rejects files larger than 15 MB to protect against denial-of-service memory exhaustion.

### 3.2 Feature Extraction Backbone
* **Architecture**: EfficientNet-B0 transfer learning backbone initialized with ImageNet-1K pretrained weights.
* **Fine-Tuning Strategy**: Lower convolutional stages are frozen to preserve general edge/texture descriptors, while the top convolutional projection block (`features[8]`) and dense classification head were fine-tuned on generative diffusion artifacts.
* **Target Layer for Explanations**: `model.backbone.features[8]` (`Conv2dNormActivation`), providing $7 \times 7$ spatial activation maps with 1,280 channels.

### 3.3 Explainability Implementation
* **Method**: Gradient-weighted Class Activation Mapping (Grad-CAM).
* **Mathematical Formula**:
  $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial y^c}{\partial A_{i,j}^k}$$
  $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_{k} \alpha_k^c A^k\right)$$
* **Interpolation**: Bilinear upsampling directly to native resolution $(W_{\text{orig}}, H_{\text{orig}})$.
* **Colormap**: Vectorized pure NumPy 256-level JET lookup table without third-party graphics dependencies.

### 3.4 API & Frontend Interface
* **Backend Framework**: FastAPI with Starlette runtime, CORS middleware, and multipart stream parsing.
* **Frontend Design**: Semantic HTML5, CSS3 Glassmorphism, and Vanilla JavaScript ES6 modules.
* **Zero-Build Deployment**: Hosted natively through FastAPI's static file handler, enabling single-command startup without requiring Node.js or npm.
