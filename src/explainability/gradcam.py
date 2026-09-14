"""
SignalScope Explainability: Grad-CAM Core Engine
Implements Gradient-weighted Class Activation Mapping for CNN classifiers.
Computes activation heatmaps, overlays, and prediction metrics.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from src.explainability.colormap import apply_colormap, overlay_heatmap
from src.models import build_model
from src.training.dataset import get_transforms


CLASS_LABELS = {
    0: "REAL",
    1: "AI-GENERATED (SYNTHETIC)"
}


class GradCAM:
    """
    Grad-CAM explanation generator for convolutional neural network classifiers.
    Hooks into the final convolutional feature layer to compute gradients of the
    target class score with respect to feature activation maps.
    """
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self.model.eval()

        # Identify target layer if not explicitly specified
        if target_layer is None:
            if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "features"):
                # EfficientNet-B0 top feature block (Conv2dNormActivation: Conv2d 320->1280, BN, SiLU)
                self.target_layer = self.model.backbone.features[8]
                self.target_layer_name = "backbone.features[8] (Conv2dNormActivation)"
            else:
                # Fallback search for last Conv2d or container
                conv_modules = [m for m in self.model.modules() if isinstance(m, nn.Conv2d)]
                if conv_modules:
                    self.target_layer = conv_modules[-1]
                    self.target_layer_name = f"last Conv2d: {self.target_layer}"
                else:
                    raise ValueError("Could not automatically locate a convolutional layer for Grad-CAM.")
        else:
            self.target_layer = target_layer
            self.target_layer_name = str(target_layer.__class__.__name__)

        self.activations = []
        self.gradients = []
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        self._remove_hooks()

        def forward_hook(module, inp, out):
            self.activations.append(out)
            # Register backward hook on output tensor to capture gradient w.r.t activation
            def backward_hook(grad):
                self.gradients.append(grad)
            out.register_hook(backward_hook)

        h = self.target_layer.register_forward_hook(forward_hook)
        self.handles.append(h)

    def _remove_hooks(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()
        self.activations.clear()
        self.gradients.clear()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generates Grad-CAM activation map for a batch of 1 image.

        Args:
            input_tensor: Shape (1, 3, H, W) normalized image tensor.
            target_class: Class index to explain (0 or 1). If None, uses model's predicted class.

        Returns:
            Dict containing:
              - cam: 2D numpy array of shape (H_feat, W_feat) in [0, 1]
              - cam_tensor: 2D torch.Tensor of shape (H_feat, W_feat)
              - pred_class: Predicted class integer
              - target_class: Target class integer explained
              - confidence: Confidence probability for predicted class
              - probabilities: Dict mapping class names to probabilities
              - logits: Raw logit outputs
        """
        input_tensor = input_tensor.to(self.device)

        # Ensure parameters allow gradient propagation
        for p in self.model.parameters():
            p.requires_grad = True

        self.activations.clear()
        self.gradients.clear()

        # Forward pass
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred_class].item())

        if target_class is None:
            target_class = pred_class
        else:
            target_class = int(target_class)

        # Backward pass on target class logit
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=False)

        if not self.activations or not self.gradients:
            raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")

        # Extract features and gradients: shape (C, H_feat, W_feat)
        act = self.activations[-1][0].detach()
        grad = self.gradients[-1][0].detach()

        # Global average pooling on gradients -> channel weights alpha
        # Shape: (C, 1, 1)
        weights = torch.mean(grad, dim=(1, 2), keepdim=True)

        # Weighted combination of activation maps
        # Shape: (H_feat, W_feat)
        cam = torch.sum(weights * act, dim=0)

        # ReLU: keep only features with positive influence on target class
        cam = torch.relu(cam)

        # Normalize CAM to [0, 1]
        c_min = float(cam.min())
        c_max = float(cam.max())
        if c_max > c_min:
            cam_norm = (cam - c_min) / (c_max - c_min)
        else:
            cam_norm = torch.zeros_like(cam)

        cam_np = cam_norm.cpu().numpy()

        prob_real = float(probs[0, 0].item())
        prob_synthetic = float(probs[0, 1].item())

        return {
            "cam": cam_np,
            "cam_tensor": cam_norm,
            "pred_class": pred_class,
            "pred_label": CLASS_LABELS.get(pred_class, "UNKNOWN"),
            "target_class": target_class,
            "target_label": CLASS_LABELS.get(target_class, "UNKNOWN"),
            "confidence": confidence,
            "probabilities": {
                "real": prob_real,
                "synthetic": prob_synthetic
            },
            "logits": [float(l.item()) for l in logits[0]],
            "target_layer_name": self.target_layer_name
        }

    def close(self):
        """Detaches hooks cleanly."""
        self._remove_hooks()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def load_detector_model(
    model_path: Union[str, Path] = "models/baseline/best_model.pt",
    config_path: Union[str, Path] = "config/train_config.json",
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, dict, torch.device]:
    """Loads the SignalScope classifier model and checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = build_model(cfg).to(device)

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, cfg, device


def explain_image(
    image_path: Union[str, Path],
    model: Optional[nn.Module] = None,
    model_path: Union[str, Path] = "models/baseline/best_model.pt",
    config_path: Union[str, Path] = "config/train_config.json",
    target_class: Optional[Union[int, str]] = None,
    colormap: str = "jet",
    alpha: float = 0.5,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Performs complete end-to-end inference and Grad-CAM explanation for an image file.

    Args:
        image_path: Path to image file.
        model: Optional pre-loaded PyTorch model instance.
        model_path: Checkpoint path if model is None.
        config_path: Config path if model is None.
        target_class: Specific class to explain (0/1 or "real"/"synthetic"). If None, uses predicted class.
        colormap: "jet" or "turbo" colormap palette.
        alpha: Overlay blend factor (0.0=original only, 1.0=heatmap only).
        device: torch.device.

    Returns:
        Dict containing predictions, PIL heatmap, PIL overlay, metrics, and metadata.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load model if not provided
    if model is None:
        model, cfg, device = load_detector_model(model_path, config_path, device)
        image_size = cfg.get("data", {}).get("image_size", 224)
    else:
        if device is None:
            device = next(model.parameters()).device
        image_size = 224

    # Resolve target class string alias if needed
    if isinstance(target_class, str):
        target_lower = target_class.strip().lower()
        if target_lower in ["0", "real"]:
            target_class = 0
        elif target_lower in ["1", "ai", "synthetic", "fake"]:
            target_class = 1
        elif target_lower == "auto" or target_lower == "none":
            target_class = None
        else:
            raise ValueError(f"Unrecognized target_class: '{target_class}'. Use 0 ('real'), 1 ('synthetic'), or 'auto'.")

    # Load original image
    with Image.open(image_path) as img:
        original_img = img.convert("RGB")
    orig_w, orig_h = original_img.size

    # Prepare input tensor using standard validation transforms
    _, val_transform = get_transforms(image_size)
    input_tensor = val_transform(original_img).unsqueeze(0).to(device)

    # Compute Grad-CAM
    with GradCAM(model, device=device) as gcam:
        result = gcam.generate(input_tensor, target_class=target_class)

    cam_tensor = result["cam_tensor"]  # Shape: (H_feat, W_feat)

    # Upsample activation heatmap to match original image dimensions using bilinear interpolation
    cam_4d = cam_tensor.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, H_feat, W_feat)
    cam_upsampled = F.interpolate(
        cam_4d,
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False
    ).squeeze().cpu().numpy()

    # Generate pure heatmap image
    heatmap_img = apply_colormap(cam_upsampled, colormap=colormap)

    # Generate overlay image
    overlay_img = overlay_heatmap(original_img, heatmap_img, alpha=alpha)

    return {
        "image_path": str(image_path),
        "image_name": image_path.name,
        "original_size": (orig_w, orig_h),
        "target_layer_name": result["target_layer_name"],
        "pred_class": result["pred_class"],
        "pred_label": result["pred_label"],
        "target_class": result["target_class"],
        "target_label": result["target_label"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "logits": result["logits"],
        "heatmap_img": heatmap_img,
        "overlay_img": overlay_img,
        "cam_array": cam_upsampled,
        "colormap": colormap,
        "alpha": alpha
    }


def save_explanation_artifacts(
    explanation_result: Dict[str, Any],
    output_dir: Union[str, Path] = "reports/explanations",
    prefix: str = ""
) -> Dict[str, str]:
    """
    Saves the pure heatmap, overlay, and metadata JSON to disk.

    Args:
        explanation_result: Output dict from explain_image().
        output_dir: Destination folder.
        prefix: Optional filename prefix.

    Returns:
        Dict mapping artifact types to saved absolute paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_name = explanation_result["image_name"]
    stem = Path(img_name).stem
    if prefix:
        stem = f"{prefix}_{stem}"

    heatmap_path = out_dir / f"{stem}_heatmap.png"
    overlay_path = out_dir / f"{stem}_overlay.png"
    meta_path = out_dir / f"{stem}_explanation.json"

    # Save images
    explanation_result["heatmap_img"].save(heatmap_path, "PNG")
    explanation_result["overlay_img"].save(overlay_path, "PNG")

    # Save JSON metadata (excluding non-serializable objects)
    meta = {
        "image_name": img_name,
        "image_path": explanation_result["image_path"],
        "original_dimensions": {
            "width": explanation_result["original_size"][0],
            "height": explanation_result["original_size"][1]
        },
        "target_layer": explanation_result["target_layer_name"],
        "predicted_class_id": explanation_result["pred_class"],
        "predicted_label": explanation_result["pred_label"],
        "target_class_id": explanation_result["target_class"],
        "target_label": explanation_result["target_label"],
        "confidence": round(explanation_result["confidence"], 4),
        "confidence_percentage": f"{explanation_result['confidence'] * 100.0:.2f}%",
        "probabilities": {
            "real": round(explanation_result["probabilities"]["real"], 4),
            "synthetic": round(explanation_result["probabilities"]["synthetic"], 4)
        },
        "logits": [round(x, 4) for x in explanation_result["logits"]],
        "visualization": {
            "colormap": explanation_result["colormap"],
            "alpha": explanation_result["alpha"],
            "heatmap_file": heatmap_path.name,
            "overlay_file": overlay_path.name
        },
        "interpretation_disclaimer": (
            "Grad-CAM indicates regions that positively influenced the neural network's decision. "
            "It is a feature attribution method, not a ground-truth pixel-level forgery mask."
        )
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "heatmap": str(heatmap_path.resolve()),
        "overlay": str(overlay_path.resolve()),
        "metadata": str(meta_path.resolve())
    }
