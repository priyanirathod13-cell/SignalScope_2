"""
SignalScope Standalone Inference Module
Provides fast forward inference without explainability overhead.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Union, Optional

from PIL import Image
import numpy as np
import scipy.ndimage
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model
from src.training.dataset import get_transforms


def load_model(
    model_path: Union[str, Path] = "models/baseline/best_model.pt",
    config_path: Union[str, Path] = "config/train_config.json",
    device: Optional[torch.device] = None
):
    """Loads model and weights into eval mode."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = build_model(cfg).to(device)
    ckpt = torch.load(model_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model, cfg, device


def predict_image(
    image_path: Union[str, Path],
    model: Optional[torch.nn.Module] = None,
    model_path: Union[str, Path] = "models/baseline/best_model.pt",
    config_path: Union[str, Path] = "config/train_config.json",
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Runs binary classification on a single image file."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if model is None:
        model, cfg, device = load_model(model_path, config_path, device)
        image_size = cfg.get("data", {}).get("image_size", 224)
    else:
        if device is None:
            device = next(model.parameters()).device
        image_size = 224

    _, val_transform = get_transforms(image_size)

    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        w, h = img_rgb.size

    t0 = time.perf_counter()
    img_proc = img_rgb

    tensor = val_transform(img_proc).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits / 1.0195, dim=1)
        prob_synth = float(probs[0, 1].item())
        prob_real = float(probs[0, 0].item())
        pred_class = 1 if prob_synth > 0.50 else 0
        conf = float(probs[0, pred_class].item())

    dt = time.perf_counter() - t0

    label = "REAL" if pred_class == 0 else "AI-GENERATED"
    return {
        "file": image_path.name,
        "path": str(image_path.resolve()),
        "image_width": w,
        "image_height": h,
        "label": label,
        "class_id": pred_class,
        "confidence": round(conf, 4),
        "confidence_percent": f"{conf * 100.0:.2f}%",
        "probabilities": {
            "real": round(float(probs[0, 0].item()), 4),
            "synthetic": round(float(probs[0, 1].item()), 4)
        },
        "inference_time_sec": round(dt, 4),
        "model": "EfficientNet-B0 (SignalScopeClassifier)"
    }


def main():
    parser = argparse.ArgumentParser(description="SignalScope Fast Image Authenticity Predictor")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--model", type=str, default="models/baseline/best_model.pt")
    parser.add_argument("--config", type=str, default="config/train_config.json")
    args = parser.parse_args()

    res = predict_image(args.image, model_path=args.model, config_path=args.config)
    print("=" * 60)
    print("             SIGNALSCOPE INFERENCE RESULT")
    print("=" * 60)
    print(f"File:           {res['file']} ({res['image_width']}x{res['image_height']})")
    print(f"Prediction:     [{res['class_id']}] {res['label']}")
    print(f"Confidence:     {res['confidence_percent']}")
    print(f"Probabilities:  Real: {res['probabilities']['real']*100:.2f}% | AI: {res['probabilities']['synthetic']*100:.2f}%")
    print(f"Latency:        {res['inference_time_sec']*1000:.1f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
