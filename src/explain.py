"""
SignalScope Explainability CLI: Generate Visual Attributions for Images
Usage:
  python -m src.explain --image path/to/image.jpg
  python -m src.explain --image path/to/image.png --output-dir reports/explanations --colormap jet --alpha 0.5
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability import (
    explain_image,
    save_explanation_artifacts,
    load_detector_model,
    CLASS_LABELS
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SignalScope: Visual Explainability (Grad-CAM) for Image Authenticity Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to target image file for inference and explanation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/baseline/best_model.pt",
        help="Path to trained model checkpoint (.pt)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_config.json",
        help="Path to model training configuration JSON"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/explanations",
        help="Directory where heatmap, overlay, and metadata will be saved"
    )
    parser.add_argument(
        "--target-class",
        type=str,
        default="auto",
        help="Class to explain: 'auto' (predicted class), 'real' / '0', or 'synthetic' / '1'"
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="jet",
        choices=["jet", "turbo"],
        help="Colormap palette for attribution heatmap"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Transparency blend factor for overlay (0.0 = original image only, 1.0 = heatmap only)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional prefix for saved artifact filenames"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)

    if not image_path.exists():
        print(f"ERROR: Image file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 68)
    print("        SIGNALSCOPE: VISUAL EXPLAINABILITY (Grad-CAM)")
    print("=" * 68)
    print(f"Target Image:        {image_path}")
    print(f"Model Checkpoint:    {args.model}")
    print(f"Output Directory:    {args.output_dir}")

    # Load model
    print("\nLoading forensic detection model...")
    model, cfg, device = load_detector_model(
        model_path=args.model,
        config_path=args.config
    )
    print(f"Device:              {device}")

    # Generate explanation
    print("Running forward inference and backward gradient attribution...")
    explanation = explain_image(
        image_path=image_path,
        model=model,
        target_class=args.target_class,
        colormap=args.colormap,
        alpha=args.alpha,
        device=device
    )

    # Save artifacts
    saved_paths = save_explanation_artifacts(
        explanation_result=explanation,
        output_dir=args.output_dir,
        prefix=args.prefix
    )

    orig_w, orig_h = explanation["original_size"]
    pred_label = explanation["pred_label"]
    pred_class = explanation["pred_class"]
    conf = explanation["confidence"]
    probs = explanation["probabilities"]
    target_label = explanation["target_label"]

    print("\n" + "-" * 68)
    print("                      DETECTION RESULTS")
    print("-" * 68)
    print(f"Original Resolution: {orig_w} x {orig_h}")
    print(f"Attribution Layer:   {explanation['target_layer_name']}")
    print(f"Predicted Class:     [{pred_class}] {pred_label}")
    print(f"Confidence Score:    {conf * 100.0:.2f}%")
    print(f"Class Probabilities: Real: {probs['real'] * 100.0:.2f}% | AI/Synthetic: {probs['synthetic'] * 100.0:.2f}%")
    print(f"Explained Class:     [{explanation['target_class']}] {target_label}")
    print("-" * 68)
    print("                      SAVED ARTIFACTS")
    print("-" * 68)
    print(f"Pure Heatmap:        {saved_paths['heatmap']}")
    print(f"Attribution Overlay: {saved_paths['overlay']}")
    print(f"Metadata JSON:       {saved_paths['metadata']}")
    print("=" * 68)
    print("NOTE: Grad-CAM highlights image regions that influenced the model's")
    print("prediction. It indicates salient features, not ground-truth forgery masks.\n")


if __name__ == "__main__":
    main()
