"""
SignalScope Backend Inference & Explainability Service
Safely loads the trained EfficientNet-B0 model and provides real-time
classification, Grad-CAM heatmap generation, and base64 artifact encoding.
"""

import base64
import io
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.explainability.colormap import apply_colormap, overlay_heatmap
from src.explainability.gradcam import GradCAM, load_detector_model
from src.training.dataset import get_transforms


MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


class DetectionService:
    """
    Singleton service holding the preloaded SignalScope classifier
    and performing validated forward inference and Grad-CAM explainability.
    """
    def __init__(
        self,
        model_path: str = "models/baseline/best_model.pt",
        config_path: str = "config/train_config.json"
    ):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[Service] Initializing SignalScope detector on {self.device}...")
        self.model, self.cfg, self.device = load_detector_model(
            model_path=self.model_path,
            config_path=self.config_path,
            device=self.device
        )
        self.image_size = self.cfg.get("data", {}).get("image_size", 224)
        _, self.val_transform = get_transforms(self.image_size)
        print(f"[Service] Model loaded successfully! (Input resolution: {self.image_size}x{self.image_size})")

    @staticmethod
    def _pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
        """Encodes PIL image as base64 data URI."""
        buf = io.BytesIO()
        img.save(buf, format=format)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{encoded}"

    def validate_image_bytes(self, image_bytes: bytes, filename: str) -> Image.Image:
        """
        Validates uploaded image bytes against size limits and corrupt headers.
        Returns verified RGB PIL Image.
        """
        if not image_bytes:
            raise ValueError("Uploaded file is empty.")

        if len(image_bytes) > MAX_FILE_SIZE_BYTES:
            size_mb = len(image_bytes) / (1024 * 1024)
            raise ValueError(f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit of 15 MB.")

        ext = Path(filename).suffix.lower()
        if ext and ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file format '{ext}'. Allowed formats: JPG, JPEG, PNG, WEBP.")

        try:
            # First verify integrity without full decode
            stream = io.BytesIO(image_bytes)
            test_img = Image.open(stream)
            test_img.verify()

            # Re-open for actual processing
            stream.seek(0)
            img = Image.open(stream).convert("RGB")
        except Exception as e:
            raise ValueError(f"Corrupted or unreadable image file: {str(e)}")

        w, h = img.size
        if w < 16 or h < 16:
            raise ValueError(f"Image dimensions ({w}x{h}) are too small for forensic analysis (minimum 16x16 required).")

        return img

    def analyze_image(
        self,
        image_bytes: bytes,
        filename: str,
        colormap: str = "jet",
        alpha: float = 0.5
    ) -> Dict[str, Any]:
        """
        Performs end-to-end classification and Grad-CAM attribution.
        """
        t_start = time.perf_counter()

        # 1. Validate and decode
        img = self.validate_image_bytes(image_bytes, filename)
        orig_w, orig_h = img.size

        # 2. Preprocess
        input_tensor = self.val_transform(img).unsqueeze(0).to(self.device)

        # 3. Forward pass & Grad-CAM attribution
        with GradCAM(self.model, device=self.device) as gcam:
            cam_res = gcam.generate(input_tensor)

        pred_class = cam_res["pred_class"]
        confidence = float(cam_res["confidence"])
        label = "REAL" if pred_class == 0 else "AI-GENERATED"
        prob_real = float(cam_res["probabilities"]["real"])
        prob_synth = float(cam_res["probabilities"]["synthetic"])

        # 4. Upsample CAM to original image dimensions using bilinear interpolation
        cam_tensor = cam_res["cam_tensor"].unsqueeze(0).unsqueeze(0)
        cam_upsampled = F.interpolate(
            cam_tensor,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False
        ).squeeze().cpu().numpy()

        # 5. Render heatmaps
        heatmap_img = apply_colormap(cam_upsampled, colormap=colormap)
        overlay_img = overlay_heatmap(img, heatmap_img, alpha=alpha)

        # 6. Encode visualizations to base64 data URIs
        orig_b64 = self._pil_to_base64(img)
        heat_b64 = self._pil_to_base64(heatmap_img)
        over_b64 = self._pil_to_base64(overlay_img)

        inference_time = round(time.perf_counter() - t_start, 3)

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "confidence_percent": f"{confidence * 100.0:.1f}%",
            "model": "EfficientNet-B0 (SignalScopeClassifier)",
            "inference_time": inference_time,
            "image_width": orig_w,
            "image_height": orig_h,
            "explanation_available": True,
            "probabilities": {
                "real": round(prob_real, 4),
                "synthetic": round(prob_synth, 4)
            },
            "original_image": orig_b64,
            "heatmap_image": heat_b64,
            "overlay_image": over_b64,
            "target_layer": cam_res["target_layer_name"],
            "responsible_explanation": (
                "The heatmap highlights image regions that contributed strongly to the model's prediction. "
                "It is an interpretability aid, not proof that those exact pixels are synthetic."
            ),
            "disclaimer": "SignalScope provides a model-based estimate, not definitive proof of image origin."
        }


# Global lazy-loaded singleton instance
_detection_service: Optional[DetectionService] = None


def get_detection_service() -> DetectionService:
    global _detection_service
    if _detection_service is None:
        _detection_service = DetectionService()
    return _detection_service
