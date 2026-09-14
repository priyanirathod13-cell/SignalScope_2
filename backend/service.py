"""
SignalScope V3 Backend Inference & Forensic Explainability Service
Safely loads the trained SignalScope V3 EfficientNet-B0 model and provides real-time:
  - Letterboxed aspect-ratio preserving inference
  - Temperature-calibrated confidence intervals
  - Grad-CAM convolutional spatial attribution
  - Controlled peak-evidence occlusion test (faithfulness verification)
  - Plain-English forensic rationale ("Why SignalScope Thinks This")
  - Base64 visualization artifact encoding
"""

import base64
import io
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from src.models import build_model
from src.explainability.evidence_test import run_evidence_test
from src.explainability.colormap import apply_colormap, overlay_heatmap
from src.training.dataset import LetterboxTransform

MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

class DetectionService:
    """
    Singleton service holding the preloaded SignalScope V3 classifier
    and performing validated forward inference and explainability.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        # Default priority: V3 -> V2 -> V1
        if model_path is None:
            v3_model = PROJECT_ROOT / "models" / "v3_final_candidate" / "best_model.pt"
            v2_model = PROJECT_ROOT / "models" / "v2_expanded_data" / "best_model.pt"
            v1_model = PROJECT_ROOT / "models" / "v1_baseline" / "best_model.pt"
            if v3_model.exists():
                self.model_path = v3_model
                self.config_path = PROJECT_ROOT / "config" / "v3_train_config.json"
                self.version = "V3"
            elif v2_model.exists():
                self.model_path = v2_model
                self.config_path = PROJECT_ROOT / "config" / "v2_train_config.json"
                self.version = "V2"
            else:
                self.model_path = v1_model
                self.config_path = PROJECT_ROOT / "config" / "train_config.json"
                self.version = "V1"
        else:
            self.model_path = Path(model_path)
            self.config_path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "v3_train_config.json"
            self.version = "V3"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Service] Initializing SignalScope {self.version} detector on {self.device} from {self.model_path}...")

        # Load config
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        # Build & load model
        self.model = build_model(self.cfg).to(self.device)
        if self.model_path.exists():
            ckpt = torch.load(self.model_path, map_location=self.device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(state_dict)
            print(f"[Service] Successfully loaded weights from {self.model_path}")
        else:
            print(f"[Service] Warning: Checkpoint {self.model_path} not found; using initialized model.")
        self.model.eval()

        self.image_size = self.cfg.get("data", {}).get("image_size", 224)

        # Load temperature calibration if available
        self.temperature = 1.0
        calib_path = self.model_path.parent / "temperature_calibration.json"
        if calib_path.exists():
            try:
                with open(calib_path, "r", encoding="utf-8") as cf:
                    cal_data = json.load(cf)
                    self.temperature = cal_data.get("optimal_temperature", 1.0)
                print(f"[Service] Calibrated Temperature applied: T={self.temperature:.4f}")
            except Exception as e:
                print(f"[Service] Could not read temperature calibration: {e}")

    @staticmethod
    def _pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
        """Encodes PIL image as base64 data URI."""
        buf = io.BytesIO()
        img.save(buf, format=format)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{encoded}"

    def validate_image_bytes(self, image_bytes: bytes, filename: str) -> Image.Image:
        if not image_bytes:
            raise ValueError("Uploaded file is empty.")

        if len(image_bytes) > MAX_FILE_SIZE_BYTES:
            size_mb = len(image_bytes) / (1024 * 1024)
            raise ValueError(f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit of 15 MB.")

        ext = Path(filename).suffix.lower()
        if ext and ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file format '{ext}'. Allowed formats: JPG, JPEG, PNG, WEBP.")

        try:
            stream = io.BytesIO(image_bytes)
            test_img = Image.open(stream)
            test_img.verify()
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
        Performs end-to-end V3 classification, evidence testing, and visualization rendering.
        """
        t_start = time.perf_counter()

        img = self.validate_image_bytes(image_bytes, filename)
        orig_w, orig_h = img.size

        # Run explainability & causal occlusion evidence test
        evidence_result = run_evidence_test(
            model=self.model,
            image_input=img,
            device=self.device,
            temperature=self.temperature
        )

        pred_class = evidence_result["predicted_class"]
        label = "REAL" if pred_class == 0 else "AI-GENERATED"
        confidence = evidence_result["confidence"]
        prob_real = evidence_result["calibrated_probabilities"]["real"]
        prob_synth = evidence_result["calibrated_probabilities"]["synthetic"]

        # Confidence band
        if prob_synth >= 0.90 or prob_real >= 0.90:
            confidence_band = "HIGH CONFIDENCE"
            confidence_tier_desc = "Strong statistical evidence from multi-scale feature representations."
        elif prob_synth >= 0.70 or prob_real >= 0.70:
            confidence_band = "MODERATE CONFIDENCE"
            confidence_tier_desc = "Noticeable forensic indicators present; inspection of evidence recommended."
        else:
            confidence_band = "BORDERLINE / INCONCLUSIVE"
            confidence_tier_desc = "Ambiguous visual features near the decision boundary; manual forensic verification required."

        # Convert images to base64
        orig_b64 = self._pil_to_base64(img)
        over_b64 = self._pil_to_base64(evidence_result["overlay_image"])
        heat_b64 = self._pil_to_base64(evidence_result["heatmap_image"])

        inference_time = round(time.perf_counter() - t_start, 3)

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "confidence_percent": f"{confidence * 100.0:.1f}%",
            "confidence_band": confidence_band,
            "confidence_tier_description": confidence_tier_desc,
            "model": f"SignalScope {self.version} EfficientNet-B0",
            "inference_time": inference_time,
            "image_width": orig_w,
            "image_height": orig_h,
            "explanation_available": True,
            "probabilities": {
                "real": round(prob_real, 4),
                "synthetic": round(prob_synth, 4)
            },
            "evidence": evidence_result["evidence"],
            "forensic_rationale": evidence_result["forensic_rationale"],
            "original_image": orig_b64,
            "heatmap_image": heat_b64,
            "overlay_image": over_b64,
            "target_layer": evidence_result["evidence"]["target_layer"],
            "responsible_explanation": (
                "The heatmap highlights image regions that contributed strongly to the model's prediction. "
                "Causal occlusion verification tests whether removing these features alters the model's confidence."
            ),
            "disclaimer": "SignalScope provides a model-based estimate, not definitive proof of image origin."
        }

_detection_service: Optional[DetectionService] = None

def get_detection_service() -> DetectionService:
    global _detection_service
    if _detection_service is None:
        _detection_service = DetectionService()
    return _detection_service
