"""
SignalScope Explainability Module
Provides Grad-CAM visual interpretability, colormap rendering, and overlay generation.
"""

from src.explainability.colormap import (
    apply_colormap,
    overlay_heatmap,
    build_jet_lut,
    build_turbo_lut
)
from src.explainability.gradcam import (
    GradCAM,
    explain_image,
    load_detector_model,
    save_explanation_artifacts,
    CLASS_LABELS
)

__all__ = [
    "GradCAM",
    "explain_image",
    "load_detector_model",
    "save_explanation_artifacts",
    "apply_colormap",
    "overlay_heatmap",
    "build_jet_lut",
    "build_turbo_lut",
    "CLASS_LABELS"
]
