"""
SignalScope Explainability: Colormap and Overlay Utilities
Provides zero-external-dependency (pure NumPy + PIL) Jet and Turbo colormaps
and image alpha-blending for forensic attribution heatmaps.
"""

import numpy as np
from PIL import Image


def build_jet_lut() -> np.ndarray:
    """
    Constructs a 256-entry RGB lookup table for the standard JET colormap.
    Transitions:
      0.00 - 0.125: Dark Blue to Blue
      0.125 - 0.375: Blue to Cyan
      0.375 - 0.625: Cyan to Green to Yellow
      0.625 - 0.875: Yellow to Orange to Red
      0.875 - 1.00: Vivid Red
    Returns:
      uint8 array of shape (256, 3)
    """
    x = np.linspace(0.0, 1.0, 256)
    r = np.zeros(256, dtype=np.float32)
    g = np.zeros(256, dtype=np.float32)
    b = np.zeros(256, dtype=np.float32)

    for i, v in enumerate(x):
        if v < 0.125:
            r[i] = 0.0
            g[i] = 0.0
            b[i] = 0.5 + 4.0 * v
        elif v < 0.375:
            r[i] = 0.0
            g[i] = (v - 0.125) * 4.0
            b[i] = 1.0
        elif v < 0.625:
            r[i] = (v - 0.375) * 4.0
            g[i] = 1.0
            b[i] = 1.0 - (v - 0.375) * 4.0
        elif v < 0.875:
            r[i] = 1.0
            g[i] = 1.0 - (v - 0.625) * 4.0
            b[i] = 0.0
        else:
            r[i] = 1.0
            g[i] = 0.0
            b[i] = 0.0

    lut = np.clip(np.stack([r, g, b], axis=1) * 255.0, 0, 255).astype(np.uint8)
    return lut


def build_turbo_lut() -> np.ndarray:
    """
    Constructs a 256-entry RGB lookup table for the Google Turbo colormap.
    Turbo is a smooth, perceptually uniform alternative to Jet.
    Returns:
      uint8 array of shape (256, 3)
    """
    x = np.linspace(0.0, 1.0, 256)
    r = 0.1357 + x * (4.5974 - x * (42.3277 - x * (130.5887 - x * (150.5666 - x * 58.1375))))
    g = 0.0914 + x * (2.1856 + x * (4.8052 - x * (14.0195 - x * (4.2109 + x * 2.7747))))
    b = 0.1067 + x * (12.5925 - x * (60.1818 - x * (109.8185 - x * (88.5022 - x * 26.8183))))
    lut = np.clip(np.stack([r, g, b], axis=1) * 255.0, 0, 255).astype(np.uint8)
    return lut


_LUT_CACHE = {
    "jet": build_jet_lut(),
    "turbo": build_turbo_lut()
}


def apply_colormap(heatmap_2d: np.ndarray, colormap: str = "jet") -> Image.Image:
    """
    Converts a 2D float array in [0, 1] into an RGB PIL Image using a colormap.

    Args:
        heatmap_2d: 2D numpy array with values in [0.0, 1.0].
        colormap: "jet" or "turbo" (defaults to "jet").

    Returns:
        PIL.Image.Image in RGB format.
    """
    cmap_key = colormap.lower()
    if cmap_key not in _LUT_CACHE:
        cmap_key = "jet"

    lut = _LUT_CACHE[cmap_key]

    # Normalize if values exceed [0, 1]
    h_min = float(heatmap_2d.min())
    h_max = float(heatmap_2d.max())
    if h_max > h_min:
        norm = np.clip((heatmap_2d - h_min) / (h_max - h_min), 0.0, 1.0)
    else:
        norm = np.zeros_like(heatmap_2d, dtype=np.float32)

    indices = (norm * 255.0).astype(np.uint8)
    rgb_arr = lut[indices]  # Shape (H, W, 3)
    return Image.fromarray(rgb_arr, mode="RGB")


def overlay_heatmap(
    original_img: Image.Image,
    heatmap_img: Image.Image,
    alpha: float = 0.5
) -> Image.Image:
    """
    Blends the original RGB image with the heatmap RGB image.
    Formula: (1 - alpha) * original + alpha * heatmap

    Args:
        original_img: Original PIL Image.
        heatmap_img: Heatmap PIL Image (will be resized to match original if needed).
        alpha: Weight for the heatmap layer (0.0 = original only, 1.0 = heatmap only).

    Returns:
        Blended PIL Image in RGB mode.
    """
    orig_rgb = original_img.convert("RGB")
    heat_rgb = heatmap_img.convert("RGB")

    if orig_rgb.size != heat_rgb.size:
        heat_rgb = heat_rgb.resize(orig_rgb.size, Image.Resampling.BILINEAR)

    clamped_alpha = float(np.clip(alpha, 0.0, 1.0))
    return Image.blend(orig_rgb, heat_rgb, clamped_alpha)
