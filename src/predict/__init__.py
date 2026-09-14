"""
SignalScope Prediction Package
Programmatic and CLI inference interface for image authenticity detection.
"""

from src.predict.inference import predict_image, load_model

__all__ = ["predict_image", "load_model"]
