from pathlib import Path

import torch

from torchvision import transforms

from PIL import Image

from model.model import load_model


# ============================================================
# SignalScope Prediction
# ============================================================

PROJECT_DIR = PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "model"
    / "signalscope_resnet18_v3.pth"
)

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Image preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Classes
# ============================================================

CLASS_NAMES = {0: "Real", 1: "AI"}


# ============================================================
# Load model
# ============================================================

print("Loading SignalScope model...")

model = load_model(
    MODEL_PATH,
    DEVICE
)

print("Model loaded successfully.")


# ============================================================
# Prediction function
# ============================================================

def predict_image(image_path):
    """
    Predict whether an image is AI-generated or real.

    Returns:
        {
            "label": "AI" or "Real",
            "confidence": float,
            "ai_probability": float,
            "real_probability": float
        }
    """

    image = Image.open(
        image_path
    ).convert("RGB")

    input_tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)


    with torch.no_grad():

        output = model(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]


    prediction = torch.argmax(
        probabilities
    ).item()


    real_probability = probabilities[0].item()*100
    ai_probability = probabilities[1].item()*100
    

    confidence = (
        probabilities[prediction].item()
        * 100
    )


    return {
        "label": CLASS_NAMES[prediction],
        "confidence": confidence,
        "ai_probability": ai_probability,
        "real_probability": real_probability
    }


# ============================================================
# Test the prediction system
# ============================================================

if __name__ == "__main__":

    # Automatically select one test image
    test_images = list(
        (
            PROJECT_DIR
            / "data"
            / "dataset"
            / "test"
            / "ai"
        ).glob("*.jpg")
    )

    if not test_images:

        raise FileNotFoundError(
            "No test images found."
        )


    image_path = test_images[0]

    print()
    print("=" * 60)
    print("SignalScope Prediction Test")
    print("=" * 60)

    print(
        f"\nImage: {image_path.name}"
    )

    result = predict_image(
        image_path
    )

    print(
        f"\nPrediction: "
        f"{result['label']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence']:.2f}%"
    )

    print(
        f"AI probability: "
        f"{result['ai_probability']:.2f}%"
    )

    print(
        f"Real probability: "
        f"{result['real_probability']:.2f}%"
    )

    print("=" * 60)