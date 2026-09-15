from pathlib import Path
import sys

# Add project root to Python path
PROJECT_DIR = Path(r"D:\SignalScope")
sys.path.insert(0, str(PROJECT_DIR))

import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from model.model import load_model


# ============================================================
# Configuration
# ============================================================

AI_FOLDER = PROJECT_DIR / "data" / "external_test" / "ai"
REAL_FOLDER = PROJECT_DIR / "data" / "external_test" / "real"

OUTPUT_PATH = (
    PROJECT_DIR
    / "evaluation"
    / "gradcam_external_errors_v3.png"
)


# The four most important external mistakes
IMAGE_LIST = [
    (AI_FOLDER / "image_10.png", "AI → Real"),
    (AI_FOLDER / "image_5.png", "AI → Real"),
    (REAL_FOLDER / "real_26.jpg", "Real → AI"),
    (REAL_FOLDER / "real_20.jpg", "Real → AI"),
]


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# Load model
# ============================================================

print("\nLoading SignalScope V3 model...")

model = load_model(
    PROJECT_DIR / "model" / "signalscope_resnet18_v3.pth"
)

model = model.to(device)

model.eval()

print("Model loaded successfully.")


# ============================================================
# Image preprocessing
# ============================================================

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Grad-CAM target layer
# ============================================================

target_layer = model.layer4[-1]

cam = GradCAM(
    model=model,
    target_layers=[target_layer]
)


# ============================================================
# Process images
# ============================================================

results = []

for image_path, error_type in IMAGE_LIST:

    print(
        f"\nProcessing: {image_path.name}"
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    # Original image for display
    display_image = image.copy()

    # Prepare model input
    input_tensor = transform(
        image
    ).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():

        output = model(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = (
            torch.argmax(
                probabilities,
                dim=1
            )
            .item()
        )

    # Generate CAM for predicted class
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[
            ClassifierOutputTarget(
                predicted_class
            )
        ]
    )[0]

    # Resize original image
    rgb_image = np.array(
        display_image.resize(
            (224, 224)
        )
    ).astype(
        np.float32
    ) / 255.0

    # Overlay heatmap
    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    confidence = (
        probabilities[0, predicted_class]
        .item()
        * 100
    )

    predicted_label = (
        "AI"
        if predicted_class == 0
        else "Real"
    )

    results.append({
        "name": image_path.name,
        "error_type": error_type,
        "predicted": predicted_label,
        "confidence": confidence,
        "original": np.array(
            display_image.resize(
                (224, 224)
            )
        ),
        "cam": visualization
    })

    print(
        f"Predicted: {predicted_label}"
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )


# ============================================================
# Create figure
# ============================================================

fig, axes = plt.subplots(
    len(results),
    2,
    figsize=(10, 5 * len(results))
)


for row, result in enumerate(results):

    # Original image
    axes[row, 0].imshow(
        result["original"]
    )

    axes[row, 0].axis("off")

    axes[row, 0].set_title(
        f"{result['name']}\n"
        f"{result['error_type']}\n"
        f"Predicted: {result['predicted']} "
        f"({result['confidence']:.1f}%)"
    )


    # Grad-CAM
    axes[row, 1].imshow(
        result["cam"]
    )

    axes[row, 1].axis("off")

    axes[row, 1].set_title(
        "Grad-CAM: Where the model looked"
    )


plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print("\n" + "=" * 70)

print(
    "Grad-CAM analysis saved to:"
)

print(
    OUTPUT_PATH
)

print("=" * 70)