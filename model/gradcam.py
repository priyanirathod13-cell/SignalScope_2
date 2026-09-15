from pathlib import Path

import torch
import torch.nn as nn

from torchvision import transforms, models
from torchvision.models import ResNet18_Weights

from PIL import Image
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================================================
# SignalScope - Grad-CAM Explanation
# ============================================================

PROJECT_DIR = Path(r"D:\SignalScope")

MODEL_PATH = (
    PROJECT_DIR
    / "model"
    / "signalscope_resnet18_v2.pth"
)

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Select an image
# ============================================================

TEST_DIR = (
    PROJECT_DIR
    / "data"
    / "dataset"
    / "test"
)

# Change this to any test image you want to inspect.
IMAGE_PATH = next(
    (TEST_DIR / "ai").glob("*.jpg")
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
# Load model
# ============================================================

print("=" * 60)
print("SignalScope - Grad-CAM")
print("=" * 60)

print(f"\nDevice: {DEVICE}")
print(f"Image: {IMAGE_PATH.name}")

model = models.resnet18(
    weights=ResNet18_Weights.DEFAULT
)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()


# ============================================================
# Load image
# ============================================================

original_image = Image.open(
    IMAGE_PATH
).convert("RGB")

original_image = original_image.resize(
    (IMAGE_SIZE, IMAGE_SIZE)
)

rgb_image = (
    torch.tensor(
        list(original_image.getdata()),
        dtype=torch.float32
    )
    .reshape(
        IMAGE_SIZE,
        IMAGE_SIZE,
        3
    )
    / 255.0
).numpy()


input_tensor = transform(
    original_image
).unsqueeze(0).to(DEVICE)


# ============================================================
# Prediction
# ============================================================

with torch.no_grad():

    output = model(input_tensor)

    probabilities = torch.softmax(
        output,
        dim=1
    )[0]

    prediction = torch.argmax(
        probabilities
    ).item()


class_names = {
    0: "AI",
    1: "Real"
}

predicted_class = class_names[prediction]

confidence = (
    probabilities[prediction].item()
    * 100
)


print(
    f"\nPrediction: {predicted_class}"
)

print(
    f"Confidence: {confidence:.2f}%"
)


# ============================================================
# Grad-CAM
# ============================================================

# Last convolutional layer of ResNet-18
target_layers = [
    model.layer4[-1]
]

cam = GradCAM(
    model=model,
    target_layers=target_layers
)

targets = [
    ClassifierOutputTarget(prediction)
]

grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)[0]


# ============================================================
# Create heatmap overlay
# ============================================================

visualization = show_cam_on_image(
    rgb_image,
    grayscale_cam,
    use_rgb=True
)


# ============================================================
# Save result
# ============================================================

output_path = (
    PROJECT_DIR
    / "evaluation"
    / "gradcam_result.jpg"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

Image.fromarray(
    visualization
).save(output_path)


print(
    f"\nGrad-CAM saved to:"
    f"\n{output_path}"
)


# ============================================================
# Display
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.subplot(1, 2, 1)

plt.imshow(
    original_image
)

plt.title(
    f"Original\n"
    f"{predicted_class} "
    f"({confidence:.1f}%)"
)

plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    visualization
)

plt.title(
    "Grad-CAM\n"
    "Influential regions"
)

plt.axis("off")


plt.tight_layout()

plt.show()