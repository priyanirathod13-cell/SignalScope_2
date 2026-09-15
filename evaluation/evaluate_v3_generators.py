import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from collections import defaultdict


# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

AI_TEST_DIR = (
    PROJECT_DIR
    / "data"
    / "dataset_v3"
    / "test"
    / "ai"
)

MODEL_PATH = (
    PROJECT_DIR
    / "model"
    / "signalscope_resnet18_v3.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# Image transform
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Load model
# ============================================================

model = models.resnet18(weights=None)

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

print("Model loaded successfully.")


# ============================================================
# Generator detection from filename
# ============================================================

def get_generator(filename):

    name = filename.lower()

    if name.startswith("sd21_"):
        return "SD2.1"

    if name.startswith("sdxl_"):
        return "SDXL"

    if name.startswith("sd3_"):
        return "SD3"

    if name.startswith("dalle3_"):
        return "DALL-E 3"

    if name.startswith("midjourney6_"):
        return "Midjourney 6"

    return "Unknown"


# ============================================================
# Evaluate
# ============================================================

results = defaultdict(
    lambda: {
        "total": 0,
        "correct": 0
    }
)


image_extensions = [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.webp"
]

images = []

for extension in image_extensions:
    images.extend(
        AI_TEST_DIR.glob(extension)
    )

images = sorted(images)

print()
print("=" * 70)
print("SignalScope V3 - Generator-wise Evaluation")
print("=" * 70)

print()
print("AI test images:", len(images))


# ============================================================
# Run predictions
# ============================================================

with torch.no_grad():

    for image_path in images:

        generator = get_generator(
            image_path.name
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image_tensor = transform(
            image
        ).unsqueeze(0)

        image_tensor = image_tensor.to(
            DEVICE
        )

        output = model(
            image_tensor
        )

        prediction = torch.argmax(
            output,
            dim=1
        ).item()

        # AI = 0
        # Real = 1

        is_correct = (
            prediction == 0
        )

        results[generator]["total"] += 1

        if is_correct:
            results[generator]["correct"] += 1


# ============================================================
# Print results
# ============================================================

print()
print("=" * 70)
print("GENERATOR-WISE RESULTS")
print("=" * 70)

total_correct = 0
total_images = 0

for generator in [
    "SD2.1",
    "SDXL",
    "SD3",
    "DALL-E 3",
    "Midjourney 6",
    "Unknown"
]:

    if generator not in results:
        continue

    total = results[generator]["total"]
    correct = results[generator]["correct"]

    accuracy = (
        correct / total * 100
        if total > 0
        else 0
    )

    total_correct += correct
    total_images += total

    print(
        f"{generator:15s} | "
        f"{correct:3d}/{total:3d} | "
        f"{accuracy:6.2f}%"
    )


# ============================================================
# Overall AI accuracy
# ============================================================

overall_accuracy = (
    total_correct
    / total_images
    * 100
)

print()
print("=" * 70)
print("OVERALL AI ACCURACY")
print("=" * 70)

print(
    f"Correctly detected: "
    f"{total_correct}/{total_images}"
)

print(
    f"Accuracy: "
    f"{overall_accuracy:.2f}%"
)

print("=" * 70)