import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from collections import defaultdict
import numpy as np


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
# Transform
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
# Identify generator
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
# Store results
# ============================================================

results = defaultdict(list)


# ============================================================
# Find images
# ============================================================

images = []

for extension in [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.webp"
]:

    images.extend(
        AI_TEST_DIR.glob(extension)
    )

images = sorted(images)

print()
print("=" * 70)
print("SignalScope V3 - Confidence Analysis")
print("=" * 70)

print(
    f"\nAI test images: {len(images)}"
)


# ============================================================
# Prediction
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

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

        prediction = torch.argmax(
            probabilities
        ).item()

        ai_probability = (
            probabilities[0]
            .item()
            * 100
        )

        real_probability = (
            probabilities[1]
            .item()
            * 100
        )

        confidence = max(
            ai_probability,
            real_probability
        )

        # Since every image in this folder
        # is actually AI-generated:
        correct = prediction == 0

        results[generator].append({
            "confidence": confidence,
            "ai_probability": ai_probability,
            "real_probability": real_probability,
            "correct": correct
        })


# ============================================================
# Analysis
# ============================================================

print()
print("=" * 70)
print("CONFIDENCE ANALYSIS BY GENERATOR")
print("=" * 70)


total_images = 0
total_correct = 0

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

    data = results[generator]

    confidences = [
        item["confidence"]
        for item in data
    ]

    correct_confidences = [
        item["confidence"]
        for item in data
        if item["correct"]
    ]

    wrong_confidences = [
        item["confidence"]
        for item in data
        if not item["correct"]
    ]

    total = len(data)
    correct = len(correct_confidences)
    wrong = len(wrong_confidences)

    accuracy = (
        correct / total * 100
    )

    average_confidence = (
        np.mean(confidences)
    )

    average_correct_confidence = (
        np.mean(correct_confidences)
        if correct_confidences
        else 0
    )

    average_wrong_confidence = (
        np.mean(wrong_confidences)
        if wrong_confidences
        else 0
    )

    total_images += total
    total_correct += correct

    print()
    print(f"{generator}")
    print("-" * 70)

    print(
        f"Images                  : {total}"
    )

    print(
        f"Accuracy                : {accuracy:.2f}%"
    )

    print(
        f"Average confidence     : "
        f"{average_confidence:.2f}%"
    )

    print(
        f"Correct avg confidence : "
        f"{average_correct_confidence:.2f}%"
    )

    print(
        f"Wrong avg confidence   : "
        f"{average_wrong_confidence:.2f}%"
    )


# ============================================================
# Overall analysis
# ============================================================

overall_accuracy = (
    total_correct
    / total_images
    * 100
)

all_results = []

for generator_data in results.values():
    all_results.extend(generator_data)

all_confidences = [
    item["confidence"]
    for item in all_results
]

all_correct_confidences = [
    item["confidence"]
    for item in all_results
    if item["correct"]
]

all_wrong_confidences = [
    item["confidence"]
    for item in all_results
    if not item["correct"]
]


print()
print("=" * 70)
print("OVERALL CONFIDENCE ANALYSIS")
print("=" * 70)

print(
    f"Total images            : {total_images}"
)

print(
    f"Correctly detected      : "
    f"{total_correct}"
)

print(
    f"Incorrectly detected    : "
    f"{total_images - total_correct}"
)

print(
    f"Overall accuracy        : "
    f"{overall_accuracy:.2f}%"
)

print(
    f"Average confidence      : "
    f"{np.mean(all_confidences):.2f}%"
)

print(
    f"Correct avg confidence  : "
    f"{np.mean(all_correct_confidences):.2f}%"
)

print(
    f"Wrong avg confidence    : "
    f"{np.mean(all_wrong_confidences):.2f}%"
)

print("=" * 70)