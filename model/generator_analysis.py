from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image


# ============================================================
# SignalScope - Generator-wise Error Analysis
# ============================================================

PROJECT_DIR = Path(r"D:\SignalScope")

TEST_DIR = PROJECT_DIR / "data" / "dataset" / "test"
MODEL_PATH = PROJECT_DIR / "model" / "signalscope_resnet18.pth"

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Generator names
# ============================================================

GENERATORS = [
    "adm",
    "biggan",
    "glide",
    "midjourney",
    "sd15",
    "vqdm",
    "wukong"
]


# ============================================================
# Image transformation
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

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
print("SignalScope - Generator-wise Analysis")
print("=" * 60)

print(f"\nDevice: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


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

print("Model loaded successfully.")


# ============================================================
# Find test images
# ============================================================

ai_dir = TEST_DIR / "ai"

ai_images = list(ai_dir.glob("*.jpg"))

print(f"\nAI test images found: {len(ai_images)}")


# ============================================================
# Analyze each generator
# ============================================================

results = defaultdict(lambda: {
    "total": 0,
    "correct": 0,
    "wrong": 0,
    "confidences": []
})


for image_path in ai_images:

    filename = image_path.name.lower()

    # Determine generator from filename
    generator = None

    for name in GENERATORS:

        if f"ai_{name}_" in filename:
            generator = name
            break

    if generator is None:
        print(
            f"Warning: Could not identify generator: "
            f"{image_path.name}"
        )
        continue


    # Load image
    image = Image.open(
        image_path
    ).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(DEVICE)


    # Prediction
    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

        prediction = torch.argmax(
            probabilities
        ).item()


    # AI = class 0
    ai_confidence = (
        probabilities[0].item() * 100
    )

    results[generator]["total"] += 1
    results[generator]["confidences"].append(
        ai_confidence
    )


    if prediction == 0:

        results[generator]["correct"] += 1

    else:

        results[generator]["wrong"] += 1


# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 60)
print("GENERATOR-WISE RESULTS")
print("=" * 60)

print(
    f"{'Generator':<15}"
    f"{'Total':<10}"
    f"{'Correct':<10}"
    f"{'Wrong':<10}"
    f"{'Accuracy':<12}"
    f"{'Avg AI Conf.':<12}"
)

print("-" * 69)


overall_total = 0
overall_correct = 0


for generator in GENERATORS:

    data = results[generator]

    total = data["total"]
    correct = data["correct"]
    wrong = data["wrong"]

    if total > 0:

        accuracy = (
            correct / total * 100
        )

        avg_confidence = (
            sum(data["confidences"])
            / len(data["confidences"])
        )

    else:

        accuracy = 0
        avg_confidence = 0


    overall_total += total
    overall_correct += correct


    print(
        f"{generator:<15}"
        f"{total:<10}"
        f"{correct:<10}"
        f"{wrong:<10}"
        f"{accuracy:<12.2f}"
        f"{avg_confidence:<12.2f}"
    )


print("-" * 69)

overall_accuracy = (
    overall_correct
    / overall_total
    * 100
)

print(
    f"{'TOTAL':<15}"
    f"{overall_total:<10}"
    f"{overall_correct:<10}"
    f"{overall_total - overall_correct:<10}"
    f"{overall_accuracy:<12.2f}"
)


# ============================================================
# Find hardest generator
# ============================================================

valid_generators = [
    generator
    for generator in GENERATORS
    if results[generator]["total"] > 0
]

hardest_generator = min(
    valid_generators,
    key=lambda g:
    results[g]["correct"]
    / results[g]["total"]
)


print("\n" + "=" * 60)

print(
    f"Hardest generator: "
    f"{hardest_generator.upper()}"
)

hardest_accuracy = (
    results[hardest_generator]["correct"]
    / results[hardest_generator]["total"]
    * 100
)

print(
    f"Detection accuracy: "
    f"{hardest_accuracy:.2f}%"
)

print("=" * 60)