from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights

import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# SignalScope - Error Analysis
# ============================================================

PROJECT_DIR = Path(r"D:\SignalScope")

TEST_DIR = PROJECT_DIR / "data" / "dataset" / "test"
MODEL_PATH = PROJECT_DIR / "model" / "signalscope_resnet18.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Transform
# ============================================================

test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Dataset
# ============================================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transforms
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("=" * 60)
print("SignalScope - Error Analysis")
print("=" * 60)

print(f"\nDevice: {DEVICE}")
print(f"Test images: {len(test_dataset)}")
print(f"Class mapping: {test_dataset.class_to_idx}")


# ============================================================
# Model
# ============================================================

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
# Find incorrect predictions
# ============================================================

errors = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        for i in range(len(labels)):

            actual = labels[i].item()
            predicted = predictions[i].item()

            if actual != predicted:

                # Probability of predicted class
                confidence = (
                    probabilities[i][predicted]
                    .item()
                    * 100
                )

                image_index = (
                    test_loader.batch_size
                    * (
                        len(errors)
                    )
                )

                errors.append({
                    "actual": actual,
                    "predicted": predicted,
                    "confidence": confidence
                })


# ============================================================
# Better error collection with direct dataset indexing
# ============================================================

errors = []

for index in range(len(test_dataset)):

    image_tensor, actual = test_dataset[index]

    image_input = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(image_input)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

        predicted = torch.argmax(
            probabilities
        ).item()

    if actual != predicted:

        confidence = (
            probabilities[predicted].item()
            * 100
        )

        image_path = test_dataset.samples[index][0]

        errors.append({
            "path": image_path,
            "actual": actual,
            "predicted": predicted,
            "confidence": confidence
        })


# ============================================================
# Print errors
# ============================================================

print("\n" + "=" * 60)
print(f"TOTAL MISCLASSIFICATIONS: {len(errors)}")
print("=" * 60)

for i, error in enumerate(errors, start=1):

    actual_name = (
        "AI"
        if error["actual"] == 0
        else "Real"
    )

    predicted_name = (
        "AI"
        if error["predicted"] == 0
        else "Real"
    )

    print(
        f"{i:02d}. "
        f"{Path(error['path']).name}"
    )

    print(
        f"    Actual:    {actual_name}"
    )

    print(
        f"    Predicted: {predicted_name}"
    )

    print(
        f"    Confidence: "
        f"{error['confidence']:.2f}%"
    )

    print()


# ============================================================
# Display errors
# ============================================================

if errors:

    number_to_show = min(
        len(errors),
        15
    )

    selected_errors = errors[:number_to_show]

    columns = 5
    rows = (
        number_to_show + columns - 1
    ) // columns

    plt.figure(
        figsize=(15, 3 * rows)
    )

    for i, error in enumerate(
        selected_errors
    ):

        image = Image.open(
            error["path"]
        ).convert("RGB")

        ax = plt.subplot(
            rows,
            columns,
            i + 1
        )

        ax.imshow(image)

        actual_name = (
            "AI"
            if error["actual"] == 0
            else "Real"
        )

        predicted_name = (
            "AI"
            if error["predicted"] == 0
            else "Real"
        )

        ax.set_title(
            f"Actual: {actual_name}\n"
            f"Predicted: {predicted_name}\n"
            f"Confidence: "
            f"{error['confidence']:.1f}%",
            fontsize=9
        )

        ax.axis("off")

    plt.suptitle(
        "SignalScope - Model Errors",
        fontsize=18
    )

    plt.tight_layout()

    output_path = (
        PROJECT_DIR
        / "evaluation"
        / "error_analysis.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    print(
        f"Error visualization saved to:"
        f"\n{output_path}"
    )

    plt.show()