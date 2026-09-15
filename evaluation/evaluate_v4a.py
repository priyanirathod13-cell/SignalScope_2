from pathlib import Path
import sys

# Add project root to Python path
PROJECT_DIR = Path(r"D:\SignalScope")
sys.path.insert(0, str(PROJECT_DIR))

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from model.model import load_model


# ============================================================
# Configuration
# ============================================================

DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "dataset_v4a"
    / "test"
)

MODEL_PATH = (
    PROJECT_DIR
    / "model"
    / "signalscope_resnet18_v4a.pth"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "evaluation"
    / "confusion_matrix_v4a.png"
)


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
# Transform
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
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
    DATA_DIR,
    transform=test_transform
)

print()
print("Class mapping:")
print(test_dataset.class_to_idx)

print()
print("Test images:", len(test_dataset))


test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ============================================================
# Load model
# ============================================================

print()
print("Loading V4-A model...")

model = load_model(
    MODEL_PATH
)

model = model.to(device)

model.eval()

print("Model loaded successfully.")


# ============================================================
# Prediction
# ============================================================

all_labels = []
all_predictions = []
all_real_probabilities = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        # Class mapping:
        # AI = 0
        # Real = 1
        all_real_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
        )


# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

roc_auc = roc_auc_score(
    all_labels,
    all_real_probabilities
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# ============================================================
# Results
# ============================================================

print()
print("=" * 70)
print("V4-A INTERNAL TEST RESULTS")
print("=" * 70)

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"Macro-F1 : {macro_f1 * 100:.2f}%"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)

print()
print("Confusion Matrix:")
print(cm)

print()
print("Classification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=[
            "AI",
            "Real"
        ],
        zero_division=0
    )
)


# ============================================================
# Save confusion matrix
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "AI",
        "Real"
    ]
)

fig, ax = plt.subplots(
    figsize=(7, 7)
)

disp.plot(
    ax=ax,
    values_format="d"
)

ax.set_title(
    "SignalScope V4-A Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print()
print(
    "Confusion matrix saved to:"
)

print(
    OUTPUT_PATH
)

print("=" * 70)