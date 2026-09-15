from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt


# ============================================================
# SignalScope - Model Evaluation
# ============================================================

PROJECT_DIR = Path(r"D:\SignalScope")

TEST_DIR = PROJECT_DIR / "data" / "dataset" / "test"
MODEL_PATH = PROJECT_DIR / "model" / "signalscope_resnet18_v2.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Device
# ============================================================

print("=" * 60)
print("SignalScope - Model Evaluation")
print("=" * 60)

print(f"\nDevice: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# Test transformation
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
# Load test dataset
# ============================================================

print("\nLoading test dataset...")

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

print(f"Test images: {len(test_dataset)}")
print(f"Class mapping: {test_dataset.class_to_idx}")


# ============================================================
# Load model
# ============================================================

print("\nLoading trained ResNet-18...")

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
# Run inference
# ============================================================

print("\nRunning inference...")

all_labels = []
all_predictions = []
all_probabilities = []

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

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        # Probability of AI class
        ai_probabilities = probabilities[:, 0]

        all_probabilities.extend(
            ai_probabilities.cpu().numpy()
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
    [1 - p for p in all_probabilities]
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# ============================================================
# Results
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

print(f"\nAccuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"Macro-F1 : {macro_f1 * 100:.2f}%")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=["AI", "Real"],
        zero_division=0
    )
)


# ============================================================
# Save confusion matrix
# ============================================================

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("SignalScope - Confusion Matrix")

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(
    [0, 1],
    ["AI", "Real"]
)

plt.yticks(
    [0, 1],
    ["AI", "Real"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

output_path = (
    PROJECT_DIR
    / "evaluation"
    / "confusion_matrix.png"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(output_path)

print(
    f"\nConfusion matrix saved to:"
    f"\n{output_path}"
)