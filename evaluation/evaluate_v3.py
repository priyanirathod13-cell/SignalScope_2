import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path

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
import numpy as np


# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = PROJECT_DIR / "data" / "splits" / "test"
MODEL_PATH = PROJECT_DIR / "model" / "signalscope_resnet18_v3.pth"

BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


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
# Load test dataset
# ============================================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("\nClass mapping:")
print(test_dataset.class_to_idx)

print("\nTest images:", len(test_dataset))


# ============================================================
# Load V3 model
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


# ============================================================
# Prediction
# ============================================================

all_labels = []
all_predictions = []
all_ai_probabilities = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        predictions = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(
            predictions.cpu().numpy()
        )

        # Class mapping:
        # Real      = 0
        # Synthetic = 1
        #
        # For ROC-AUC, use probability
        # of the positive class = Synthetic/AI.
        all_ai_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
        )

print("DEBUG labels:", len(all_labels))
print("DEBUG predictions:", len(all_predictions))
print("DEBUG AI probabilities:", len(all_ai_probabilities))


# Convert to NumPy arrays

y_true = np.array(all_labels)
y_pred = np.array(all_predictions)
y_ai_prob = np.asarray(all_ai_probabilities).reshape(-1)


# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

roc_auc = roc_auc_score(
    y_true,
    y_ai_prob
)


# ============================================================
# Print results
# ============================================================

print("\n========================================")
print("V3 TEST RESULTS")
print("========================================")

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"Macro-F1 : {macro_f1 * 100:.2f}%")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# Classification report
# ============================================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Real", "AI"],
        zero_division=0
    )
)


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print(cm)


# ============================================================
# Save confusion matrix image
# ============================================================

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("SignalScope V3 - Confusion Matrix")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    [0, 1],
    ["Real", "AI"]
)

plt.yticks(
    [0, 1],
    ["Real", "AI"]
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

OUTPUT_PATH = (
    PROJECT_DIR
    / "evaluation"
    / "confusion_matrix_v3.png"
)

plt.savefig(
    OUTPUT_PATH,
    dpi=200
)

plt.close()

print(
    f"\nConfusion matrix saved to:\n{OUTPUT_PATH}"
)

print("\n========================================")
print("V3 EVALUATION COMPLETE")
print("========================================")