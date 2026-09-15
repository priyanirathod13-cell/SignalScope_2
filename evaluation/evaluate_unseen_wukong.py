from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = PROJECT_DIR / "data" / "unseen_wukong" / "test"

MODEL_PATH = (
    PROJECT_DIR
    / "model"
    / "signalscope_resnet18_unseen_wukong.pth"
)

# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)
print("Test directory:", TEST_DIR)
print("Model:", MODEL_PATH)

# --------------------------------------------------
# IMAGE TRANSFORM
# --------------------------------------------------

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# --------------------------------------------------
# DATASET
# --------------------------------------------------

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform,
)

print("Classes:", test_dataset.class_to_idx)
print("Test images:", len(test_dataset))

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)

# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = models.resnet18(weights=None)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    2,
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

all_labels = []
all_predictions = []
all_ai_probabilities = []

print("\nRunning inference...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        predictions = torch.argmax(
            probabilities,
            dim=1,
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        # ImageFolder:
        # real = 0
        # synthetic = 1
        all_ai_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
        )

print("Inference complete.")

# --------------------------------------------------
# METRICS
# --------------------------------------------------

y_true = np.asarray(all_labels)
y_pred = np.asarray(all_predictions)
y_ai_prob = np.asarray(
    all_ai_probabilities
)

accuracy = accuracy_score(
    y_true,
    y_pred,
)

precision = precision_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0,
)

recall = recall_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0,
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
)

roc_auc = roc_auc_score(
    y_true,
    y_ai_prob,
)

cm = confusion_matrix(
    y_true,
    y_pred,
)

# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("UNSEEN GENERATOR RESULTS — WUKONG")
print("=" * 60)

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

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "Real",
            "AI",
        ],
        zero_division=0,
    )
)

print("CONFUSION MATRIX")

print(cm)

print("\nCLASS MAPPING")
print("0 = Real")
print("1 = AI / Synthetic")

print("\nTest set:")
print("Real images    :", np.sum(y_true == 0))
print("Wukong images  :", np.sum(y_true == 1))

print("\nEvaluation complete.")