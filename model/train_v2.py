from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights


# ============================================================
# SignalScope - ResNet-18 V2
# Fine-tuning + Stronger Augmentation
# ============================================================

PROJECT_DIR = Path(r"D:\SignalScope")

DATA_DIR = PROJECT_DIR / "data" / "dataset"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

MODEL_DIR = PROJECT_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "signalscope_resnet18_v2.pth"


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 15

BACKBONE_LR = 0.00001
CLASSIFIER_LR = 0.0001

NUM_CLASSES = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Device information
# ============================================================

print("=" * 60)
print("SignalScope - ResNet-18 V2 Training")
print("=" * 60)

print(f"\nDevice: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"CUDA memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )


# ============================================================
# Data augmentation
# ============================================================

train_transforms = transforms.Compose([

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.75, 1.0)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),

    transforms.ToTensor(),

    transforms.RandomErasing(
        p=0.15,
        scale=(0.02, 0.15),
        ratio=(0.3, 3.3)
    ),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transforms = transforms.Compose([

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
# Load datasets
# ============================================================

print("\nLoading datasets...")

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transforms
)

print(f"Training images:   {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")

print(
    f"Class mapping: "
    f"{train_dataset.class_to_idx}"
)


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ============================================================
# Load pretrained ResNet-18
# ============================================================

print("\nLoading pretrained ResNet-18...")

weights = ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
)


# ============================================================
# Freeze everything first
# ============================================================

for parameter in model.parameters():
    parameter.requires_grad = False


# ============================================================
# Unfreeze layer3 and layer4
# ============================================================

for parameter in model.layer3.parameters():
    parameter.requires_grad = True

for parameter in model.layer4.parameters():
    parameter.requires_grad = True


# ============================================================
# Replace classifier
# ============================================================

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)


# Classifier must be trainable
for parameter in model.fc.parameters():
    parameter.requires_grad = True


model = model.to(DEVICE)

print("ResNet-18 V2 loaded.")
print("Trainable layers: layer3, layer4, classifier")


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# Optimizer
# Different learning rates for backbone and classifier
# ============================================================

optimizer = optim.AdamW(
    [
        {
            "params": model.layer3.parameters(),
            "lr": BACKBONE_LR
        },
        {
            "params": model.layer4.parameters(),
            "lr": BACKBONE_LR
        },
        {
            "params": model.fc.parameters(),
            "lr": CLASSIFIER_LR
        }
    ],
    weight_decay=0.0001
)


# ============================================================
# Training
# ============================================================

best_val_accuracy = 0.0

print("\nStarting V2 training...")
print("=" * 60)

for epoch in range(NUM_EPOCHS):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()


    train_loss = (
        running_loss
        / len(train_loader)
    )

    train_accuracy = (
        100 * correct / total
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss_total += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_total += labels.size(0)

            val_correct += (
                predictions == labels
            ).sum().item()


    val_loss = (
        val_loss_total
        / len(val_loader)
    )

    val_accuracy = (
        100 * val_correct / val_total
    )


    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_accuracy:.2f}% "
        f"| Val Loss: {val_loss:.4f} "
        f"| Val Acc: {val_accuracy:.2f}%"
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            f"  ✓ V2 model saved "
            f"(Val Acc: {val_accuracy:.2f}%)"
        )


# ============================================================
# Finished
# ============================================================

print("\n" + "=" * 60)
print("V2 TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Model saved to:\n"
    f"{MODEL_PATH}"
)