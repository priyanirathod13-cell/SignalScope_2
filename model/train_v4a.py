from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(r"D:\SignalScope")

DATA_DIR = (
    PROJECT_DIR
    / "data"
    / "dataset_v4a"
)

MODEL_PATH = (
    PROJECT_DIR
    / "model"
    / "signalscope_resnet18_v4a.pth"
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
# Image transformations
# ============================================================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        224,
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
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    transforms.RandomErasing(
        p=0.2
    )
])


val_transform = transforms.Compose([
    transforms.Resize(
        (224, 224)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Datasets
# ============================================================

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=val_transform
)


print()
print("Class mapping:")
print(train_dataset.class_to_idx)

print()
print("Train images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ============================================================
# Model
# ============================================================

print()
print("Loading pretrained ResNet18...")

weights = ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
)


# Freeze everything first

for parameter in model.parameters():
    parameter.requires_grad = False


# Unfreeze deeper layers

for parameter in model.layer3.parameters():
    parameter.requires_grad = True

for parameter in model.layer4.parameters():
    parameter.requires_grad = True


# Replace classifier

model.fc = nn.Linear(
    model.fc.in_features,
    2
)


model = model.to(device)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    [
        {
            "params": model.layer3.parameters(),
            "lr": 1e-5
        },
        {
            "params": model.layer4.parameters(),
            "lr": 1e-5
        },
        {
            "params": model.fc.parameters(),
            "lr": 1e-4
        }
    ],
    weight_decay=1e-4
)


# ============================================================
# Training
# ============================================================

epochs = 15

best_val_accuracy = 0.0


print()
print("=" * 60)
print("STARTING V4-A TRAINING")
print("=" * 60)


for epoch in range(epochs):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
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


        running_loss += (
            loss.item()
            * images.size(0)
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        correct / total
    ) * 100


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)


    val_accuracy = (
        val_correct / val_total
    ) * 100


    print(
        f"Epoch {epoch + 1:02d}/{epochs} | "
        f"Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Acc: {val_accuracy:.2f}%"
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
            f"  ✓ Best model saved "
            f"({best_val_accuracy:.2f}%)"
        )


# ============================================================
# Finished
# ============================================================

print()
print("=" * 60)
print("V4-A TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Model saved to:\n"
    f"{MODEL_PATH}"
)