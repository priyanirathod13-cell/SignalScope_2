import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_DIR / "data" / "splits" / "train"
VAL_DIR = PROJECT_DIR / "data" / "splits" / "validation"

MODEL_PATH = PROJECT_DIR / "model" / "signalscope_resnet18_v3.pth"

BATCH_SIZE = 32
EPOCHS = 15

# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# --------------------------------------------------
# Data transforms
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.15),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# Load datasets
# --------------------------------------------------

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

print("\nClass mapping:")
print(train_dataset.class_to_idx)

print("\nTraining images:", len(train_dataset))
print("Validation images:", len(val_dataset))

# --------------------------------------------------
# Data loaders
# --------------------------------------------------

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

# --------------------------------------------------
# Model
# --------------------------------------------------

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# Freeze the early layers
for param in model.parameters():
    param.requires_grad = False

# Fine-tune deeper layers
for param in model.layer3.parameters():
    param.requires_grad = True

for param in model.layer4.parameters():
    param.requires_grad = True

# Replace classifier
model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model = model.to(device)

# --------------------------------------------------
# Optimizer
# --------------------------------------------------

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

criterion = nn.CrossEntropyLoss()

# --------------------------------------------------
# Training
# --------------------------------------------------

best_val_accuracy = 0.0

print("\nStarting V3 training...\n")

for epoch in range(EPOCHS):

    # -----------------------------
    # Training
    # -----------------------------

    model.train()

    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        _, predictions = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    train_accuracy = 100 * correct / total

    # -----------------------------
    # Validation
    # -----------------------------

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predictions = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    val_accuracy = 100 * correct / total

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Acc: {val_accuracy:.2f}%"
    )

    # -----------------------------
    # Save best model
    # -----------------------------

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

print("\n========================================")
print("V3 TRAINING COMPLETE")
print("========================================")
print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
print(f"Model saved to: {MODEL_PATH}")