import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd

class SignalScopeDataset(Dataset):
    """
    PyTorch Dataset for SignalScope Real vs. Synthetic Image Classification.
    Labels: 0 = Real, 1 = Synthetic
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []

        # Read from real (0) and synthetic (1) folders
        real_dir = self.data_dir / "real"
        synth_dir = self.data_dir / "synthetic"

        if real_dir.exists():
            for f in sorted(real_dir.glob("*.*")):
                if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    self.samples.append((str(f), 0, "real_imagenet"))

        if synth_dir.exists():
            for f in sorted(synth_dir.glob("*.*")):
                if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    fname = f.name.lower()
                    if "adm" in fname:
                        gen = "adm"
                    elif "stable_diffusion" in fname or "sd15" in fname:
                        gen = "stable_diffusion_v1_5"
                    elif "wukong" in fname:
                        gen = "wukong"
                    else:
                        gen = "unknown_synthetic"
                    self.samples.append((str(f), 1, gen))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, gen = self.samples[idx]
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
        except Exception as e:
            # Fallback for unexpected read errors: return black image
            img = Image.new("RGB", (224, 224), color=0)

        if self.transform:
            img = self.transform(img)

        return img, label, gen

def get_transforms(image_size=224):
    """
    Pretrained ImageNet normalization.
    Train: gentle flips and mild color jitter (safe for forensic artifacts).
    Val/Test: deterministic resize and normalization.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    return train_transform, val_transform

def get_dataloaders(cfg):
    data_cfg = cfg.get("data", {})
    image_size = data_cfg.get("image_size", 224)
    batch_size = data_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 2)

    train_transform, val_transform = get_transforms(image_size)

    train_ds = SignalScopeDataset(data_cfg["train_dir"], transform=train_transform)
    val_ds = SignalScopeDataset(data_cfg["val_dir"], transform=val_transform)
    test_ds = SignalScopeDataset(data_cfg["test_dir"], transform=val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds

def compute_class_weights(dataset, device="cpu"):
    labels = [s[1] for s in dataset.samples]
    total = len(labels)
    real_count = labels.count(0)
    synth_count = labels.count(1)

    weight_real = total / (2.0 * real_count) if real_count > 0 else 1.0
    weight_synth = total / (2.0 * synth_count) if synth_count > 0 else 1.0

    weights = torch.tensor([weight_real, weight_synth], dtype=torch.float32, device=device)
    return weights, real_count, synth_count
