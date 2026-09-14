import os
import io
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd

def letterbox_image(img, target_size=224, fill=(128, 128, 128)):
    """
    Resizes proportionally preserving aspect ratio and pads symmetrically.
    Prevents geometric distortion and preserves spatial frequency characteristics.
    """
    w, h = img.size
    if w == target_size and h == target_size:
        return img
    scale = target_size / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = img.resize((nw, nh), Image.BICUBIC)
    new_img = Image.new("RGB", (target_size, target_size), fill)
    pad_x = (target_size - nw) // 2
    pad_y = (target_size - nh) // 2
    new_img.paste(resized, (pad_x, pad_y))
    return new_img

class LetterboxTransform:
    """
    Picklable Callable transform for PyTorch multiprocessing on Windows/POSIX.
    """
    def __init__(self, target_size=224, fill=(128, 128, 128)):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img):
        return letterbox_image(img, target_size=self.target_size, fill=self.fill)

class SocialMediaCompressionTransform:
    """
    Simulates social media recompression (e.g., WhatsApp, Telegram, Instagram).
    Applies:
    1. Subtle downscale + upscale (stripping high-frequency micro-textures)
    2. Variable lossy JPEG compression (Q=50..92) with 4:2:0 subsampling
    """
    def __init__(self, p=0.45, q_min=50, q_max=92):
        self.p = p
        self.q_min = q_min
        self.q_max = q_max

    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        # 1. Random subtle downscale + upscale (35% probability)
        w, h = img.size
        if random.random() < 0.35 and min(w, h) > 64:
            scale_factor = random.uniform(0.70, 0.95)
            dw, dh = max(32, int(w * scale_factor)), max(32, int(h * scale_factor))
            img = img.resize((dw, dh), Image.BILINEAR).resize((w, h), Image.BICUBIC)
            
        # 2. JPEG recompression
        quality = random.randint(self.q_min, self.q_max)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality, subsampling=2) # 2 = 4:2:0
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

class SignalScopeDataset(Dataset):
    """
    PyTorch Dataset for SignalScope Real vs. Synthetic Image Classification.
    Labels: 0 = Real, 1 = Synthetic
    Supports loading directly from CSV split file or from directory.
    """
    def __init__(self, data_source, transform=None):
        self.data_source = Path(data_source)
        self.transform = transform
        self.samples = []

        if self.data_source.is_file() and self.data_source.suffix.lower() == ".csv":
            df = pd.read_csv(self.data_source)
            for _, row in df.iterrows():
                p = str(row["image_path"])
                lbl = int(row["label"])
                gen = str(row.get("generator", "unknown"))
                self.samples.append((p, lbl, gen))
        else:
            # Fallback to directory scan
            real_dir = self.data_source / "real"
            synth_dir = self.data_source / "synthetic"

            if real_dir.exists():
                for f in sorted(real_dir.glob("*.*")):
                    if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
                        fname = f.name.lower()
                        if "cifar" in fname:
                            gen = "real_cifar10"
                        elif "astro" in fname:
                            gen = "real_night_sky"
                        else:
                            gen = "real_imagenet"
                        self.samples.append((str(f), 0, gen))

            if synth_dir.exists():
                for f in sorted(synth_dir.glob("*.*")):
                    if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
                        fname = f.name.lower()
                        if "adm" in fname:
                            gen = "adm"
                        elif "stable_diffusion_v1_4" in fname or "sd14" in fname:
                            gen = "stable_diffusion_v1_4"
                        elif "stable_diffusion" in fname or "sd15" in fname:
                            gen = "stable_diffusion_v1_5"
                        elif "wukong" in fname:
                            gen = "wukong"
                        elif "cosmic" in fname:
                            gen = "synth_cosmic_fantasy"
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
        except Exception:
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        if self.transform:
            img = self.transform(img)

        return img, label, gen

def get_transforms(image_size=224, letterbox=True, is_train=True):
    """
    Pretrained ImageNet normalization with robust training augmentations.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    resize_op = LetterboxTransform(target_size=image_size) if letterbox else transforms.Resize((image_size, image_size))

    if is_train:
        return transforms.Compose([
            SocialMediaCompressionTransform(p=0.45, q_min=50, q_max=92),
            resize_op,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
    else:
        return transforms.Compose([
            resize_op,
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])

def get_dataloaders(cfg):
    data_cfg = cfg.get("data", {})
    image_size = data_cfg.get("image_size", 224)
    batch_size = data_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 2)
    letterbox = data_cfg.get("letterbox", True)

    train_transform = get_transforms(image_size, letterbox=letterbox, is_train=True)
    val_transform = get_transforms(image_size, letterbox=letterbox, is_train=False)

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
