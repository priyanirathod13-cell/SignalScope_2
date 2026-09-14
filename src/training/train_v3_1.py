"""
SignalScope V3.1: Final Targeted Fix Training Script
Trains EfficientNet-B0 on data/v3_1/ with:
- Aspect-Ratio Balanced Dataset (eliminates letterbox padding shortcut)
- Realistic Social Media Compression Augmentation (Q50-95, chroma subsampling, resize)
- Unfrozen Upper Blocks (6, 7, 8) with Differential Learning Rates (bb=1e-4, hd=1e-3)
- Balanced Class Weights & Early Stopping
- Saves checkpoints strictly to models/v3_1_fix/
"""

import os
import sys
import time
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(r"C:\SignalScope")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model
from src.training.dataset import get_dataloaders, compute_class_weights, SignalScopeDataset, get_transforms
from src.evaluation.metrics import calculate_metrics, format_confusion_matrix

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_device(device_setting="auto"):
    if device_setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_setting)

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, max_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(dataloader)
    t0 = time.time()

    for batch_idx, (images, labels, _) in enumerate(dataloader, 1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 40 == 0 or batch_idx == num_batches:
            batch_time = time.time() - t0
            cur_loss = running_loss / total
            cur_acc = correct / total
            cur_lr_bb = optimizer.param_groups[0]["lr"]
            cur_lr_hd = optimizer.param_groups[1]["lr"]
            print(f"  [Epoch {epoch}/{max_epochs} | Batch {batch_idx:3d}/{num_batches}] Loss: {cur_loss:.4f} | Acc: {cur_acc*100:.2f}% | LRs: (bb={cur_lr_bb:.1e}, hd={cur_lr_hd:.1e}) | Elapsed: {batch_time:.1f}s", flush=True)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    all_targets = []
    all_preds = []
    all_probs = []
    all_gens = []

    with torch.no_grad():
        for images, labels, gens in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_targets.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())
            all_gens.extend(gens)

    val_loss = running_loss / total
    metrics = calculate_metrics(all_targets, all_preds, all_probs)
    metrics["loss"] = val_loss
    return metrics, all_targets, all_preds, all_probs, all_gens

def main():
    config_path = PROJECT_ROOT / "config" / "v3_1_train_config.json"
    print(f"[V3.1 Training Engine] Loading config from {config_path}...")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    train_cfg = cfg.get("training", {})
    set_seed(train_cfg.get("seed", 42))
    device = get_device(train_cfg.get("device", "auto"))
    print(f"[V3.1 Training Engine] Active compute device: {device}")

    # Build model (blocks 6, 7, 8 unfrozen)
    model = build_model(cfg).to(device)

    # Parameter groups with differential learning rates
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    backbone_lr = train_cfg.get("backbone_lr", 1e-4)
    head_lr = train_cfg.get("head_lr", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 0.01)

    optimizer = AdamW([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr}
    ], weight_decay=weight_decay)

    # Dataloaders
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = get_dataloaders(cfg)

    # Class weights
    class_weights, real_count, synth_count = compute_class_weights(train_ds, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Class Weights: Real={class_weights[0]:.4f}, Synth={class_weights[1]:.4f}")

    max_epochs = train_cfg.get("max_epochs", 3)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    ckpt_dir = PROJECT_ROOT / train_cfg.get("checkpoint_dir", "models/v3_1_fix")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = ckpt_dir / "best_model.pt"
    final_model_path = ckpt_dir / "final_model.pt"

    best_val_auc = 0.0
    best_epoch = 0
    history = []

    print("\n" + "=" * 70)
    print(f"STARTING V3.1 TRAINING RUN ({max_epochs} EPOCHS)")
    print("=" * 70)
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        print(f"\n--- Epoch {epoch}/{max_epochs} ---")
        t_ep_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, max_epochs
        )

        val_metrics, _, _, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        ep_duration = time.time() - t_ep_start
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        val_auc = val_metrics["roc_auc"]
        val_f1 = val_metrics.get("macro_f1", val_metrics.get("f1", 0.0))

        print(f"Epoch {epoch} Complete in {ep_duration:.1f}s:", flush=True)
        print(f"  Train: Loss={train_loss:.4f} | Acc={train_acc*100:.2f}%", flush=True)
        print(f"  Val:   Loss={val_loss:.4f} | Acc={val_acc*100:.2f}% | AUC={val_auc:.4f} | F1={val_f1*100:.2f}%", flush=True)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_roc_auc": val_auc,
            "val_f1": val_f1,
            "duration_seconds": ep_duration
        }
        history.append(record)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            print(f"  >> New Best Val AUC ({best_val_auc:.4f})! Saving to {best_model_path}")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": best_val_auc,
                "val_acc": val_acc,
                "config": cfg,
                "version": "V3.1"
            }, best_model_path)

    # Save final model
    torch.save({
        "epoch": max_epochs,
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "version": "V3.1"
    }, final_model_path)

    total_training_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"V3.1 TRAINING COMPLETE in {total_training_time/60:.2f} minutes")
    print(f"Best Val AUC: {best_val_auc:.4f} (Epoch {best_epoch})")
    print(f"Checkpoints: {best_model_path} and {final_model_path}")
    print("=" * 70)

    # Save training history
    with open(ckpt_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
