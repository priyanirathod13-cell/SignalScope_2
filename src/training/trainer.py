import os
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models import build_model
from src.training.dataset import get_dataloaders, compute_class_weights
from src.evaluation.metrics import calculate_metrics

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_device(device_setting="auto"):
    if device_setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_setting)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in dataloader:
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

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1] # Probability of Class 1 (Synthetic)
            _, preds = torch.max(outputs, 1)

            all_targets.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    val_loss = running_loss / total
    metrics = calculate_metrics(all_targets, all_preds, all_probs)
    metrics["loss"] = round(val_loss, 4)
    return metrics, all_targets, all_preds, all_probs

def train_model(cfg):
    print("=" * 65)
    print("   SIGNALSCOPE: BASELINE MODEL TRAINING PIPELINE")
    print("=" * 65)

    training_cfg = cfg.get("training", {})
    seed = training_cfg.get("seed", 42)
    set_seed(seed)

    device = get_device(training_cfg.get("device", "auto"))
    print(f"Execution Device: {device}")

    # DataLoaders
    print("\n[1/4] Loading and preparing datasets...")
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = get_dataloaders(cfg)
    print(f"  Training samples:   {len(train_ds)}")
    print(f"  Validation samples: {len(val_ds)}")
    print(f"  Test samples:       {len(test_ds)}")

    # Class weights
    class_weights, real_cnt, synth_cnt = compute_class_weights(train_ds, device=device)
    print(f"  Train class balance: {real_cnt} Real vs. {synth_cnt} Synthetic")
    print(f"  Class Weights: Real={class_weights[0]:.3f}, Synthetic={class_weights[1]:.3f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Model
    print("\n[2/4] Building EfficientNet-B0 transfer learning architecture...")
    model = build_model(cfg).to(device)

    # Optimizer & Scheduler
    lr = training_cfg.get("learning_rate", 1e-3)
    wd = training_cfg.get("weight_decay", 1e-2)
    epochs = training_cfg.get("epochs", 3)
    
    # Trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable parameters: {sum(p.numel() for p in trainable_params):,} / {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(trainable_params, lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "models/baseline"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_roc_auc": [],
        "val_macro_f1": []
    }

    best_val_auc = 0.0
    best_epoch = 0

    print(f"\n[3/4] Starting training for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        print(f"\n--- Epoch {epoch}/{epochs} (lr={optimizer.param_groups[0]['lr']:.6f}) ---")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        val_auc = val_metrics["roc_auc"]
        val_f1 = val_metrics["macro_f1"]

        history["epochs"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_roc_auc"].append(val_auc)
        history["val_macro_f1"].append(val_f1)

        epoch_time = time.time() - epoch_start
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}% | Val ROC-AUC: {val_auc:.4f} | Val Macro-F1: {val_f1:.4f}")
        print(f"  Epoch Duration: {epoch_time:.1f}s")

        # Checkpoint if best ROC-AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_ckpt_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": cfg
            }, best_ckpt_path)
            print(f"  >>> Best model saved to {best_ckpt_path} (Val AUC: {best_val_auc:.4f}) <<<")

    # Save final model
    final_ckpt_path = checkpoint_dir / "final_model.pt"
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "final_metrics": val_metrics,
        "config": cfg
    }, final_ckpt_path)
    print(f"\nFinal model saved to {final_ckpt_path}")

    # Save history
    history_path = checkpoint_dir / "train_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    total_training_time = time.time() - start_time
    print(f"\nTotal Training Completed in {total_training_time/60:.2f} minutes.")
    print(f"Best Validation ROC-AUC: {best_val_auc:.4f} at Epoch {best_epoch}")

    return model, history, best_val_auc
