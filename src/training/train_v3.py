"""
SignalScope V3: Final Candidate Training & Evaluation Pipeline
- Trains EfficientNet-B0 on data/v3/splits/ (14,759 train, 3,161 val, 3,165 test)
- Letterboxed aspect-ratio preserving preprocessing (no 2:1 distortion)
- Unfrozen upper blocks (6, 7, 8) + Classifier head with differential learning rates:
    * Backbone blocks 6-8: 1e-4
    * Classifier head: 1e-3
- Anti-shortcut balance: Astrophotography vs Cosmic Fantasy Art
- Evaluates on:
    1) Development Test Set (3,165 images) across all generators
    2) Sacred Unseen Generator Holdout Set (1,916 images: 958 Real, 958 Wukong Synthetic)
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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

        if batch_idx % 50 == 0 or batch_idx == num_batches:
            batch_time = time.time() - t0
            cur_loss = running_loss / total
            cur_acc = correct / total
            cur_lr_bb = optimizer.param_groups[0]["lr"]
            cur_lr_hd = optimizer.param_groups[1]["lr"]
            print(f"  [Epoch {epoch}/{max_epochs} | Batch {batch_idx:3d}/{num_batches}] Loss: {cur_loss:.4f} | Acc: {cur_acc*100:.2f}% | LRs: (bb={cur_lr_bb:.1e}, hd={cur_lr_hd:.1e}) | Elapsed: {batch_time:.1f}s")

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

            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)

            all_targets.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_gens.extend(gens)

    val_loss = running_loss / total if total > 0 else 0.0
    metrics = calculate_metrics(all_targets, all_preds, all_probs)
    metrics["loss"] = round(val_loss, 4)
    return metrics, all_targets, all_preds, all_probs, all_gens

def train_v3(cfg_path="config/v3_train_config.json"):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    print("=" * 75)
    print("   SIGNALSCOPE V3: FINAL CANDIDATE TRAINING & GENERALIZATION PIPELINE")
    print("=" * 75)

    training_cfg = cfg.get("training", {})
    seed = training_cfg.get("seed", 42)
    set_seed(seed)

    device = get_device(training_cfg.get("device", "auto"))
    print(f"Execution Device: {device}")

    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "models/v3_final_candidate"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with open(checkpoint_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print("\n[1/5] Loading and preparing V3 stratified datasets (Letterbox Enabled)...")
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = get_dataloaders(cfg)
    print(f"  Training samples:   {len(train_ds):,} images")
    print(f"  Validation samples: {len(val_ds):,} images")
    print(f"  Test samples:       {len(test_ds):,} images")

    class_weights, real_cnt, synth_cnt = compute_class_weights(train_ds, device=device)
    print(f"  Train class balance: {real_cnt:,} Real vs. {synth_cnt:,} Synthetic")
    print(f"  Class Weights: Real={class_weights[0]:.4f}, Synthetic={class_weights[1]:.4f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("\n[2/5] Building EfficientNet-B0 transfer learning architecture (Blocks 6-8 unfrozen)...")
    model = build_model(cfg).to(device)

    # Differential learning rate setup
    backbone_params = [p for p in model.backbone.features.parameters() if p.requires_grad]
    head_params = [p for p in model.backbone.classifier.parameters() if p.requires_grad]

    total_params = sum(p.numel() for p in model.parameters())
    bb_trainable = sum(p.numel() for p in backbone_params)
    hd_trainable = sum(p.numel() for p in head_params)
    print(f"  Total parameters:             {total_params:,}")
    print(f"  Trainable backbone params:    {bb_trainable:,} (Blocks 6, 7, 8)")
    print(f"  Trainable classifier params:  {hd_trainable:,} (Linear Head)")
    print(f"  Total trainable params:       {bb_trainable + hd_trainable:,} ({(bb_trainable+hd_trainable)/total_params*100:.1f}%)")

    bb_lr = training_cfg.get("backbone_lr", 1e-4)
    hd_lr = training_cfg.get("head_lr", 1e-3)
    wd = training_cfg.get("weight_decay", 1e-2)
    max_epochs = training_cfg.get("max_epochs", 3)
    patience = training_cfg.get("early_stopping_patience", 2)

    optimizer = AdamW([
        {"params": backbone_params, "lr": bb_lr},
        {"params": head_params, "lr": hd_lr}
    ], weight_decay=wd)

    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

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
    patience_counter = 0

    print(f"\n[3/5] Starting V3 training for {max_epochs} epochs (Patience={patience})...")
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        cur_bb_lr = optimizer.param_groups[0]["lr"]
        cur_hd_lr = optimizer.param_groups[1]["lr"]
        print(f"\n--- Epoch {epoch}/{max_epochs} (Backbone LR: {cur_bb_lr:.6f} | Head LR: {cur_hd_lr:.6f}) ---")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, max_epochs)
        val_metrics, _, _, _, _ = evaluate(model, val_loader, criterion, device)
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

        epoch_duration = time.time() - epoch_start
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val ROC-AUC: {val_auc:.4f} | Val Macro-F1: {val_f1:.4f}")
        print(f"  Epoch Duration: {epoch_duration:.1f}s ({epoch_duration/60:.2f} min)")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            best_ckpt_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": cfg,
                "version": "V3"
            }, best_ckpt_path)
            print(f"  >>> NEW BEST MODEL saved to {best_ckpt_path} (Val ROC-AUC: {best_val_auc:.4f}) <<<")
        else:
            patience_counter += 1
            print(f"  Validation AUC did not improve (patience {patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\n[Early Stopping] Triggered after {epoch} epochs. Reverting to Epoch {best_epoch}.")
                break

    final_ckpt_path = checkpoint_dir / "final_model.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "final_metrics": val_metrics,
        "config": cfg,
        "version": "V3"
    }, final_ckpt_path)

    with open(checkpoint_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.2f} minutes. Best Epoch: {best_epoch} (Val AUC: {best_val_auc:.4f})")

    print("\n[4/5] Evaluating Best V3 Checkpoint on Development Test Set...")
    test_metrics = evaluate_v3_test_set(cfg, checkpoint_dir / "best_model.pt", best_epoch)

    print("\n[5/5] Evaluating on Sacred Unseen Generator Holdout Set (Wukong)...")
    holdout_metrics = evaluate_v3_holdout_set(cfg, checkpoint_dir / "best_model.pt", best_epoch)

    return test_metrics, holdout_metrics

def evaluate_v3_test_set(cfg, best_model_path, best_epoch):
    device = get_device(cfg.get("training", {}).get("device", "auto"))
    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])

    print(f"Loading checkpoint: {best_model_path}")
    model = build_model(cfg).to(device)
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, _, test_loader, _, _, test_ds = get_dataloaders(cfg)
    print(f"Evaluating on development test set ({len(test_ds):,} images)...")

    t_start = time.perf_counter()
    all_targets = []
    all_preds = []
    all_probs = []
    all_gens = []

    with torch.no_grad():
        for images, labels, gens in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)

            all_targets.extend(labels.numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_gens.extend(gens)

    total_test_time = time.perf_counter() - t_start
    avg_inference_time_ms = (total_test_time / len(test_ds)) * 1000.0

    overall_metrics = calculate_metrics(all_targets, all_preds, all_probs)

    # Compute FPR and FNR
    tn = overall_metrics["confusion_matrix"]["tn"]
    fp = overall_metrics["confusion_matrix"]["fp"]
    fn = overall_metrics["confusion_matrix"]["fn"]
    tp = overall_metrics["confusion_matrix"]["tp"]
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    overall_metrics["fpr"] = round(fpr, 4)
    overall_metrics["fnr"] = round(fnr, 4)

    gen_groups = defaultdict(lambda: {"targets": [], "preds": [], "probs": []})
    for t, p, pr, g in zip(all_targets, all_preds, all_probs, all_gens):
        gen_groups[g]["targets"].append(t)
        gen_groups[g]["preds"].append(p)
        gen_groups[g]["probs"].append(pr)

    gen_metrics = {}
    for g, data in sorted(gen_groups.items()):
        targets_arr = np.array(data["targets"])
        preds_arr = np.array(data["preds"])
        acc = float(np.mean(targets_arr == preds_arr))
        gen_metrics[g] = {
            "count": len(data["targets"]),
            "accuracy": round(acc, 4),
            "class": "REAL" if targets_arr[0] == 0 else "SYNTHETIC"
        }

    # Source grouping
    cifake_targets = [t for t, g in zip(all_targets, all_gens) if "cifar" in g or "sd14" in g or "v1_4" in g]
    cifake_preds = [p for p, g in zip(all_preds, all_gens) if "cifar" in g or "sd14" in g or "v1_4" in g]

    genimage_targets = [t for t, g in zip(all_targets, all_gens) if g in ["adm", "stable_diffusion_v1_5", "real_imagenet"]]
    genimage_preds = [p for p, g in zip(all_preds, all_gens) if g in ["adm", "stable_diffusion_v1_5", "real_imagenet"]]

    curated_targets = [t for t, g in zip(all_targets, all_gens) if g in ["real_night_sky", "synth_cosmic_fantasy"]]
    curated_preds = [p for p, g in zip(all_preds, all_gens) if g in ["real_night_sky", "synth_cosmic_fantasy"]]

    cifake_acc = float(np.mean(np.array(cifake_targets) == np.array(cifake_preds))) if cifake_targets else 0.0
    genimage_acc = float(np.mean(np.array(genimage_targets) == np.array(genimage_preds))) if genimage_targets else 0.0
    curated_acc = float(np.mean(np.array(curated_targets) == np.array(curated_preds))) if curated_targets else 0.0

    source_metrics = {
        "CIFAKE": {"count": len(cifake_targets), "accuracy": round(cifake_acc, 4)},
        "GenImage": {"count": len(genimage_targets), "accuracy": round(genimage_acc, 4)},
        "Curated_Anti_Shortcut": {"count": len(curated_targets), "accuracy": round(curated_acc, 4)}
    }

    test_results = {
        "version": "V3",
        "dataset": "SignalScope V3 Development Test Set",
        "total_test_samples": len(test_ds),
        "best_epoch": best_epoch,
        "inference_time_ms_per_sample": round(avg_inference_time_ms, 2),
        "overall_metrics": overall_metrics,
        "generator_breakdown": gen_metrics,
        "source_breakdown": source_metrics
    }

    out_json = checkpoint_dir / "test_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)
    print(f"Saved test metrics to {out_json}")

    print("\n" + "=" * 65)
    print("           V3 DEVELOPMENT TEST SET RESULTS")
    print("=" * 65)
    print(f"Accuracy:        {overall_metrics['accuracy']*100:.2f}%")
    print(f"ROC-AUC:         {overall_metrics['roc_auc']:.4f}")
    print(f"Macro F1-Score:  {overall_metrics['macro_f1']:.4f}")
    print(f"Macro Precision: {overall_metrics['macro_precision']:.4f}")
    print(f"Macro Recall:    {overall_metrics['macro_recall']:.4f}")
    print(f"Synthetic F1:    {overall_metrics['synthetic_f1']:.4f}")
    print(f"Real F1:         {overall_metrics['real_f1']:.4f}")
    print(f"False Positive Rate (FPR): {fpr*100:.2f}%")
    print(f"False Negative Rate (FNR): {fnr*100:.2f}%")
    print(f"Inference Latency: {avg_inference_time_ms:.1f} ms / sample")
    print("\nConfusion Matrix:")
    print(format_confusion_matrix(overall_metrics["confusion_matrix"]))

    print("\nPer-Generator Breakdown:")
    for g, d in gen_metrics.items():
        print(f"  - {g:24s} ({d['class']:9s}): {d['count']:4d} images | Acc: {d['accuracy']*100:.2f}%")

    print("\nPer-Source Benchmark:")
    for s, d in source_metrics.items():
        print(f"  - {s:22s}: {d['count']:4d} images | Acc: {d['accuracy']*100:.2f}%")
    print("=" * 65)

    return test_results

def evaluate_v3_holdout_set(cfg, best_model_path, best_epoch):
    device = get_device(cfg.get("training", {}).get("device", "auto"))
    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    data_cfg = cfg.get("data", {})
    holdout_dir = Path(data_cfg.get("holdout_dir", "data/v3/generator_holdout_test"))

    _, val_transform = get_transforms(
        image_size=data_cfg.get("image_size", 224),
        letterbox=data_cfg.get("letterbox", True)
    )

    holdout_ds = SignalScopeDataset(holdout_dir, transform=val_transform)
    holdout_loader = DataLoader(
        holdout_ds,
        batch_size=data_cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 2)
    )

    print(f"Evaluating Sacred Unseen Generator Holdout Set ({len(holdout_ds):,} images: {holdout_dir})...")

    model = build_model(cfg).to(device)
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_targets = []
    all_preds = []
    all_probs = []
    all_gens = []

    with torch.no_grad():
        for images, labels, gens in holdout_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)

            all_targets.extend(labels.numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_gens.extend(gens)

    holdout_metrics = calculate_metrics(all_targets, all_preds, all_probs)
    
    # Specific Wukong synthetic accuracy
    wukong_targets = [t for t, g in zip(all_targets, all_gens) if g == "wukong"]
    wukong_preds = [p for p, g in zip(all_preds, all_gens) if g == "wukong"]
    wukong_acc = float(np.mean(np.array(wukong_targets) == np.array(wukong_preds))) if wukong_targets else 0.0

    real_targets = [t for t in all_targets if t == 0]
    real_preds = [p for p, t in zip(all_preds, all_targets) if t == 0]
    real_acc = float(np.mean(np.array(real_targets) == np.array(real_preds))) if real_targets else 0.0

    holdout_results = {
        "version": "V3",
        "dataset": "SignalScope V3 Sacred Unseen Generator Holdout Set",
        "held_out_generator": "Wukong",
        "total_holdout_samples": len(holdout_ds),
        "real_samples": len(real_targets),
        "unseen_synthetic_samples": len(wukong_targets),
        "overall_metrics": holdout_metrics,
        "wukong_detection_accuracy": round(wukong_acc, 4),
        "real_accuracy": round(real_acc, 4)
    }

    out_json = checkpoint_dir / "holdout_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(holdout_results, f, indent=2)
    print(f"Saved holdout metrics to {out_json}")

    print("\n" + "=" * 65)
    print("      V3 SACRED UNSEEN-GENERATOR (WUKONG) TEST RESULTS")
    print("=" * 65)
    print(f"Holdout Samples:           {len(holdout_ds):,} (958 Real, 958 Wukong)")
    print(f"Holdout Accuracy:          {holdout_metrics['accuracy']*100:.2f}%")
    print(f"Holdout ROC-AUC:           {holdout_metrics['roc_auc']:.4f}")
    print(f"Holdout Macro F1-Score:    {holdout_metrics['macro_f1']:.4f}")
    print(f"Wukong Detection Accuracy: {wukong_acc*100:.2f}%")
    print(f"Real Photograph Accuracy:  {real_acc*100:.2f}%")
    print("\nConfusion Matrix:")
    print(format_confusion_matrix(holdout_metrics["confusion_matrix"]))
    print("=" * 65)

    return holdout_results

if __name__ == "__main__":
    train_v3()
