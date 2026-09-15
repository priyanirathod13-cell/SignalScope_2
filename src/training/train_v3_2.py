"""
SignalScope V3.2: Targeted De-biasing Training and Evaluation Pipeline
Solves the AI-photograph false-real failure mode:
  1. Loads frozen V3 baseline weights (models/v3_final_candidate/best_model.pt)
  2. Applies Symmetric Aspect-Ratio Padding Augmentation (p=0.5) to both Real and Synthetic
     training images, breaking the spurious correlation between gray margins/borders and the REAL class.
  3. Applies Social Media Compression (Q=50..92, subtle downscale+upscale) to withstand WhatsApp/Instagram recompression.
  4. Differential learning rates: Backbone blocks 6-8 at 5e-5, Classifier head at 3e-4.
  5. Evaluates on Development Test Set (N=3,165), Sacred Unseen Wukong Holdout Set (N=1,916),
     and the 4 diagnostic failure cases.
  6. Calibrates temperature and saves full reports to models/v3_2_fix/ and reports/v3_2_fix/.
"""

import os
import sys
import time
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import scipy.optimize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model
from src.training.dataset import (
    get_dataloaders,
    compute_class_weights,
    SignalScopeDataset,
    get_transforms,
    LetterboxTransform
)
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

def calibrate_temperature(model, val_loader, device):
    """Calculates optimal temperature T on validation logits."""
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            outputs = model(images)
            all_logits.append(outputs.cpu())
            all_labels.append(labels)
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    def ece_calc(probs, targets, n_bins=10):
        confidences, predictions = torch.max(probs, 1)
        accuracies = predictions.eq(targets)
        ece = torch.zeros(1)
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece.item()

    uncal_probs = torch.softmax(logits, dim=1)
    ece_pre = ece_calc(uncal_probs, labels)

    def nll(t):
        t_val = float(t[0])
        scaled = logits / max(t_val, 1e-4)
        loss = nn.CrossEntropyLoss()(scaled, labels)
        return loss.item()

    res = scipy.optimize.minimize(nll, [1.0], method='Nelder-Mead')
    opt_t = float(res.x[0])

    cal_probs = torch.softmax(logits / opt_t, dim=1)
    ece_post = ece_calc(cal_probs, labels)

    return opt_t, ece_pre, ece_post

def evaluate_test_set(cfg, model_path):
    device = get_device(cfg.get("training", {}).get("device", "auto"))
    model = build_model(cfg).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, _, test_loader, _, _, test_ds = get_dataloaders(cfg)
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

            all_targets.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_gens.extend(gens)

    overall_metrics = calculate_metrics(all_targets, all_preds, all_probs)
    tn = overall_metrics["confusion_matrix"]["tn"]
    fp = overall_metrics["confusion_matrix"]["fp"]
    fn = overall_metrics["confusion_matrix"]["fn"]
    tp = overall_metrics["confusion_matrix"]["tp"]
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    overall_metrics["fpr"] = round(fpr, 4)
    overall_metrics["fnr"] = round(fnr, 4)

    gen_groups = defaultdict(lambda: {"targets": [], "preds": []})
    for t, p, g in zip(all_targets, all_preds, all_gens):
        gen_groups[g]["targets"].append(t)
        gen_groups[g]["preds"].append(p)

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

    return overall_metrics, gen_metrics

def evaluate_holdout_set(cfg, model_path):
    device = get_device(cfg.get("training", {}).get("device", "auto"))
    data_cfg = cfg.get("data", {})
    holdout_dir = Path(data_cfg.get("holdout_dir", "data/v3/generator_holdout_test"))

    val_transform = get_transforms(
        image_size=data_cfg.get("image_size", 224),
        letterbox=data_cfg.get("letterbox", True),
        is_train=False
    )

    holdout_ds = SignalScopeDataset(holdout_dir, transform=val_transform)
    holdout_loader = DataLoader(
        holdout_ds,
        batch_size=data_cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 0)
    )

    model = build_model(cfg).to(device)
    ckpt = torch.load(model_path, map_location=device)
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

            all_targets.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_gens.extend(gens)

    holdout_metrics = calculate_metrics(all_targets, all_preds, all_probs)
    wukong_targets = [t for t, g in zip(all_targets, all_gens) if g == "wukong"]
    wukong_preds = [p for p, g in zip(all_preds, all_gens) if g == "wukong"]
    wukong_acc = float(np.mean(np.array(wukong_targets) == np.array(wukong_preds))) if wukong_targets else 0.0

    real_targets = [t for t in all_targets if t == 0]
    real_preds = [p for p, t in zip(all_preds, all_targets) if t == 0]
    real_acc = float(np.mean(np.array(real_targets) == np.array(real_preds))) if real_targets else 0.0

    return holdout_metrics, wukong_acc, real_acc

def evaluate_diagnostic_images(model, device, temperature=1.0):
    diagnostic_paths = [
        ("image_1_foggy_road", "data/diagnostic_failures/image_1_foggy_road.jpeg"),
        ("image_2_three_people_indoor", "data/diagnostic_failures/image_2_three_people_indoor.jpeg"),
        ("image_3_instagram_screenshot", "data/diagnostic_failures/image_3_instagram_screenshot.jpeg"),
        ("image_4_alpine_cabin", "data/diagnostic_failures/image_4_alpine_cabin.jpeg")
    ]
    
    transform = get_transforms(image_size=224, letterbox=True, is_train=False)
    results = {}

    model.eval()
    with torch.no_grad():
        for name, p_str in diagnostic_paths:
            p = Path(p_str)
            if not p.exists():
                results[name] = {"error": "File not found"}
                continue
            with Image.open(p) as img:
                img_rgb = img.convert("RGB")
            tensor = transform(img_rgb).unsqueeze(0).to(device)
            logits = model(tensor)
            scaled = logits / temperature
            probs = torch.softmax(scaled, dim=1)[0]
            p_real = float(probs[0].item())
            p_synth = float(probs[1].item())
            pred_class = 1 if p_synth > 0.50 else 0
            results[name] = {
                "p_real": round(p_real, 4),
                "p_synthetic": round(p_synth, 4),
                "pred_label": "AI-GENERATED" if pred_class == 1 else "REAL"
            }
    return results

def run_v3_2_pipeline(cfg_path="config/v3_2_train_config.json"):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    print("=" * 75)
    print("   SIGNALSCOPE V3.2: TARGETED ANTI-SHORTCUT FINE-TUNING & EVALUATION")
    print("=" * 75)

    training_cfg = cfg.get("training", {})
    seed = training_cfg.get("seed", 42)
    set_seed(seed)

    device = get_device(training_cfg.get("device", "auto"))
    print(f"Execution Device: {device}")

    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "models/v3_2_fix"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(training_cfg.get("reports_dir", "reports/v3_2_fix"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open(checkpoint_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print("\n[1/6] Loading datasets with Symmetric Aspect Padding & Compression...")
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = get_dataloaders(cfg)
    print(f"  Training samples:   {len(train_ds):,} images (Aspect Padding: ON)")
    print(f"  Validation samples: {len(val_ds):,} images")
    print(f"  Test samples:       {len(test_ds):,} images")

    class_weights, real_cnt, synth_cnt = compute_class_weights(train_ds, device=device)
    print(f"  Train balance: {real_cnt:,} Real vs {synth_cnt:,} Synthetic")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("\n[2/6] Initializing EfficientNet-B0 from V3 Baseline Checkpoint...")
    base_ckpt_path = Path(training_cfg.get("base_checkpoint", "models/v3_final_candidate/best_model.pt"))
    model = build_model(cfg).to(device)
    if base_ckpt_path.exists():
        ckpt = torch.load(base_ckpt_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"  Successfully loaded weights from {base_ckpt_path}")
    else:
        print(f"  WARNING: Base checkpoint {base_ckpt_path} not found! Initializing fresh.")

    # Differential LR setup
    backbone_params = [p for p in model.backbone.features.parameters() if p.requires_grad]
    head_params = [p for p in model.backbone.classifier.parameters() if p.requires_grad]
    bb_lr = training_cfg.get("backbone_lr", 5e-5)
    hd_lr = training_cfg.get("head_lr", 3e-4)
    wd = training_cfg.get("weight_decay", 0.01)
    max_epochs = training_cfg.get("max_epochs", 2)

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

    print(f"\n[3/6] Starting V3.2 fine-tuning for {max_epochs} epochs...")
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
            best_ckpt_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": cfg,
                "version": "V3.2"
            }, best_ckpt_path)
            print(f"  >>> NEW BEST MODEL saved to {best_ckpt_path} (Val ROC-AUC: {best_val_auc:.4f}) <<<")

    final_ckpt_path = checkpoint_dir / "final_model.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "final_metrics": val_metrics,
        "config": cfg,
        "version": "V3.2"
    }, final_ckpt_path)

    with open(checkpoint_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nFine-tuning completed in {(time.time() - start_time)/60:.2f} minutes. Best Epoch: {best_epoch}")

    best_ckpt = checkpoint_dir / "best_model.pt"

    print("\n[4/6] Calibrating Temperature...")
    opt_t, ece_pre, ece_post = calibrate_temperature(model, val_loader, device)
    calib_data = {
        "optimal_temperature": round(opt_t, 4),
        "ece_uncalibrated": round(ece_pre, 4),
        "ece_calibrated": round(ece_post, 4)
    }
    with open(checkpoint_dir / "temperature_calibration.json", "w", encoding="utf-8") as f:
        json.dump(calib_data, f, indent=2)
    with open(reports_dir / "temperature_calibration.json", "w", encoding="utf-8") as f:
        json.dump(calib_data, f, indent=2)
    print(f"  Optimal Temperature: T={opt_t:.4f} (ECE: {ece_pre*100:.2f}% -> {ece_post*100:.2f}%)")

    print("\n[5/6] Evaluating on Development Test Set (N=3,165) & Holdout Set (N=1,916)...")
    test_metrics, gen_metrics = evaluate_test_set(cfg, best_ckpt)
    holdout_metrics, wukong_acc, holdout_real_acc = evaluate_holdout_set(cfg, best_ckpt)

    test_summary = {
        "version": "V3.2",
        "test_samples": len(test_ds),
        "accuracy": test_metrics["accuracy"],
        "roc_auc": test_metrics["roc_auc"],
        "macro_f1": test_metrics["macro_f1"],
        "fpr": test_metrics["fpr"],
        "fnr": test_metrics["fnr"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "generator_breakdown": gen_metrics
    }
    with open(checkpoint_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_summary, f, indent=2)
    with open(reports_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_summary, f, indent=2)

    holdout_summary = {
        "version": "V3.2",
        "wukong_unseen_acc": round(wukong_acc, 4),
        "real_acc": round(holdout_real_acc, 4),
        "overall_acc": holdout_metrics["accuracy"],
        "roc_auc": holdout_metrics["roc_auc"]
    }
    with open(checkpoint_dir / "holdout_metrics.json", "w", encoding="utf-8") as f:
        json.dump(holdout_summary, f, indent=2)
    with open(reports_dir / "holdout_metrics.json", "w", encoding="utf-8") as f:
        json.dump(holdout_summary, f, indent=2)

    print(f"  Test Accuracy:    {test_metrics['accuracy']*100:.2f}%")
    print(f"  Test ROC-AUC:     {test_metrics['roc_auc']:.4f}")
    print(f"  Test Macro-F1:    {test_metrics['macro_f1']:.4f}")
    print(f"  Test FPR:         {test_metrics['fpr']*100:.2f}%")
    print(f"  Test FNR:         {test_metrics['fnr']*100:.2f}%")
    print(f"  Wukong Holdout:   {wukong_acc*100:.2f}%")

    print("\n[6/6] Testing Diagnostic Failure Cases with V3.2...")
    diag_results = evaluate_diagnostic_images(model, device, temperature=opt_t)
    for name, r in diag_results.items():
        print(f"  {name:30s} => {r['pred_label']:12s} (Real: {r['p_real']*100:.2f}%, Synth: {r['p_synthetic']*100:.2f}%)")

    with open(reports_dir / "diagnostic_eval_v3_2.json", "w", encoding="utf-8") as f:
        json.dump(diag_results, f, indent=2)

    print("\n" + "=" * 75)
    print("   V3.2 PIPELINE COMPLETE")
    print("=" * 75)

if __name__ == "__main__":
    run_v3_2_pipeline()
