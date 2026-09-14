"""
SignalScope V2: Training & Evaluation Pipeline
Trains EfficientNet-B0 on data/v2/splits/ (21,743 images total, 15,220 train)
Saves best checkpoint to models/v2_expanded_data/best_model.pt
Evaluates on development test set (3,262 images) with per-generator forensics.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model
from src.training.dataset import get_dataloaders, compute_class_weights
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

        if batch_idx % 100 == 0 or batch_idx == num_batches:
            batch_time = time.time() - t0
            cur_loss = running_loss / total
            cur_acc = correct / total
            print(f"  [Epoch {epoch}/{max_epochs} | Batch {batch_idx:3d}/{num_batches}] Loss: {cur_loss:.4f} | Acc: {cur_acc*100:.2f}% | Elapsed: {batch_time:.1f}s")

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

def train_v2(cfg_path="config/v2_train_config.json"):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    print("=" * 70)
    print("   SIGNALSCOPE V2: EFFICIENTNET-B0 EXPANDED TRAINING PIPELINE")
    print("=" * 70)

    training_cfg = cfg.get("training", {})
    seed = training_cfg.get("seed", 42)
    set_seed(seed)

    device = get_device(training_cfg.get("device", "auto"))
    print(f"Execution Device: {device}")

    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "models/v2_expanded_data"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with open(checkpoint_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print("\n[1/4] Loading and preparing V2 expanded datasets...")
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = get_dataloaders(cfg)
    print(f"  Training samples:   {len(train_ds):,} images")
    print(f"  Validation samples: {len(val_ds):,} images")
    print(f"  Test samples:       {len(test_ds):,} images")

    class_weights, real_cnt, synth_cnt = compute_class_weights(train_ds, device=device)
    print(f"  Train class balance: {real_cnt:,} Real vs. {synth_cnt:,} Synthetic")
    print(f"  Class Weights: Real={class_weights[0]:.4f}, Synthetic={class_weights[1]:.4f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("\n[2/4] Building EfficientNet-B0 transfer learning architecture...")
    model = build_model(cfg).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_cnt = sum(p.numel() for p in trainable_params)
    print(f"  Trainable parameters: {trainable_cnt:,} / {total_params:,} ({trainable_cnt/total_params*100:.1f}%)")

    lr = training_cfg.get("learning_rate", 1e-3)
    wd = training_cfg.get("weight_decay", 1e-2)
    max_epochs = training_cfg.get("max_epochs", 3)
    patience = training_cfg.get("early_stopping_patience", 2)

    optimizer = AdamW(trainable_params, lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)

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

    print(f"\n[3/4] Starting training for up to {max_epochs} epochs (Patience={patience})...")
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"\n--- Epoch {epoch}/{max_epochs} (lr={cur_lr:.6f}) ---")

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
                "version": "V2"
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
        "version": "V2"
    }, final_ckpt_path)

    with open(checkpoint_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.2f} minutes. Best Epoch: {best_epoch} (Val AUC: {best_val_auc:.4f})")

    print("\n[4/4] Evaluating Best V2 Checkpoint on Development Test Set...")
    evaluate_v2_test_set(cfg, checkpoint_dir / "best_model.pt", best_epoch)

def evaluate_v2_test_set(cfg, best_model_path, best_epoch):
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

    cifake_targets = [t for t, g in zip(all_targets, all_gens) if "cifar" in g or "sd14" in g or "v1_4" in g]
    cifake_preds = [p for p, g in zip(all_preds, all_gens) if "cifar" in g or "sd14" in g or "v1_4" in g]
    genimage_targets = [t for t, g in zip(all_targets, all_gens) if not ("cifar" in g or "sd14" in g or "v1_4" in g)]
    genimage_preds = [p for p, g in zip(all_preds, all_gens) if not ("cifar" in g or "sd14" in g or "v1_4" in g)]

    cifake_acc = float(np.mean(np.array(cifake_targets) == np.array(cifake_preds))) if cifake_targets else 0.0
    genimage_acc = float(np.mean(np.array(genimage_targets) == np.array(genimage_preds))) if genimage_targets else 0.0

    source_metrics = {
        "CIFAKE": {"count": len(cifake_targets), "accuracy": round(cifake_acc, 4)},
        "GenImage": {"count": len(genimage_targets), "accuracy": round(genimage_acc, 4)}
    }

    test_results = {
        "version": "V2",
        "dataset": "SignalScope V2 Expanded Development Test Set",
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

    print("\n" + "=" * 60)
    print("           V2 DEVELOPMENT TEST SET RESULTS")
    print("=" * 60)
    print(f"Accuracy:        {overall_metrics['accuracy']*100:.2f}%")
    print(f"ROC-AUC:         {overall_metrics['roc_auc']:.4f}")
    print(f"Macro F1-Score:  {overall_metrics['macro_f1']:.4f}")
    print(f"Macro Precision: {overall_metrics['macro_precision']:.4f}")
    print(f"Macro Recall:    {overall_metrics['macro_recall']:.4f}")
    print(f"Synthetic F1:    {overall_metrics['synthetic_f1']:.4f}")
    print(f"Real F1:         {overall_metrics['real_f1']:.4f}")
    print(f"Inference Latency: {avg_inference_time_ms:.1f} ms / sample")
    print("\nConfusion Matrix:")
    print(format_confusion_matrix(overall_metrics["confusion_matrix"]))

    print("\nPer-Generator Breakdown:")
    for g, d in gen_metrics.items():
        print(f"  - {g:22s} ({d['class']:9s}): {d['count']:4d} images | Acc: {d['accuracy']*100:.2f}%")

    print("\nPer-Source Benchmark:")
    for s, d in source_metrics.items():
        print(f"  - {s:10s}: {d['count']:4d} images | Acc: {d['accuracy']*100:.2f}%")
    print("=" * 60)

    report_md = f"""# SignalScope V2 Model Evaluation Report

* **Model Version:** V2 (Expanded Dataset)
* **Architecture:** EfficientNet-B0 (Transfer Learning)
* **Pretrained Weights:** ImageNet-1K
* **Checkpoint:** `{best_model_path}`
* **Best Validation Epoch:** {best_epoch}
* **Test Dataset:** SignalScope V2 Development Test Set ({len(test_ds):,} images)
* **Official SIH Held-Out Test Set Used:** **NO** (Preserved untouched)

---

## 1. Overall Performance Metrics

| Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **Accuracy** | **{overall_metrics['accuracy']*100:.2f}%** | Overall binary prediction accuracy |
| **ROC-AUC** | **{overall_metrics['roc_auc']:.4f}** | Area under receiver operating characteristic curve |
| **Macro F1-Score** | **{overall_metrics['macro_f1']:.4f}** | Harmonic mean of precision and recall |
| **Macro Precision** | **{overall_metrics['macro_precision']:.4f}** | Unweighted average precision across both classes |
| **Macro Recall** | **{overall_metrics['macro_recall']:.4f}** | Unweighted average recall across both classes |
| **Synthetic F1** | **{overall_metrics['synthetic_f1']:.4f}** | Performance specifically on AI-generated images |
| **Real F1** | **{overall_metrics['real_f1']:.4f}** | Performance specifically on Real photographs |
| **Inference Latency** | **{avg_inference_time_ms:.1f} ms** | Average per-sample inference time on CPU |

---

## 2. Confusion Matrix

```
{format_confusion_matrix(overall_metrics['confusion_matrix'])}
```

* **True Negatives (Real correctly identified):** {overall_metrics['confusion_matrix']['tn']}
* **False Positives (Real misclassified as AI):** {overall_metrics['confusion_matrix']['fp']}
* **False Negatives (AI misclassified as Real):** {overall_metrics['confusion_matrix']['fn']}
* **True Positives (AI correctly identified):** {overall_metrics['confusion_matrix']['tp']}

---

## 3. Generative Architecture Breakdown

| Generator / Source | Category | Class | Test Samples | Accuracy |
| :--- | :--- | :---: | :---: | :---: |
"""
    for g, d in sorted(gen_metrics.items()):
        report_md += f"| `{g}` | {d['class']} | {d['class']} | {d['count']} | **{d['accuracy']*100:.2f}%** |\n"

    report_md += f"""
---

## 4. Benchmark by Source Dataset

| Dataset Source | Samples | Accuracy |
| :--- | :---: | :---: |
| **CIFAKE** (CIFAR-10 Camera + SD v1.4) | {source_metrics['CIFAKE']['count']} | **{source_metrics['CIFAKE']['accuracy']*100:.2f}%** |
| **GenImage** (ImageNet-1K Camera + ADM + SD v1.5 + Wukong) | {source_metrics['GenImage']['count']} | **{source_metrics['GenImage']['accuracy']*100:.2f}%** |
"""

    out_md = checkpoint_dir / "test_evaluation_report.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Saved evaluation markdown to {out_md}")

    return test_results

if __name__ == "__main__":
    train_v2()
