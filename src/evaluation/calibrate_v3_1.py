"""
SignalScope V3.1: Post-Hoc Temperature Scaling Calibration
Optimizes temperature parameter T on validation set logits to minimize Expected Calibration Error (ECE)
and negative log-likelihood (NLL).
Defines calibrated confidence bands for deployment.
"""

import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(r"C:\SignalScope")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model
from src.training.dataset import SignalScopeDataset, get_transforms

def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(probs)
    bin_details = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (probs >= bin_lower) & (probs < bin_upper) if i < n_bins - 1 else (probs >= bin_lower) & (probs <= bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_acc = np.mean(labels[in_bin] == 1)
            bin_conf = np.mean(probs[in_bin])
            bin_error = abs(bin_acc - bin_conf)
            ece += (bin_count / total) * bin_error
            bin_details.append({
                "bin": f"{bin_lower:.1f}-{bin_upper:.1f}",
                "count": int(bin_count),
                "accuracy": round(float(bin_acc), 4),
                "confidence": round(float(bin_conf), 4),
                "error": round(float(bin_error), 4)
            })

    return float(ece), bin_details

def optimize_temperature(logits_tensor, labels_tensor):
    temp = nn.Parameter(torch.ones(1) * 1.0)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temp], lr=0.01, max_iter=50)

    def eval_loss():
        optimizer.zero_grad()
        scaled = logits_tensor / temp
        loss = criterion(scaled, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    return float(temp.item())

def run_calibration(
    cfg_path="config/v3_1_train_config.json",
    model_path="models/v3_1_fix/best_model.pt",
    output_dir="reports/v3_1_fix"
):
    print("=" * 70)
    print("SIGNALSCOPE V3.1 TEMPERATURE SCALING CALIBRATION")
    print("=" * 70)

    with open(PROJECT_ROOT / cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model & load checkpoint
    model = build_model(cfg).to(device)
    ckpt = torch.load(PROJECT_ROOT / model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from: {model_path}")

    # Validation DataLoader
    val_transform = get_transforms(image_size=cfg["data"]["image_size"], letterbox=cfg["data"]["letterbox"], is_train=False)
    val_ds = SignalScopeDataset(PROJECT_ROOT / cfg["data"]["val_dir"], transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=2)
    print(f"Validation dataset: {len(val_ds)} samples")

    # Collect logits and labels
    all_logits = []
    all_labels = []

    print("Extracting validation set logits...")
    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    # 1. Uncalibrated Metrics
    uncal_probs = torch.softmax(logits_tensor, dim=1)[:, 1].numpy()
    uncal_nll = nn.CrossEntropyLoss()(logits_tensor, labels_tensor).item()
    uncal_ece, uncal_bins = compute_ece(uncal_probs, labels_tensor.numpy())

    print(f"\n--- Uncalibrated (T = 1.0000) ---")
    print(f"  Validation NLL: {uncal_nll:.4f}")
    print(f"  Validation ECE: {uncal_ece*100:.2f}%")

    # 2. Optimize Temperature T
    print("\nOptimizing Temperature T via L-BFGS...")
    optimal_T = optimize_temperature(logits_tensor, labels_tensor)
    print(f"  Optimal Temperature T: {optimal_T:.4f}")

    # 3. Calibrated Metrics
    cal_logits = logits_tensor / optimal_T
    cal_probs = torch.softmax(cal_logits, dim=1)[:, 1].numpy()
    cal_nll = nn.CrossEntropyLoss()(cal_logits, labels_tensor).item()
    cal_ece, cal_bins = compute_ece(cal_probs, labels_tensor.numpy())

    print(f"\n--- Calibrated (T = {optimal_T:.4f}) ---")
    print(f"  Validation NLL: {cal_nll:.4f} (Reduction: {uncal_nll - cal_nll:.4f})")
    print(f"  Validation ECE: {cal_ece*100:.2f}% (Reduction: {(uncal_ece - cal_ece)*100:.2f}%)")

    # Calibration result structure
    result = {
        "model_path": str(model_path),
        "temperature": round(optimal_T, 4),
        "optimal_temperature": round(optimal_T, 4),
        "uncalibrated_nll": round(uncal_nll, 4),
        "calibrated_nll": round(cal_nll, 4),
        "uncalibrated_ece": round(uncal_ece, 4),
        "calibrated_ece": round(cal_ece, 4),
        "ece_reduction_percent": round((uncal_ece - cal_ece) / max(1e-6, uncal_ece) * 100, 2),
        "confidence_bands": {
            "high_confidence_synthetic": [0.85, 1.00],
            "moderate_synthetic": [0.60, 0.85],
            "uncertain_ambiguous": [0.40, 0.60],
            "moderate_real": [0.15, 0.40],
            "high_confidence_real": [0.00, 0.15]
        },
        "bin_details": cal_bins
    }

    # Save to models/v3_1_fix/ and reports/v3_1_fix/
    out_dir_path = PROJECT_ROOT / output_dir
    out_dir_path.mkdir(parents=True, exist_ok=True)
    model_dir = PROJECT_ROOT / "models" / "v3_1_fix"
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "temperature_calibration.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    with open(out_dir_path / "temperature_calibration.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Generate Markdown Report
    md_content = f"""# SignalScope V3.1 Temperature Scaling Calibration Report

## 1. Overview
- **Model Checkpoint**: `{model_path}`
- **Calibration Split**: `data/v3_1/splits/val.csv` ({len(val_ds):,} validation samples)
- **Method**: Scalar Temperature Scaling via L-BFGS optimization on cross-entropy loss.
- **Optimal Temperature ($T$)**: `{optimal_T:.4f}`

---

## 2. Quantitative Calibration Results

| Metric | Uncalibrated ($T=1.0000$) | Calibrated ($T={optimal_T:.4f}$) | Improvement |
| :--- | :--- | :--- | :--- |
| **Negative Log-Likelihood (NLL)** | {uncal_nll:.4f} | **{cal_nll:.4f}** | {uncal_nll - cal_nll:.4f} lower |
| **Expected Calibration Error (ECE)** | {uncal_ece*100:.2f}% | **{cal_ece*100:.2f}%** | **{((uncal_ece - cal_ece)/max(1e-6, uncal_ece))*100:.1f}% reduction** |

---

## 3. Operational Confidence Bands
Calibrated probabilities are mapped to the following operational decision bands:

- **$\ge 0.85$**: High-Confidence Synthetic (Autonomous flagging / watermark detection)
- **$0.60 - 0.85$**: Moderate Synthetic (Review recommended)
- **$0.40 - 0.60$**: Ambiguous / Uncertain (Borderline probability, human verification required)
- **$0.15 - 0.40$**: Moderate Real (Likely natural photography)
- **$< 0.15$**: High-Confidence Real (Unambiguous camera capture)

---

## 4. Reliability Diagram Bins

| Bin Interval | Count | Mean Confidence | Observed Accuracy | Calibration Error |
| :--- | :--- | :--- | :--- | :--- |
"""
    for b in cal_bins:
        md_content += f"| {b['bin']} | {b['count']} | {b['confidence']*100:.1f}% | {b['accuracy']*100:.1f}% | {b['error']*100:.2f}% |\n"

    with open(out_dir_path / "V3_1_CALIBRATION_REPORT.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nCalibration report saved to: {out_dir_path / 'V3_1_CALIBRATION_REPORT.md'}")
    return result

if __name__ == "__main__":
    run_calibration()
