"""
SignalScope V3: Post-Hoc Temperature Scaling Calibration
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model
from src.training.dataset import get_dataloaders

def compute_ece(probs, labels, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) across n_bins.
    probs: numpy array of predicted confidence for positive class (or top predicted class)
    labels: numpy array of true binary labels (0 or 1)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(probs)
    bin_details = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Samples in this confidence interval
        in_bin = (probs >= bin_lower) & (probs < bin_upper) if i < n_bins - 1 else (probs >= bin_lower) & (probs <= bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_acc = np.mean(labels[in_bin] == 1)  # accuracy for synthetic class
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
    """
    Learns scalar temperature T via L-BFGS to minimize NLL on validation logits.
    """
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
    cfg_path="config/v3_train_config.json",
    model_path="models/v3_final_candidate/best_model.pt",
    output_dir="reports/v3_final_candidate"
):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading V3 model for calibration from {model_path}...")
    model = build_model(cfg).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, val_loader, _, _, val_ds, _ = get_dataloaders(cfg)
    print(f"Extracting validation logits across {len(val_ds):,} samples...")

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for imgs, lbls, _ in val_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            all_logits.append(outputs.cpu())
            all_labels.append(lbls.cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Uncalibrated (T=1.0)
    uncal_probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    labels_np = labels.numpy()
    uncal_ece, uncal_bins = compute_ece(uncal_probs, labels_np)

    print(f"Uncalibrated (T=1.000) Validation ECE: {uncal_ece*100:.2f}%")

    # Fit temperature T
    optimal_T = optimize_temperature(logits, labels)
    print(f"Optimal Temperature Parameter T: {optimal_T:.4f}")

    # Calibrated
    cal_logits = logits / optimal_T
    cal_probs = torch.softmax(cal_logits, dim=1)[:, 1].numpy()
    cal_ece, cal_bins = compute_ece(cal_probs, labels_np)
    print(f"Calibrated (T={optimal_T:.4f}) Validation ECE: {cal_ece*100:.2f}% (Reduction: -{(uncal_ece - cal_ece)*100:.2f}%)")

    # Define confidence tiers
    confidence_bands = {
        "HIGH_CONFIDENCE": {
            "synthetic_range": [0.90, 1.00],
            "real_range": [0.00, 0.10],
            "action": "Automated verification / Flagging with high certainty"
        },
        "MODERATE_CONFIDENCE": {
            "synthetic_range": [0.70, 0.90],
            "real_range": [0.10, 0.30],
            "action": "Secondary review recommended / Check forensic explainability"
        },
        "LOW_CONFIDENCE_BORDERLINE": {
            "range": [0.30, 0.70],
            "action": "Inconclusive authenticity score / Human forensic inspection required"
        }
    }

    cal_results = {
        "version": "V3",
        "validation_samples": len(val_ds),
        "optimal_temperature": round(optimal_T, 4),
        "uncalibrated_ece": round(uncal_ece, 4),
        "calibrated_ece": round(cal_ece, 4),
        "ece_reduction": round(uncal_ece - cal_ece, 4),
        "uncalibrated_bins": uncal_bins,
        "calibrated_bins": cal_bins,
        "confidence_bands": confidence_bands
    }

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    with open(out_p / "temperature_calibration.json", "w", encoding="utf-8") as f:
        json.dump(cal_results, f, indent=2)

    # Save to models directory too
    model_dir = Path(model_path).parent
    with open(model_dir / "temperature_calibration.json", "w", encoding="utf-8") as f:
        json.dump(cal_results, f, indent=2)

    md_path = out_p / "V3_CALIBRATION_REPORT.md"
    md_content = f"""# SignalScope V3 Temperature Scaling Calibration Report

* **Model:** SignalScope V3 EfficientNet-B0
* **Validation Samples:** {len(val_ds):,}
* **Optimal Temperature (T):** **{optimal_T:.4f}**
* **Pre-Calibration ECE:** {uncal_ece*100:.2f}%
* **Post-Calibration ECE:** **{cal_ece*100:.2f}%** (ECE Improvement: **-{(uncal_ece - cal_ece)*100:.2f}%**)

---

## 1. Reliability & Calibration Bins

| Confidence Bin | Samples | Observed Synthetic Frequency | Mean Confidence | Calibration Error |
| :---: | :---: | :---: | :---: | :---: |
"""
    for b in cal_bins:
        md_content += f"| {b['bin']} | {b['count']} | {b['accuracy']*100:.1f}% | {b['confidence']*100:.1f}% | {b['error']*100:.2f}% |\n"

    md_content += f"""
---

## 2. Calibrated Confidence Bands

* **HIGH CONFIDENCE (>=90% Synthetic or <=10% Real):** Extremely reliable automated detection.
* **MODERATE CONFIDENCE (70-90% Synthetic or 10-30% Real):** Reliable identification; forensic explainability inspection advised.
* **BORDERLINE / INCONCLUSIVE (30-70%):** Model uncertainty elevated; manual forensic evaluation required.
"""

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved calibration report to {md_path}")

    return cal_results

if __name__ == "__main__":
    run_calibration()
