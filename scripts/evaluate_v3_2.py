"""
SignalScope V3.2 Evaluation & Diagnostic Verification Suite
Evaluates trained checkpoint models/v3_2_fix/best_model.pt across:
  1. Development Test Set (N=3,165)
  2. Sacred Unseen Generator Holdout Set (N=1,916, Wukong)
  3. Diagnostic failure images (Foggy Road, Three People Indoor, Instagram Screenshot, Alpine Cabin)
  4. Generates V3 vs V3.2 comparison report
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
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model
from src.training.dataset import (
    get_dataloaders,
    SignalScopeDataset,
    get_transforms
)
from src.evaluation.metrics import calculate_metrics, format_confusion_matrix

def main():
    cfg_path = PROJECT_ROOT / "config" / "v3_2_train_config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_path = PROJECT_ROOT / "models" / "v3_2_fix" / "best_model.pt"
    reports_dir = PROJECT_ROOT / "reports" / "v3_2_fix"
    models_dir = PROJECT_ROOT / "models" / "v3_2_fix"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading V3.2 model from {model_path}...")
    model = build_model(cfg).to(device)
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # Load temperature
    temp_path = models_dir / "temperature_calibration.json"
    temperature = 1.0
    if temp_path.exists():
        with open(temp_path, "r") as f:
            cal = json.load(f)
            temperature = cal.get("optimal_temperature", 1.0)
    print(f"Applied Temperature: T={temperature:.4f}")

    # 1. Evaluate Development Test Set
    print("\n[1/3] Evaluating Development Test Set (N=3,165)...")
    _, _, test_loader, _, _, test_ds = get_dataloaders(cfg)
    all_targets = []
    all_preds = []
    all_probs = []
    all_gens = []

    t0 = time.time()
    with torch.no_grad():
        for images, labels, gens in test_loader:
            images = images.to(device)
            outputs = model(images)
            scaled = outputs / temperature
            probs = torch.softmax(scaled, dim=1)[:, 1]
            _, preds = torch.max(scaled, 1)

            all_targets.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_gens.extend(gens)

    test_time = time.time() - t0
    test_metrics = calculate_metrics(all_targets, all_preds, all_probs)
    tn = test_metrics["confusion_matrix"]["tn"]
    fp = test_metrics["confusion_matrix"]["fp"]
    fn = test_metrics["confusion_matrix"]["fn"]
    tp = test_metrics["confusion_matrix"]["tp"]
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    test_metrics["fpr"] = round(fpr, 4)
    test_metrics["fnr"] = round(fnr, 4)

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

    test_summary = {
        "version": "V3.2",
        "total_test_samples": len(test_ds),
        "overall_metrics": test_metrics,
        "generator_breakdown": gen_metrics,
        "evaluation_time_sec": round(test_time, 2)
    }
    with open(models_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_summary, f, indent=2)
    with open(reports_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_summary, f, indent=2)

    print(f"  Test Accuracy:    {test_metrics['accuracy']*100:.2f}%")
    print(f"  Test ROC-AUC:     {test_metrics['roc_auc']:.4f}")
    print(f"  Test Macro-F1:    {test_metrics['macro_f1']:.4f}")
    print(f"  Test FPR:         {fpr*100:.2f}% ({fp}/{fp+tn})")
    print(f"  Test FNR:         {fnr*100:.2f}% ({fn}/{fn+tp})")

    # 2. Evaluate Holdout Set (Wukong)
    print("\n[2/3] Evaluating Sacred Unseen Holdout Set (N=1,916, Wukong)...")
    holdout_dir = PROJECT_ROOT / "data" / "v3" / "generator_holdout_test"
    val_transform = get_transforms(
        image_size=224,
        letterbox=True,
        is_train=False
    )
    holdout_ds = SignalScopeDataset(holdout_dir, transform=val_transform)
    holdout_loader = DataLoader(
        holdout_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    h_targets = []
    h_preds = []
    h_probs = []
    h_gens = []

    with torch.no_grad():
        for images, labels, gens in holdout_loader:
            images = images.to(device)
            outputs = model(images)
            scaled = outputs / temperature
            probs = torch.softmax(scaled, dim=1)[:, 1]
            _, preds = torch.max(scaled, 1)

            h_targets.extend(labels.cpu().numpy().tolist())
            h_preds.extend(preds.cpu().numpy().tolist())
            h_probs.extend(probs.cpu().numpy().tolist())
            h_gens.extend(gens)

    holdout_metrics = calculate_metrics(h_targets, h_preds, h_probs)
    wukong_targets = [t for t, g in zip(h_targets, h_gens) if g == "wukong"]
    wukong_preds = [p for p, g in zip(h_preds, h_gens) if g == "wukong"]
    wukong_acc = float(np.mean(np.array(wukong_targets) == np.array(wukong_preds))) if wukong_targets else 0.0

    real_targets = [t for t in h_targets if t == 0]
    real_preds = [p for p, t in zip(h_preds, h_targets) if t == 0]
    real_acc = float(np.mean(np.array(real_targets) == np.array(real_preds))) if real_targets else 0.0

    holdout_summary = {
        "version": "V3.2",
        "held_out_generator": "Wukong",
        "total_holdout_samples": len(holdout_ds),
        "wukong_unseen_acc": round(wukong_acc, 4),
        "real_acc": round(real_acc, 4),
        "overall_acc": holdout_metrics["accuracy"],
        "roc_auc": holdout_metrics["roc_auc"],
        "metrics": holdout_metrics
    }
    with open(models_dir / "holdout_metrics.json", "w", encoding="utf-8") as f:
        json.dump(holdout_summary, f, indent=2)
    with open(reports_dir / "holdout_metrics.json", "w", encoding="utf-8") as f:
        json.dump(holdout_summary, f, indent=2)

    print(f"  Holdout Overall Accuracy: {holdout_metrics['accuracy']*100:.2f}%")
    print(f"  Holdout ROC-AUC:          {holdout_metrics['roc_auc']:.4f}")
    print(f"  Unseen Wukong Accuracy:   {wukong_acc*100:.2f}% ({sum(wukong_preds)}/{len(wukong_preds)})")
    print(f"  Holdout Real Accuracy:    {real_acc*100:.2f}%")

    # 3. Evaluate Diagnostic Images
    print("\n[3/3] Evaluating Diagnostic Failure Cases under V3 vs V3.2...")
    diag_images = [
        ("image_1_foggy_road", PROJECT_ROOT / "data" / "diagnostic_failures" / "image_1_foggy_road.jpeg"),
        ("image_2_three_people_indoor", PROJECT_ROOT / "data" / "diagnostic_failures" / "image_2_three_people_indoor.jpeg"),
        ("image_3_instagram_screenshot", PROJECT_ROOT / "data" / "diagnostic_failures" / "image_3_instagram_screenshot.jpeg"),
        ("image_4_alpine_cabin", PROJECT_ROOT / "data" / "diagnostic_failures" / "image_4_alpine_cabin.jpeg")
    ]

    # Load V3 model for exact side-by-side comparison
    v3_path = PROJECT_ROOT / "models" / "v3_final_candidate" / "best_model.pt"
    v3_model = build_model(cfg).to(device)
    v3_ckpt = torch.load(v3_path, map_location=device)
    v3_model.load_state_dict(v3_ckpt["model_state_dict"])
    v3_model.eval()

    diag_comparison = {}
    print(f"{'Image Name':32s} | {'V3 P(Synth)':12s} | {'V3 Label':12s} | {'V3.2 P(Synth)':14s} | {'V3.2 Label':12s} | Status")
    print("-" * 100)

    for name, p in diag_images:
        if not p.exists():
            continue
        with Image.open(p) as img:
            img_rgb = img.convert("RGB")
        tensor = val_transform(img_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            # V3 prediction (T=1.0195)
            v3_logits = v3_model(tensor)
            v3_probs = torch.softmax(v3_logits / 1.0195, dim=1)[0]
            v3_synth = float(v3_probs[1].item())
            v3_label = "AI-GENERATED" if v3_synth > 0.50 else "REAL"

            # V3.2 prediction (T=temperature)
            v32_logits = model(tensor)
            v32_probs = torch.softmax(v32_logits / temperature, dim=1)[0]
            v32_synth = float(v32_probs[1].item())
            v32_label = "AI-GENERATED" if v32_synth > 0.50 else "REAL"

        status = "FIXED (AI DETECTED)" if v32_label == "AI-GENERATED" else "STILL REAL"
        print(f"{name:32s} | {v3_synth*100:10.2f}% | {v3_label:12s} | {v32_synth*100:12.2f}% | {v32_label:12s} | {status}")

        diag_comparison[name] = {
            "v3": {"p_real": round(1.0 - v3_synth, 4), "p_synthetic": round(v3_synth, 4), "label": v3_label},
            "v3_2": {"p_real": round(1.0 - v32_synth, 4), "p_synthetic": round(v32_synth, 4), "label": v32_label},
            "status": status
        }

    with open(reports_dir / "diagnostic_comparison_v3_vs_v3_2.json", "w", encoding="utf-8") as f:
        json.dump(diag_comparison, f, indent=2)

    print("\n" + "=" * 75)
    print("   V3.2 EVALUATION COMPLETE")
    print("=" * 75)

if __name__ == "__main__":
    main()
