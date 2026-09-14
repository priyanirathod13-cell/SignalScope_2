import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json, csv
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from src.models import build_model
from src.training.dataset import LetterboxTransform

# Load model configuration & weights
with open('config/v3_train_config.json') as f:
    cfg = json.load(f)
model = build_model(cfg)
ckpt = torch.load('models/v3_final_candidate/best_model.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
model.eval()

# Calibrated temperature
calib_file = Path('models/v3_final_candidate/temperature_calibration.json')
temp = 1.0195
if calib_file.exists():
    with open(calib_file) as cf:
        temp = json.load(cf).get('optimal_temperature', 1.0195)

norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def evaluate_split_probabilities(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    
    y_true = []
    p_synth_list = []
    generators = []
    
    for r in rows:
        img_p = Path(r['image_path'])
        if not img_p.exists():
            continue
        try:
            im = Image.open(img_p).convert('RGB')
        except Exception:
            continue
        t_lb = norm(LetterboxTransform(224)(im)).unsqueeze(0)
        with torch.no_grad():
            out = model(t_lb).squeeze()
            scaled = out / temp
            probs = torch.softmax(scaled, dim=0).tolist()
        
        y_true.append(int(r['label']))
        p_synth_list.append(probs[1])
        generators.append(r['generator'])
        
    return np.array(y_true), np.array(p_synth_list), generators

print("Evaluating VALIDATION set for threshold sweep...")
val_y, val_p, val_gen = evaluate_split_probabilities('data/v3/splits/val.csv')
print(f"Validation samples evaluated: {len(val_y)} (Real: {np.sum(val_y == 0)}, Synth: {np.sum(val_y == 1)})")

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

val_sweep_results = []

for th in thresholds:
    val_pred = (val_p >= th).astype(int)
    
    tp = int(np.sum((val_y == 1) & (val_pred == 1)))
    fp = int(np.sum((val_y == 0) & (val_pred == 1)))
    tn = int(np.sum((val_y == 0) & (val_pred == 0)))
    fn = int(np.sum((val_y == 1) & (val_pred == 0)))
    
    acc = (tp + tn) / (tp + fp + tn + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Synth recall
    real_rec = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Specificity / Real recall
    f1_synth = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    prec_real = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1_real = 2 * prec_real * real_rec / (prec_real + real_rec) if (prec_real + real_rec) > 0 else 0.0
    macro_f1 = (f1_synth + f1_real) / 2.0
    
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    
    val_sweep_results.append({
        'threshold': th,
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'real_recall': round(real_rec, 4),
        'synth_recall': round(rec, 4),
        'macro_f1': round(macro_f1, 4),
        'fpr': round(fpr, 4),
        'fnr': round(fnr, 4),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    })

print("\n=== VALIDATION THRESHOLD SWEEP RESULTS ===")
print("Thresh | Accuracy | Precision | Real Recall | Synth Recall | Macro-F1 |   FPR   |   FNR   | TP / FP / TN / FN")
print("-------|----------|-----------|-------------|--------------|----------|---------|---------|-------------------")
for r in val_sweep_results:
    print(f" {r['threshold']:.2f}  |  {r['accuracy']*100:5.2f}%  |  {r['precision']*100:6.2f}%  |   {r['real_recall']*100:6.2f}%   |    {r['synth_recall']*100:6.2f}%   |  {r['macro_f1']*100:5.2f}%  | {r['fpr']*100:5.2f}%  | {r['fnr']*100:5.2f}%  | {r['tp']:4d} / {r['fp']:3d} / {r['tn']:4d} / {r['fn']:3d}")

# Select best threshold by Macro-F1 / balanced FPR on validation
best_val = max(val_sweep_results, key=lambda x: x['macro_f1'])
print(f"\nOptimal Validation Threshold: {best_val['threshold']:.2f} (Macro-F1: {best_val['macro_f1']*100:.2f}%, FPR: {best_val['fpr']*100:.2f}%, FNR: {best_val['fnr']*100:.2f}%)")

# Now evaluate honest test set with the sweep
print("\nEvaluating TEST set across threshold sweep...")
test_y, test_p, test_gen = evaluate_split_probabilities('data/v3/splits/test.csv')
test_sweep_results = []

for th in thresholds:
    test_pred = (test_p >= th).astype(int)
    tp = int(np.sum((test_y == 1) & (test_pred == 1)))
    fp = int(np.sum((test_y == 0) & (test_pred == 1)))
    tn = int(np.sum((test_y == 0) & (test_pred == 0)))
    fn = int(np.sum((test_y == 1) & (test_pred == 0)))
    
    acc = (tp + tn) / (tp + fp + tn + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    real_rec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_synth = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    prec_real = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1_real = 2 * prec_real * real_rec / (prec_real + real_rec) if (prec_real + real_rec) > 0 else 0.0
    macro_f1 = (f1_synth + f1_real) / 2.0
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    
    test_sweep_results.append({
        'threshold': th,
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'real_recall': round(real_rec, 4),
        'synth_recall': round(rec, 4),
        'macro_f1': round(macro_f1, 4),
        'fpr': round(fpr, 4),
        'fnr': round(fnr, 4),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    })

print("\n=== TEST SET THRESHOLD SWEEP RESULTS ===")
print("Thresh | Accuracy | Precision | Real Recall | Synth Recall | Macro-F1 |   FPR   |   FNR   | TP / FP / TN / FN")
print("-------|----------|-----------|-------------|--------------|----------|---------|---------|-------------------")
for r in test_sweep_results:
    print(f" {r['threshold']:.2f}  |  {r['accuracy']*100:5.2f}%  |  {r['precision']*100:6.2f}%  |   {r['real_recall']*100:6.2f}%   |    {r['synth_recall']*100:6.2f}%   |  {r['macro_f1']*100:5.2f}%  | {r['fpr']*100:5.2f}%  | {r['fnr']*100:5.2f}%  | {r['tp']:4d} / {r['fp']:3d} / {r['tn']:4d} / {r['fn']:3d}")

# Save sweep results
sweep_file = Path('reports/v3_final_candidate/threshold_sweep_results.json')
with open(sweep_file, 'w', encoding='utf-8') as f:
    json.dump({
        'val_sweep': val_sweep_results,
        'test_sweep': test_sweep_results,
        'optimal_val_threshold': best_val
    }, f, indent=2)
print(f"\nSaved threshold sweep results to {sweep_file}")
