import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json, csv
from PIL import Image
import numpy as np
import scipy.ndimage
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

# Load calibrated temperature
calib_file = Path('models/v3_final_candidate/temperature_calibration.json')
temp = 1.0195
if calib_file.exists():
    with open(calib_file) as cf:
        temp = json.load(cf).get('optimal_temperature', 1.0195)
print(f"Loaded V3 Model. Optimal temperature: {temp:.4f}")

norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Read test.csv
with open('data/v3/splits/test.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

real_samples = [r for r in rows if r['label'] == '0']
print(f"Total REAL test samples: {len(real_samples)}")

false_positives = []
correct_reals = []

for idx, r in enumerate(real_samples):
    img_path = Path(r['image_path'])
    if not img_path.exists():
        continue
    
    try:
        im = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        continue
        
    w, h = im.size
    ar = max(w / h, h / w)
    
    # Letterbox input
    t_lb = norm(LetterboxTransform(224)(im)).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(t_lb).squeeze()
        scaled = logits / temp
        probs = torch.softmax(scaled, dim=0).tolist()
    
    p_real = probs[0]
    p_synth = probs[1]
    pred = 1 if p_synth > 0.5 else 0
    
    # Measure image characteristics
    gray_arr = np.array(im.convert('L'), dtype=np.float32)
    lap_var = float(scipy.ndimage.laplace(gray_arr).var())
    brightness = float(np.mean(gray_arr))
    contrast = float(np.std(gray_arr))
    
    info = {
        'path': str(img_path),
        'name': img_path.name,
        'generator': r['generator'],
        'source_dataset': r.get('source_dataset', 'Unknown'),
        'width': w,
        'height': h,
        'aspect_ratio': round(ar, 3),
        'p_real': round(p_real, 4),
        'p_synth': round(p_synth, 4),
        'confidence': round(max(p_real, p_synth), 4),
        'lap_var': round(lap_var, 1),
        'brightness': round(brightness, 1),
        'contrast': round(contrast, 1),
        'raw_logits': [round(float(l), 4) for l in logits.tolist()]
    }
    
    if pred == 1:
        false_positives.append(info)
    else:
        correct_reals.append(info)

print(f"\nEvaluation Complete:")
print(f"Total True Reals: {len(correct_reals)}")
print(f"Total False Positives: {len(false_positives)}")
print(f"Observed FPR: {len(false_positives) / len(real_samples) * 100:.2f}%")

# Group FP by generator / source
fp_by_gen = {}
for fp in false_positives:
    gen = fp['generator']
    fp_by_gen[gen] = fp_by_gen.get(gen, 0) + 1

print("\n--- FALSE POSITIVES BY GENERATOR/SOURCE ---")
for gen, count in fp_by_gen.items():
    total_in_gen = sum(1 for r in real_samples if r['generator'] == gen)
    print(f"  {gen:20s}: {count:3d} / {total_in_gen:4d} ({count / total_in_gen * 100:.2f}%)")

# Statistical comparison: FP vs Correct
fp_synth_probs = [fp['p_synth'] for fp in false_positives]
fp_lap = [fp['lap_var'] for fp in false_positives]
cr_lap = [cr['lap_var'] for cr in correct_reals]
fp_bright = [fp['brightness'] for fp in false_positives]
cr_bright = [cr['brightness'] for cr in correct_reals]
fp_contrast = [fp['contrast'] for fp in false_positives]
cr_contrast = [cr['contrast'] for cr in correct_reals]

print("\n--- STATISTICAL COMPARISON (FP vs Correct Reals) ---")
print(f"FP Synthetic Probability Range: min={min(fp_synth_probs):.4f}, max={max(fp_synth_probs):.4f}, median={np.median(fp_synth_probs):.4f}")
print(f"FP Laplacian Variance (Sharpness): median={np.median(fp_lap):.1f}, mean={np.mean(fp_lap):.1f}")
print(f"Correct Laplacian Variance:       median={np.median(cr_lap):.1f}, mean={np.mean(cr_lap):.1f}")
print(f"FP Brightness (0-255):            median={np.median(fp_bright):.1f}, mean={np.mean(fp_bright):.1f}")
print(f"Correct Brightness:               median={np.median(cr_bright):.1f}, mean={np.mean(cr_bright):.1f}")
print(f"FP Contrast (Std Dev):            median={np.median(fp_contrast):.1f}, mean={np.mean(fp_contrast):.1f}")
print(f"Correct Contrast:                 median={np.median(cr_contrast):.1f}, mean={np.mean(cr_contrast):.1f}")

# Save detailed results to JSON
out_json = Path('reports/v3_final_candidate/real_false_positives_detailed.json')
out_json.parent.mkdir(parents=True, exist_ok=True)
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump({
        'total_real_samples': len(real_samples),
        'total_false_positives': len(false_positives),
        'fpr': len(false_positives) / len(real_samples),
        'fp_by_generator': fp_by_gen,
        'false_positives': false_positives
    }, f, indent=2)
print(f"\nSaved detailed analysis to {out_json}")
