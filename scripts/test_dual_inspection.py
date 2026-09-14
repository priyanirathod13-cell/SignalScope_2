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

with open('config/v3_train_config.json') as f:
    cfg = json.load(f)
model = build_model(cfg)
ckpt = torch.load('models/v3_final_candidate/best_model.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
model.eval()

norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def evaluate_dual_inspection(im):
    w, h = im.size
    is_square = (w == h)
    
    # 1. Letterbox view
    t_lb = norm(LetterboxTransform(224)(im)).unsqueeze(0)
    with torch.no_grad():
        out_lb = model(t_lb).squeeze()
    p_lb = torch.softmax(out_lb / 1.0195, dim=0).tolist()
    
    if is_square:
        return {
            'predicted_label': 'AI-GENERATED' if p_lb[1] > 0.5 else 'REAL',
            'confidence': max(p_lb),
            'p_synth': p_lb[1],
            'p_real': p_lb[0],
            'rule': 'square_direct'
        }
    
    # 2. Unpadded active content view
    cs = min(w, h)
    cc = im.crop(((w-cs)//2, (h-cs)//2, (w-cs)//2+cs, (h-cs)//2+cs)).resize((224, 224), Image.Resampling.BICUBIC)
    t_cc = norm(cc).unsqueeze(0)
    with torch.no_grad():
        out_cc = model(t_cc).squeeze()
    p_cc = torch.softmax(out_cc / 1.0195, dim=0).tolist()
    
    # 3. Laplacian variance
    gray_arr = np.array(im.convert('L'), dtype=np.float32)
    lap_var = float(scipy.ndimage.laplace(gray_arr).var())
    
    # Decision logic:
    # If the unpadded visual pixels exhibit strong synthetic indicators (p_cc_synth >= 0.60)
    # AND low-to-moderate high-frequency sensor noise (lap_var < 1200),
    # then the grey letterbox padding is masking an AI image.
    if p_cc[1] >= 0.60 and lap_var < 1200.0:
        return {
            'predicted_label': 'AI-GENERATED',
            'confidence': p_cc[1],
            'p_synth': p_cc[1],
            'p_real': p_cc[0],
            'lap_var': lap_var,
            'rule': 'content_synthetic_override'
        }
    else:
        return {
            'predicted_label': 'AI-GENERATED' if p_lb[1] > 0.5 else 'REAL',
            'confidence': max(p_lb),
            'p_synth': p_lb[1],
            'p_real': p_lb[0],
            'lap_var': lap_var,
            'rule': 'standard_letterbox'
        }

print('=== 1. PRELOADED EXAMPLES ===')
for ex in ['example_1_real_nature.jpg', 'example_2_stable_diffusion.png', 'example_3_wukong_diffusion.png', 'example_4_adm_diffusion.png', 'example_5_diagnostic_sky.jpeg']:
    p = Path('data/examples') / ex
    res = evaluate_dual_inspection(Image.open(p).convert('RGB'))
    print(f'{ex:30s}: {res["predicted_label"]:12s} ({res["confidence"]*100:5.1f}%) [rule: {res["rule"]}]')

print('\n=== 2. ALL WHATSAPP AI IMAGES ===')
downloads = Path(r'C:\Users\Priyani Rathod\Downloads')
ai_files = sorted(downloads.glob('WhatsApp Image 2026-09-1*.jpeg'))
ai_corr = 0
for p in ai_files:
    res = evaluate_dual_inspection(Image.open(p).convert('RGB'))
    is_correct = (res['predicted_label'] == 'AI-GENERATED')
    if is_correct: ai_corr += 1
    tag = 'CORRECT' if is_correct else 'FAIL'
    print(f'{p.name[-25:]:25s}: {res["predicted_label"]:12s} ({res["confidence"]*100:5.1f}%) -> {tag}')

print(f'WhatsApp AI Accuracy: {ai_corr}/{len(ai_files)} ({ai_corr/len(ai_files)*100:.1f}%)')

print('\n=== 3. REAL IMAGENET TEST SET (20 samples) ===')
with open('data/v3/splits/test.csv') as f:
    rows = list(csv.DictReader(f))
real_imgnet = [r['image_path'] for r in rows if r['generator'] == 'real_imagenet'][:20]
img_corr = 0
for p in real_imgnet:
    res = evaluate_dual_inspection(Image.open(p).convert('RGB'))
    if res['predicted_label'] == 'REAL': img_corr += 1
print(f'Real ImageNet Accuracy: {img_corr}/{len(real_imgnet)} ({img_corr/len(real_imgnet)*100:.1f}%)')

print('\n=== 4. REAL ASTRO SAMPLES (20 samples) ===')
astro_files = list(Path('data/v3/real').glob('real_astro*'))[:20]
astro_corr = 0
for p in astro_files:
    res = evaluate_dual_inspection(Image.open(p).convert('RGB'))
    if res['predicted_label'] == 'REAL': astro_corr += 1
print(f'Real Astro Accuracy: {astro_corr}/{len(astro_files)} ({astro_corr/len(astro_files)*100:.1f}%)')
