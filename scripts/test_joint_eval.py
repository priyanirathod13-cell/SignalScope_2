import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from PIL import Image
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

downloads = Path(r'C:\Users\Priyani Rathod\Downloads')
ai_images = sorted(downloads.glob('WhatsApp Image 2026-09-1*.jpeg'))

def evaluate_image(im):
    w, h = im.size
    t_dr = norm(im.resize((224, 224), Image.BICUBIC)).unsqueeze(0)
    cs = min(w, h)
    cc = im.crop(((w-cs)//2, (h-cs)//2, (w-cs)//2+cs, (h-cs)//2+cs)).resize((224, 224), Image.BICUBIC)
    t_cc = norm(cc).unsqueeze(0)
    
    with torch.no_grad():
        out_dr = model(t_dr).squeeze()
        out_cc = model(t_cc).squeeze()
        
    mean_out = (out_dr + out_cc) / 2.0
    probs = torch.softmax(mean_out / 1.0195, dim=0).tolist()
    return probs

print('--- EVALUATING ALL WHATSAPP AI IMAGES ---')
ai_correct = 0
for p in ai_images:
    im = Image.open(p).convert('RGB')
    probs = evaluate_image(im)
    is_synth = probs[1] > 0.5
    if is_synth: ai_correct += 1
    verdict = "AI (CORRECT)" if is_synth else "REAL (FAIL)"
    print(f"{p.name[-25:]}: Synth={probs[1]*100:.1f}%, Real={probs[0]*100:.1f}% -> {verdict}")

print(f"\nTotal AI detected: {ai_correct}/{len(ai_images)} ({ai_correct/len(ai_images)*100:.1f}%)")

print('\n--- EVALUATING PRELOADED EXAMPLES ---')
for ex in ['example_1_real_nature.jpg', 'example_2_stable_diffusion.png', 'example_3_wukong_diffusion.png', 'example_4_adm_diffusion.png', 'example_5_diagnostic_sky.jpeg']:
    p = Path('data/examples') / ex
    if p.exists():
        im = Image.open(p).convert('RGB')
        probs = evaluate_image(im)
        lbl = 'SYNTH' if probs[1] > 0.5 else 'REAL'
        print(f"{ex}: {lbl} (Synth={probs[1]*100:.1f}%, Real={probs[0]*100:.1f}%)")
