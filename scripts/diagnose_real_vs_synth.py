import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json, csv
from PIL import Image, ImageOps
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

def get_tensors(im):
    w, h = im.size
    # 1. Letterbox (grey 128)
    t_lb = norm(LetterboxTransform(224)(im)).unsqueeze(0)
    
    # 2. Direct Resize (squished)
    t_dr = norm(im.resize((224, 224), Image.Resampling.BICUBIC)).unsqueeze(0)
    
    # 3. Center Crop (unpadded square crop min(w, h), resized to 224)
    cs = min(w, h)
    cc = im.crop(((w-cs)//2, (h-cs)//2, (w-cs)//2+cs, (h-cs)//2+cs)).resize((224, 224), Image.Resampling.BICUBIC)
    t_cc = norm(cc).unsqueeze(0)
    
    # 4. Black padding (0,0,0)
    t_lb_black = norm(LetterboxTransform(224, fill=(0, 0, 0))(im)).unsqueeze(0)

    # 5. Replicate border padding
    scale = 224.0 / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized_im = im.resize((nw, nh), Image.Resampling.BICUBIC)
    pad_w = (224 - nw) // 2
    pad_h = (224 - nh) // 2
    arr = transforms.ToTensor()(resized_im).unsqueeze(0)
    padded_tensor = torch.nn.functional.pad(arr, (pad_w, 224 - nw - pad_w, pad_h, 224 - nh - pad_h), mode='replicate')
    padded_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(padded_tensor.squeeze(0)).unsqueeze(0)

    with torch.no_grad():
        logits_lb = model(t_lb).squeeze()
        logits_dr = model(t_dr).squeeze()
        logits_cc = model(t_cc).squeeze()
        logits_blk = model(t_lb_black).squeeze()
        logits_rep = model(padded_norm).squeeze()

    def p(l):
        return torch.softmax(l / 1.0195, dim=0).tolist()

    return {
        'lb_grey': p(logits_lb),
        'dr': p(logits_dr),
        'cc': p(logits_cc),
        'blk': p(logits_blk),
        'rep': p(logits_rep),
    }

# Test on 10 Real ImageNet images
with open('data/v3/splits/test.csv') as f:
    rows = list(csv.DictReader(f))
real_imgnet = [r for r in rows if r['generator'] == 'real_imagenet'][:10]

print("=== REAL IMAGENET SAMPLES (Expected: REAL) ===")
for r in real_imgnet:
    im = Image.open(r['image_path']).convert('RGB')
    res = get_tensors(im)
    name = Path(r['image_path']).name[:20]
    print(f"{name:20s} | LB_Grey={res['lb_grey'][0]*100:5.1f}% | DR={res['dr'][0]*100:5.1f}% | CC={res['cc'][0]*100:5.1f}% | BLK={res['blk'][0]*100:5.1f}% | REP={res['rep'][0]*100:5.1f}%")

print("\n=== USER WHATSAPP AI IMAGES (Expected: SYNTH) ===")
downloads = Path(r'C:\Users\Priyani Rathod\Downloads')
for p in sorted(downloads.glob('WhatsApp Image 2026-09-15 at 02.08*.jpeg')) + sorted(downloads.glob('WhatsApp Image 2026-09-15 at 02.09*.jpeg')) + sorted(downloads.glob('WhatsApp Image 2026-09-15 at 02.10*.jpeg')):
    im = Image.open(p).convert('RGB')
    res = get_tensors(im)
    name = p.name[-24:]
    print(f"{name:24s} | LB_Grey={res['lb_grey'][1]*100:5.1f}% | DR={res['dr'][1]*100:5.1f}% | CC={res['cc'][1]*100:5.1f}% | BLK={res['blk'][1]*100:5.1f}% | REP={res['rep'][1]*100:5.1f}%")

print("\n=== PRELOADED EXAMPLES ===")
for ex in ['example_1_real_nature.jpg', 'example_2_stable_diffusion.png', 'example_3_wukong_diffusion.png', 'example_4_adm_diffusion.png', 'example_5_diagnostic_sky.jpeg']:
    p = Path('data/examples') / ex
    if p.exists():
        im = Image.open(p).convert('RGB')
        res = get_tensors(im)
        print(f"{ex:30s} | LB_Grey Real={res['lb_grey'][0]*100:5.1f}% Synth={res['lb_grey'][1]*100:5.1f}% | CC Real={res['cc'][0]*100:5.1f}% Synth={res['cc'][1]*100:5.1f}% | REP Real={res['rep'][0]*100:5.1f}% Synth={res['rep'][1]*100:5.1f}%")
