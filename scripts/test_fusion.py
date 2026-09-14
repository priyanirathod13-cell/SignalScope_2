import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json, csv
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

with open('data/v3/splits/test.csv') as f:
    rows = list(csv.DictReader(f))

real_imgnet = [r for r in rows if r['generator'] == 'real_imagenet'][:30]
real_cifar = [r for r in rows if r['generator'] == 'real_cifar10'][:30]
synth_sd15 = [r for r in rows if r['generator'] == 'stable_diffusion_v1_5'][:30]
synth_sd14 = [r for r in rows if r['generator'] == 'stable_diffusion_v1_4'][:30]

downloads = Path(r'C:\Users\Priyani Rathod\Downloads')
user_ai = sorted(downloads.glob('WhatsApp Image 2026-09-1*.jpeg'))

def get_views(im):
    w, h = im.size
    t_lb = norm(LetterboxTransform(224)(im)).unsqueeze(0)
    t_dr = norm(im.resize((224, 224), Image.BICUBIC)).unsqueeze(0)
    cs = min(w, h)
    cc = im.crop(((w-cs)//2, (h-cs)//2, (w-cs)//2+cs, (h-cs)//2+cs)).resize((224, 224), Image.BICUBIC)
    t_cc = norm(cc).unsqueeze(0)
    
    with torch.no_grad():
        out_lb = model(t_lb).squeeze()
        out_dr = model(t_dr).squeeze()
        out_cc = model(t_cc).squeeze()
        
    p_lb = torch.softmax(out_lb / 1.0195, dim=0).tolist()
    p_dr = torch.softmax(out_dr / 1.0195, dim=0).tolist()
    p_cc = torch.softmax(out_cc / 1.0195, dim=0).tolist()
    return p_lb, p_dr, p_cc

print("Testing user AI images:")
for p in user_ai:
    im = Image.open(p).convert('RGB')
    pl, pd, pc = get_views(im)
    print(f"{p.name[-24:]}: LB={pl[1]*100:.1f}%, DR={pd[1]*100:.1f}%, CC={pc[1]*100:.1f}%")

print("\nTesting Real ImageNet samples:")
for r in real_imgnet[:5]:
    im = Image.open(r['image_path']).convert('RGB')
    pl, pd, pc = get_views(im)
    print(f"{Path(r['image_path']).name[-24:]}: LB_Real={pl[0]*100:.1f}%, DR_Real={pd[0]*100:.1f}%, CC_Real={pc[0]*100:.1f}%")
