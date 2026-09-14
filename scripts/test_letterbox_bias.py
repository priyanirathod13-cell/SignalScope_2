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

def lb_fn(im):
    return norm(LetterboxTransform(224)(im)).unsqueeze(0)

def dr_fn(im):
    return norm(im.resize((224, 224), Image.Resampling.BICUBIC)).unsqueeze(0)

# Load metadata
import csv
with open('data/v3/splits/test.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

imagenet_real = [r for r in rows if r['generator'] == 'real_imagenet'][:30]
sd15_synth = [r for r in rows if r['generator'] == 'stable_diffusion_v1_5'][:30]
cifar_real = [r for r in rows if r['generator'] == 'real_cifar10'][:30]
cifar_synth = [r for r in rows if r['generator'] == 'stable_diffusion_v1_4'][:30]

def evaluate_subset(name, subset, fn):
    corr = 0
    for r in subset:
        im = Image.open(r['image_path']).convert('RGB')
        t = fn(im)
        with torch.no_grad():
            pred = model(t).argmax(dim=1).item()
        if pred == int(r['label']):
            corr += 1
    print(f"{name}: {corr}/{len(subset)} ({corr/len(subset)*100:.1f}%)")

print("--- EVALUATION WITH LETTERBOX ---")
evaluate_subset("Real ImageNet", imagenet_real, lb_fn)
evaluate_subset("Real CIFAR-10", cifar_real, lb_fn)
evaluate_subset("Synth SD 1.5", sd15_synth, lb_fn)
evaluate_subset("Synth SD 1.4", cifar_synth, lb_fn)

print("\n--- EVALUATION WITH DIRECT RESIZE ---")
evaluate_subset("Real ImageNet", imagenet_real, dr_fn)
evaluate_subset("Real CIFAR-10", cifar_real, dr_fn)
evaluate_subset("Synth SD 1.5", sd15_synth, dr_fn)
evaluate_subset("Synth SD 1.4", cifar_synth, dr_fn)
