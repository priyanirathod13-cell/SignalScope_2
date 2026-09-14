import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from src.models import build_model
from src.training.dataset import LetterboxTransform
from src.explainability.gradcam import GradCAM

with open('config/v3_train_config.json') as f:
    cfg = json.load(f)
model = build_model(cfg)
ckpt = torch.load('models/v3_final_candidate/best_model.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
model.eval()

# Load FP details
with open('reports/v3_final_candidate/real_false_positives_detailed.json') as f:
    data = json.load(f)

fps = data['false_positives']
# Take 3 CIFAR and 3 ImageNet representative false positives
cifar_fps = [fp for fp in fps if fp['generator'] == 'real_cifar10'][:3]
imgnet_fps = [fp for fp in fps if fp['generator'] == 'real_imagenet'][:3]

norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("=== GRAD-CAM ON REPRESENTATIVE REAL FALSE POSITIVES ===")
for sample in cifar_fps + imgnet_fps:
    im = Image.open(sample['path']).convert('RGB')
    t = norm(LetterboxTransform(224)(im)).unsqueeze(0)
    with GradCAM(model) as gcam:
        res = gcam.generate(t, target_class=1) # Synthetic class activation
    cam = res['cam_tensor'].squeeze().numpy()
    peak = float(np.max(cam))
    mean_val = float(np.mean(cam))
    print(f"{sample['name']:30s} ({sample['generator']:15s}): p_synth={sample['p_synth']:.4f} | CAM Peak: {peak:.4f}, Mean: {mean_val:.4f}")
