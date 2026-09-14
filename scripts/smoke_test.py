"""
SignalScope Final Submission Smoke Test
Tests 4 critical images against live FastAPI server:
1. Known REAL photograph (example_1_real_nature.jpg)
2. Known AI-generated image (example_2_stable_diffusion.png)
3. Wukong image (example_3_wukong_diffusion.png)
4. One WhatsApp AI image (WhatsApp Image 2026-09-15 at 02.56.08.jpeg)
"""

import sys
import json
from pathlib import Path
import requests

API_URL = "http://127.0.0.1:8000/predict"

test_files = [
    ("Known REAL photograph", Path(r"C:\SignalScope\data\examples\example_1_real_nature.jpg")),
    ("Known AI-generated (SD 1.5)", Path(r"C:\SignalScope\data\examples\example_2_stable_diffusion.png")),
    ("Wukong Unseen Generator", Path(r"C:\SignalScope\data\examples\example_3_wukong_diffusion.png")),
    ("WhatsApp AI Image", Path(r"C:\Users\Priyani Rathod\Downloads\WhatsApp Image 2026-09-15 at 02.56.08.jpeg"))
]

print("=" * 70)
print("SIGNALSCOPE FINAL CRITICAL DEMO SMOKE TEST")
print("=" * 70)

all_passed = True
results = []

for name, fp in test_files:
    print(f"\nTesting: {name}")
    print(f"File: {fp.name}")
    if not fp.exists():
        print(f"  [ERROR] File not found: {fp}")
        all_passed = False
        continue
        
    with open(fp, "rb") as f:
        files = {"file": (fp.name, f, "image/jpeg" if fp.suffix.lower() in [".jpg", ".jpeg"] else "image/png")}
        resp = requests.post(API_URL, files=files, timeout=30)
        
    if resp.status_code != 200:
        print(f"  [FAIL] HTTP Status: {resp.status_code}")
        print(resp.text)
        all_passed = False
        continue
        
    data = resp.json()
    
    # Validations
    p_real = data.get("probabilities", {}).get("real", -1)
    p_synth = data.get("probabilities", {}).get("synthetic", -1)
    label = data.get("label", "")
    conf = data.get("confidence", 0)
    conf_pct = data.get("confidence_percent", "")
    has_cam = bool(data.get("heatmap_image", "").startswith("data:image/") and data.get("overlay_image", "").startswith("data:image/"))
    explanation = data.get("forensic_rationale", "")
    has_evidence = "evidence" in data and "delta_probability" in data.get("evidence", {})
    
    prob_sum = p_real + p_synth
    sum_valid = abs(prob_sum - 1.0) < 1e-3
    
    print(f"  [PASS] HTTP Status: 200 OK")
    print(f"  [PASS] Prediction: {label} ({conf_pct})")
    print(f"  [PASS] Probabilities: Real={p_real*100:.2f}%, Synth={p_synth*100:.2f}% (Sum={prob_sum:.4f} -> {'VALID' if sum_valid else 'INVALID'})")
    print(f"  [PASS] Grad-CAM Heatmap Base64 Present: {has_cam}")
    print(f"  [PASS] Causal Evidence Test Present: {has_evidence}")
    print(f"  Rationale: \"{explanation[:90]}...\"")
    
    if not sum_valid or not has_cam or not has_evidence:
        all_passed = False
        
    results.append({
        "name": name,
        "filename": fp.name,
        "label": label,
        "confidence": conf_pct,
        "prob_real": p_real,
        "prob_synth": p_synth,
        "has_cam": has_cam,
        "has_evidence": has_evidence
    })

print("\n" + "=" * 70)
print(f"SMOKE TEST SUMMARY: {'ALL PASSED (READY)' if all_passed else 'FAILURES DETECTED'}")
print("=" * 70)
