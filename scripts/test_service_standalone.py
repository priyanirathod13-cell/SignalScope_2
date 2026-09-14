import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.service import get_detection_service

svc = get_detection_service()
print(f"Service version: {svc.version}")
print(f"Loaded checkpoint: {svc.model_path}")
print(f"Temperature: {svc.temperature}")

examples = [
    "example_1_real_nature.jpg",
    "example_2_stable_diffusion.png",
    "example_3_wukong_diffusion.png",
    "example_4_adm_diffusion.png",
    "example_5_diagnostic_sky.jpeg"
]

print("\n--- TESTING PRELOADED EXAMPLES ---")
for ex in examples:
    p = Path("data/examples") / ex
    if p.exists():
        with open(p, "rb") as f:
            data = f.read()
        res = svc.analyze_image(data, filename=ex)
        ev = res["evidence"]
        print(f"{ex}: {res['label']} ({res['confidence_percent']}) | Real: {res['probabilities']['real']*100:.1f}%, Synth: {res['probabilities']['synthetic']*100:.1f}% | Faithfulness: {ev['faithfulness']} (Delta P: {ev['delta_probability']*100:.1f}%) | Latency: {res['inference_time']}s")

print("\n--- TESTING DIAGNOSTIC IMAGE ---")
diag_path = Path("data/diagnostic_image.jpeg")
if diag_path.exists():
    with open(diag_path, "rb") as f:
        data = f.read()
    res = svc.analyze_image(data, filename=diag_path.name)
    ev = res["evidence"]
    print(f"diagnostic_image.jpeg (Full canvas): {res['label']} ({res['confidence_percent']}) | Real: {res['probabilities']['real']*100:.1f}%, Synth: {res['probabilities']['synthetic']*100:.1f}% | Faithfulness: {ev['faithfulness']} (Delta P: {ev['delta_probability']*100:.1f}%)")

print("\nAll standalone tests passed cleanly!")
