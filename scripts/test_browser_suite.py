import requests
import json
from pathlib import Path

url = 'http://127.0.0.1:8000/predict'

# Gather the test images:
downloads = Path(r'C:\Users\Priyani Rathod\Downloads')

# 1. Ordinary Real Photographs:
# - Gym equipment: WhatsApp Image 2026-09-15 at 02.08.20.jpeg
# - Young man in polo against wood wall: WhatsApp Image 2026-09-15 at 02.08.28.jpeg
# - Young man studying at desk: WhatsApp Image 2026-09-15 at 02.10.51.jpeg
ordinary_real = [
    ('Ordinary Real 1 (Gym Room)', downloads / 'WhatsApp Image 2026-09-15 at 02.08.20.jpeg'),
    ('Ordinary Real 2 (Portrait Wall)', downloads / 'WhatsApp Image 2026-09-15 at 02.08.28.jpeg'),
    ('Ordinary Real 3 (Indoor Desk Study)', downloads / 'WhatsApp Image 2026-09-15 at 02.10.51.jpeg'),
]

# 2. Unusual Real Photographs:
# - Sunset over dark water: WhatsApp Image 2026-09-15 at 02.09.34.jpeg
# - Nature macro/wildlife: data/examples/example_1_real_nature.jpg
# - Magazine cover poster: WhatsApp Image 2026-09-15 at 02.08.48.jpeg
unusual_real = [
    ('Unusual Real 1 (Sunset Over Water)', downloads / 'WhatsApp Image 2026-09-15 at 02.09.34.jpeg'),
    ('Unusual Real 2 (Nature / Stingray)', Path('data/examples/example_1_real_nature.jpg')),
    ('Unusual Real 3 (Magazine Style Photo)', downloads / 'WhatsApp Image 2026-09-15 at 02.08.48.jpeg'),
]

# 3. Real Night / Sky Image:
# - data/v3/real/real_astro_night_sky_0011.jpg
night_sky_real = [
    ('Real Night/Sky Image', Path('data/v3/real/real_astro_night_sky_0011.jpg')),
]

# 4. Real Astrophotography-like Image:
# - data/v3/real/real_astro_night_sky_0000.jpg
astro_real = [
    ('Real Astrophotography Image', Path('data/v3/real/real_astro_night_sky_0000.jpg')),
]

# 5. 3 Synthetic Images:
# - example_2_stable_diffusion.png
# - example_3_wukong_diffusion.png
# - example_4_adm_diffusion.png
synthetic_images = [
    ('Synthetic 1 (Stable Diffusion v1.5)', Path('data/examples/example_2_stable_diffusion.png')),
    ('Synthetic 2 (Wukong Diffusion)', Path('data/examples/example_3_wukong_diffusion.png')),
    ('Synthetic 3 (ADM Diffusion)', Path('data/examples/example_4_adm_diffusion.png')),
]

# 6. Original V2 Diagnostic Image:
# - data/diagnostic_image.jpeg
diagnostic_image = [
    ('Original V2 Diagnostic Image', Path('data/diagnostic_image.jpeg')),
]

all_tests = ordinary_real + unusual_real + night_sky_real + astro_real + synthetic_images + diagnostic_image

results = []

print("=== RUNNING EMPIRICAL BROWSER HTTP TEST SUITE ===")
for category, p in all_tests:
    if not p.exists():
        print(f"Missing file: {p}")
        continue
    
    with open(p, 'rb') as f:
        resp = requests.post(url, files={'file': (p.name, f, 'image/jpeg')})
    
    if resp.status_code == 200:
        d = resp.json()
        lbl = d['label']
        conf = d['confidence_percent']
        pr = d['probabilities']['real'] * 100
        ps = d['probabilities']['synthetic'] * 100
        faith = d['evidence']['faithfulness']
        dp = d['evidence']['delta_probability'] * 100
        lat = d['inference_time']
        
        entry = {
            'category': category,
            'filename': p.name,
            'prediction': lbl,
            'confidence': conf,
            'prob_real': round(pr, 2),
            'prob_synth': round(ps, 2),
            'faithfulness': faith,
            'delta_p': round(dp, 2),
            'latency_s': lat
        }
        results.append(entry)
        print(f"{category:36s} | {p.name[-25:]:25s} | {lbl:12s} ({conf}) | Real: {pr:5.1f}% | Synth: {ps:5.1f}% | Faith: {faith} (dp={dp:+.1f}%) | Lat: {lat:.3f}s")
    else:
        print(f"Error {resp.status_code} for {p.name}")

out_path = Path('reports/v3_final_candidate/browser_suite_results.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved suite results to {out_path}")
