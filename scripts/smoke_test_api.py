import urllib.request
import urllib.parse
import json
import mimetypes
import os
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

print(f"Connecting to {BASE_URL}...")

# 1. Test GET / (frontend static index)
req = urllib.request.Request(f"{BASE_URL}/")
with urllib.request.urlopen(req) as resp:
    status_code = resp.status
    content_type = resp.headers.get("Content-Type")
    html_data = resp.read().decode("utf-8")
    print(f"1. GET /: Status {status_code}, Content-Type: {content_type}, Length: {len(html_data)} chars")
    assert "SignalScope" in html_data, "Brand not found in HTML"
    assert "dropZone" in html_data, "Dropzone not found in HTML"

# 2. Test GET /api/health
req = urllib.request.Request(f"{BASE_URL}/api/health")
with urllib.request.urlopen(req) as resp:
    health_data = json.loads(resp.read().decode("utf-8"))
    print(f"2. GET /api/health: Status {resp.status}, Response: {json.dumps(health_data, indent=2)}")
    assert health_data["status"] == "healthy"

# 3. Test GET /api/examples
req = urllib.request.Request(f"{BASE_URL}/api/examples")
with urllib.request.urlopen(req) as resp:
    examples = json.loads(resp.read().decode("utf-8"))
    print(f"3. GET /api/examples: Found {len(examples)} examples")
    for ex in examples:
        print(f"   - {ex['id']}: {ex['title']} ({ex['source']})")

# 4. Test POST /api/predict-example with example_1_real_nature
post_data = json.dumps({"example_id": "example_1_real_nature"}).encode("utf-8")
req = urllib.request.Request(
    f"{BASE_URL}/api/predict-example",
    data=post_data,
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req) as resp:
    pred_res = json.loads(resp.read().decode("utf-8"))
    print(f"\n4. POST /api/predict-example (example_1_real_nature):")
    print(f"   Verdict: {pred_res['label']} ({pred_res['confidence_percent']})")
    print(f"   Model: {pred_res['model']}")
    print(f"   Latency: {pred_res['inference_time']}s")
    print(f"   Probabilities: Real={pred_res['probabilities']['real']*100:.1f}%, Synth={pred_res['probabilities']['synthetic']*100:.1f}%")
    print(f"   Faithfulness: {pred_res['evidence']['faithfulness']} (Delta P: {pred_res['evidence']['delta_probability']*100:.1f}%)")
    print(f"   Overlay Image present: {pred_res['overlay_image'].startswith('data:image/')}")
    print(f"   Heatmap Image present: {pred_res['heatmap_image'].startswith('data:image/')}")

# 5. Test POST /predict with multipart upload of a test image (example_2_stable_diffusion.png)
def upload_file(url, file_path):
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    filename = Path(file_path).name
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

upload_res = upload_file(f"{BASE_URL}/predict", "data/examples/example_2_stable_diffusion.png")
print(f"\n5. POST /predict (Multipart Upload - example_2_stable_diffusion.png):")
print(f"   Verdict: {upload_res['label']} ({upload_res['confidence_percent']})")
print(f"   Model: {upload_res['model']}")
print(f"   Probabilities: Real={upload_res['probabilities']['real']*100:.1f}%, Synth={upload_res['probabilities']['synthetic']*100:.1f}%")
print(f"   Evidence: {upload_res['evidence']['faithfulness']} (Delta P: {upload_res['evidence']['delta_probability']*100:.1f}%)")
print(f"   Rationale: {upload_res['forensic_rationale']}")

# 6. Test POST /predict with data/diagnostic_image.jpeg
diag_res = upload_file(f"{BASE_URL}/predict", "data/diagnostic_image.jpeg")
print(f"\n6. POST /predict (Multipart Upload - data/diagnostic_image.jpeg):")
print(f"   Verdict: {diag_res['label']} ({diag_res['confidence_percent']})")
print(f"   Model: {diag_res['model']}")
print(f"   Probabilities: Real={diag_res['probabilities']['real']*100:.1f}%, Synth={diag_res['probabilities']['synthetic']*100:.1f}%")
print(f"   Evidence: {diag_res['evidence']['faithfulness']} (Delta P: {diag_res['evidence']['delta_probability']*100:.1f}%)")
print(f"   Confidence Band: {diag_res['confidence_band']}")
print(f"   Rationale: {diag_res['forensic_rationale']}")

print("\n--- ALL API SMOKE TESTS PASSED CLEANLY ---")
