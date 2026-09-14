"""
SignalScope Backend API Application
FastAPI web server providing /predict, /api/examples, and static frontend hosting.
"""

import io
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.service import get_detection_service


app = FastAPI(
    title="SignalScope API",
    description="Forensic AI Image Authenticity Detection & Explainability Engine",
    version="1.0.0"
)

# Enable CORS for local cross-origin development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "data" / "examples"
FRONTEND_DIR = PROJECT_ROOT / "frontend"


# Pre-defined gallery of test examples
SAMPLE_REGISTRY = [
    {
        "id": "example_1_real_nature",
        "title": "Authentic Photo (Nature)",
        "source": "ImageNet (Camera Photography)",
        "expected_label": "REAL",
        "filename": "example_1_real_nature.jpg"
    },
    {
        "id": "example_2_stable_diffusion",
        "title": "Stable Diffusion v1.5",
        "source": "Latent Diffusion Model",
        "expected_label": "AI-GENERATED",
        "filename": "example_2_stable_diffusion.png"
    },
    {
        "id": "example_3_wukong_diffusion",
        "title": "Wukong Diffusion",
        "source": "Multilingual Diffusion Model",
        "expected_label": "AI-GENERATED",
        "filename": "example_3_wukong_diffusion.png"
    },
    {
        "id": "example_4_adm_diffusion",
        "title": "ADM Guided Diffusion",
        "source": "Ablated Diffusion Model",
        "expected_label": "AI-GENERATED",
        "filename": "example_4_adm_diffusion.png"
    },
    {
        "id": "example_5_diagnostic_sky",
        "title": "Night Sky & Landscape",
        "source": "Challenging Forensic Benchmark",
        "expected_label": "AI-GENERATED",
        "filename": "example_5_diagnostic_sky.jpeg"
    }
]


@app.on_event("startup")
def startup_event():
    """Pre-warm the PyTorch model into memory at application launch."""
    get_detection_service()


@app.get("/api/health")
def health_check():
    """Healthcheck endpoint verifying model status."""
    service = get_detection_service()
    return {
        "status": "healthy",
        "service": "SignalScope",
        "model": "EfficientNet-B0 (SignalScopeClassifier)",
        "device": str(service.device),
        "input_resolution": service.image_size
    }


@app.get("/api/examples")
def list_examples() -> List[Dict[str, Any]]:
    """Returns the list of available preloaded sample images."""
    items = []
    for item in SAMPLE_REGISTRY:
        file_path = EXAMPLES_DIR / item["filename"]
        if file_path.exists():
            items.append({
                "id": item["id"],
                "title": item["title"],
                "source": item["source"],
                "expected_label": item["expected_label"],
                "thumbnail_url": f"/api/examples/{item['id']}/image"
            })
    return items


@app.get("/api/examples/{example_id}/image")
def get_example_image(example_id: str):
    """Serves the raw image file for a given example."""
    match = next((item for item in SAMPLE_REGISTRY if item["id"] == example_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="Example image not found.")

    file_path = EXAMPLES_DIR / match["filename"]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Example file is missing from disk.")

    media_type = "image/jpeg" if file_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
    return FileResponse(file_path, media_type=media_type)


class ExamplePredictRequest(BaseModel):
    example_id: str


@app.post("/api/predict-example")
def predict_example(payload: ExamplePredictRequest):
    """Performs analysis on a chosen preloaded example image."""
    match = next((item for item in SAMPLE_REGISTRY if item["id"] == payload.example_id), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Invalid example_id: '{payload.example_id}'")

    file_path = EXAMPLES_DIR / match["filename"]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Example file missing on server.")

    with open(file_path, "rb") as f:
        image_bytes = f.read()

    service = get_detection_service()
    try:
        res = service.analyze_image(image_bytes, filename=match["filename"])
        print(f"[EXAMPLE] Analyzed '{payload.example_id}' -> Verdict: {res['label']} ({res['confidence_percent']}) | Real: {res['probabilities']['real']*100:.1f}%, Synth: {res['probabilities']['synthetic']*100:.1f}%")
        return res
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal inference failure.")


@app.post("/predict")
@app.post("/api/predict")
async def predict_image(
    file: UploadFile = File(...),
    colormap: str = Query("jet", pattern="^(jet|turbo)$"),
    alpha: float = Query(0.5, ge=0.0, le=1.0)
):
    """
    Main image analysis endpoint.
    Accepts multipart image upload, validates format, runs CNN forward inference
    and Grad-CAM attribution, returning classification results and visual overlays.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided in upload.")

    try:
        image_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read uploaded file stream.")

    # Save last uploaded file for diagnostic inspection
    debug_path = PROJECT_ROOT / "data" / "debug_last_upload.png"
    try:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_path, "wb") as df:
            df.write(image_bytes)
    except Exception:
        pass

    service = get_detection_service()
    try:
        result = service.analyze_image(
            image_bytes=image_bytes,
            filename=file.filename,
            colormap=colormap,
            alpha=alpha
        )
        print(f"[PREDICT] Uploaded '{file.filename}' -> Verdict: {result['label']} ({result['confidence_percent']}) | Real: {result['probabilities']['real']*100:.1f}%, Synth: {result['probabilities']['synthetic']*100:.1f}%")
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        # Avoid exposing raw Python stack traces
        print(f"[API ERROR] {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the image. Please verify file integrity."
        )


# Mount frontend static files if available
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
