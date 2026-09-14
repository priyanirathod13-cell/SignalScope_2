"""
SignalScope CIFAKE Dataset Adapter
Downloads, validates, and ingests CIFAKE (CIFAR-10 real vs Stable Diffusion v1.4 synthetic).
Enforces SignalScope standard: REAL = 0, SYNTHETIC = 1.
"""

import os
import io
import time
import hashlib
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional
import pyarrow.parquet as pq
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
RAW_CIFAKE_DIR = PROJECT_ROOT / "data" / "raw" / "cifake"
DEFAULT_PARQUET_URL = "https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset/resolve/main/data/test-00000-of-00001.parquet"

def compute_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def download_cifake_parquet(url: str = DEFAULT_PARQUET_URL, dest_dir: Optional[Path] = None) -> Path:
    dest_dir = dest_dir or CACHE_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    dest_path = dest_dir / f"cifake_{filename}"
    
    if dest_path.exists() and dest_path.stat().st_size > 1024 * 1024:
        print(f"[CIFAKE] Using cached parquet file: {dest_path} ({dest_path.stat().st_size / 1024 / 1024:.2f} MB)")
        return dest_path

    print(f"[CIFAKE] Downloading parquet from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp:
        content = resp.read()
    dest_path.write_bytes(content)
    print(f"[CIFAKE] Downloaded {len(content) / 1024 / 1024:.2f} MB in {time.time() - t0:.2f}s -> {dest_path}")
    return dest_path

def ingest_cifake(
    limit_per_class: int = 8000,
    output_dir: Optional[Path] = None,
    parquet_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Ingests images from CIFAKE parquet.
    CIFAKE labels: 0 = FAKE, 1 = REAL.
    SignalScope labels: 0 = REAL, 1 = SYNTHETIC.
    """
    out_dir = output_dir or RAW_CIFAKE_DIR
    real_out = out_dir / "real"
    synth_out = out_dir / "synthetic"
    real_out.mkdir(parents=True, exist_ok=True)
    synth_out.mkdir(parents=True, exist_ok=True)

    if parquet_path is None or not parquet_path.exists():
        parquet_path = download_cifake_parquet()

    print(f"[CIFAKE] Reading parquet table: {parquet_path}...")
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    print(f"[CIFAKE] Total rows in parquet: {len(df)}")

    records = []
    real_count = 0
    synth_count = 0

    print(f"[CIFAKE] Extracting up to {limit_per_class} real and {limit_per_class} synthetic images...")
    for idx, row in df.iterrows():
        cifake_lbl = int(row["label"]) # 0 = FAKE, 1 = REAL
        img_dict = row["image"]
        img_bytes = img_dict["bytes"]

        if cifake_lbl == 1: # REAL
            if real_count >= limit_per_class:
                continue
            signal_label = 0
            label_str = "real"
            generator = "real_cifar10"
            gen_family = "real_camera"
            src_ds = "CIFAKE-CIFAR10"
            dest_folder = real_out
            fname = f"real_cifar10_{real_count:05d}.png"
            real_count += 1
        elif cifake_lbl == 0: # FAKE
            if synth_count >= limit_per_class:
                continue
            signal_label = 1
            label_str = "synthetic"
            generator = "stable_diffusion_v1_4"
            gen_family = "latent_diffusion"
            src_ds = "CIFAKE-SD14"
            dest_folder = synth_out
            fname = f"synthetic_stable_diffusion_v1_4_{synth_count:05d}.png"
            synth_count += 1
        else:
            continue

        file_path = dest_folder / fname
        if not file_path.exists():
            file_path.write_bytes(img_bytes)

        # Validate with PIL and extract image metadata
        try:
            with Image.open(io.BytesIO(img_bytes)) as pil_img:
                w, h = pil_img.size
                fmt = pil_img.format or "PNG"
        except Exception:
            w, h, fmt = 32, 32, "PNG"

        sha = compute_sha256_bytes(img_bytes)
        rel_path = str(file_path.relative_to(PROJECT_ROOT)).replace("\\", "/")

        records.append({
            "image_path": rel_path,
            "filename": fname,
            "label": signal_label,
            "label_str": label_str,
            "dataset_name": "CIFAKE",
            "source": src_ds,
            "generator": generator,
            "generator_family": gen_family,
            "original_dataset_split": "test",
            "width": w,
            "height": h,
            "file_format": fmt,
            "file_size": len(img_bytes),
            "sha256": sha
        })

        if real_count >= limit_per_class and synth_count >= limit_per_class:
            break

    print(f"[CIFAKE] Successfully extracted: {real_count} Real, {synth_count} Synthetic (Total: {len(records)})")
    return records

if __name__ == "__main__":
    ingest_cifake(limit_per_class=100)
