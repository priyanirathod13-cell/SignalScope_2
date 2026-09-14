"""
SignalScope Multi-Dataset Orchestrator & Validator
Unifies GenImage, CIFAKE, and future datasets into a leakage-safe,
deduplicated multi-dataset catalog with comprehensive metadata and audit logging.
"""

import os
import sys
import json

import time
import shutil
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "dataset_config.json"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
METADATA_DIR = DATA_DIR / "metadata"
QUARANTINE_DIR = DATA_DIR / "quarantine"
REPORTS_DIR = PROJECT_ROOT / "reports" / "dataset"

from src.data.cifake_adapter import ingest_cifake, compute_sha256_bytes

def load_config(config_path: Path = CONFIG_PATH) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file missing: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_sha256_file(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def detect_genimage_metadata(filename: str, label_str: str) -> Tuple[str, str, str]:
    fname_lower = filename.lower()
    if label_str == "real":
        return "real_imagenet", "real_camera", "ImageNet-1K"
    if "adm" in fname_lower:
        return "adm", "pixel_diffusion", "GenImage-ADM"
    elif "stable_diffusion" in fname_lower or "sd15" in fname_lower:
        return "stable_diffusion_v1_5", "latent_diffusion", "GenImage-SD15"
    elif "wukong" in fname_lower:
        return "wukong", "multilingual_diffusion", "GenImage-Wukong"
    return "unknown_synthetic", "unknown", "GenImage-Unknown"

def scan_genimage_raw(genimage_dir: Path) -> List[Dict[str, Any]]:
    records = []
    for label_str, label_int in [("real", 0), ("synthetic", 1)]:
        folder = genimage_dir / label_str
        if not folder.exists():
            continue
        print(f"[MultiDataset] Scanning GenImage {label_str} folder: {folder}...")
        for p in sorted(folder.glob("*.*")):
            if p.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                continue
            if p.stat().st_size == 0:
                continue
            try:
                with Image.open(p) as im:
                    w, h = im.size
                    fmt = im.format or p.suffix[1:].upper()
            except Exception:
                continue

            generator, family, source = detect_genimage_metadata(p.name, label_str)
            sha = compute_sha256_file(p)
            rel_path = str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")

            records.append({
                "image_path": rel_path,
                "filename": p.name,
                "label": label_int,
                "label_str": label_str,
                "dataset_name": "GenImage",
                "source": source,
                "generator": generator,
                "generator_family": family,
                "original_dataset_split": "train",
                "width": w,
                "height": h,
                "file_format": fmt,
                "file_size": p.stat().st_size,
                "sha256": sha
            })
    return records

def run_multi_dataset_pipeline(config_path: Path = CONFIG_PATH) -> pd.DataFrame:
    print("=" * 68)
    print("   SIGNALSCOPE: MULTI-DATASET EXPANSION & VALIDATION PIPELINE")
    print("=" * 68)

    cfg = load_config(config_path)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict[str, Any]] = []

    # 1. Ingest GenImage (Existing)
    # Checks if genimage is in data/raw/genimage or data/raw
    genimage_path = PROJECT_ROOT / cfg["datasets"]["genimage"].get("raw_dir", "data/raw/genimage")
    if not genimage_path.exists():
        fallback_path = DATA_DIR / "raw"
        if (fallback_path / "real").exists():
            genimage_path = fallback_path

    if cfg["datasets"]["genimage"].get("enabled", True):
        print(f"\n[1/4] Processing GenImage dataset from {genimage_path}...")
        gen_records = scan_genimage_raw(genimage_path)
        print(f"  -> Collected {len(gen_records)} GenImage records")
        all_records.extend(gen_records)

    # 2. Ingest CIFAKE (New)
    if cfg["datasets"].get("cifake", {}).get("enabled", True):
        limit = cfg["datasets"]["cifake"].get("sample_limit_per_class", 8000)
        cifake_raw_dir = PROJECT_ROOT / cfg["datasets"]["cifake"].get("raw_dir", "data/raw/cifake")
        print(f"\n[2/4] Processing CIFAKE dataset (target: {limit}/class) into {cifake_raw_dir}...")
        cifake_records = ingest_cifake(limit_per_class=limit, output_dir=cifake_raw_dir)
        print(f"  -> Collected {len(cifake_records)} CIFAKE records")
        all_records.extend(cifake_records)

    # 3. Deduplication & Data Quality Audit
    print(f"\n[3/4] Performing SHA-256 deduplication across {len(all_records)} total records...")
    df = pd.DataFrame(all_records)
    initial_count = len(df)

    # Detect duplicates
    duplicates_df = df[df.duplicated(subset=["sha256"], keep=False)].sort_values("sha256")
    dup_count = len(duplicates_df)
    print(f"  -> Found {dup_count} duplicated entries by SHA-256 hash.")

    # Deduplicate keeping the first canonical copy
    dedup_df = df.drop_duplicates(subset=["sha256"], keep="first").copy()
    print(f"  -> Clean unique records retained: {len(dedup_df)} (Removed {initial_count - len(dedup_df)} duplicates)")

    # 4. Generate Master Metadata Catalogs
    print(f"\n[4/4] Writing master metadata catalogs to {METADATA_DIR}...")
    
    # images.csv
    images_csv_path = METADATA_DIR / "images.csv"
    dedup_df.to_csv(images_csv_path, index=False)
    print(f"  -> Saved master images catalog: {images_csv_path} ({len(dedup_df)} rows)")

    # datasets.csv
    datasets_records = [
        {
            "dataset_name": "GenImage",
            "source_url": cfg["datasets"]["genimage"]["source"],
            "license": cfg["datasets"]["genimage"]["license"],
            "paper": cfg["datasets"]["genimage"]["paper"],
            "total_images": len(dedup_df[dedup_df["dataset_name"] == "GenImage"]),
            "real_count": len(dedup_df[(dedup_df["dataset_name"] == "GenImage") & (dedup_df["label"] == 0)]),
            "synthetic_count": len(dedup_df[(dedup_df["dataset_name"] == "GenImage") & (dedup_df["label"] == 1)])
        },
        {
            "dataset_name": "CIFAKE",
            "source_url": cfg["datasets"]["cifake"]["source"],
            "license": cfg["datasets"]["cifake"]["license"],
            "paper": cfg["datasets"]["cifake"]["paper"],
            "total_images": len(dedup_df[dedup_df["dataset_name"] == "CIFAKE"]),
            "real_count": len(dedup_df[(dedup_df["dataset_name"] == "CIFAKE") & (dedup_df["label"] == 0)]),
            "synthetic_count": len(dedup_df[(dedup_df["dataset_name"] == "CIFAKE") & (dedup_df["label"] == 1)])
        }
    ]
    datasets_df = pd.DataFrame(datasets_records)
    datasets_df.to_csv(METADATA_DIR / "datasets.csv", index=False)

    # generators.csv
    gen_summary = dedup_df.groupby(["generator", "generator_family", "label", "label_str"]).agg(
        count=("filename", "count"),
        dataset=("dataset_name", "first")
    ).reset_index()
    gen_summary["percentage"] = (gen_summary["count"] / len(dedup_df) * 100).round(2)
    gen_summary.to_csv(METADATA_DIR / "generators.csv", index=False)

    # duplicate_report.csv
    dup_report_path = REPORTS_DIR / "duplicate_report.csv"
    duplicates_df.to_csv(dup_report_path, index=False)

    print(f"\nMulti-Dataset Ingestion Summary:")
    print(f"  Total Real images:      {len(dedup_df[dedup_df['label'] == 0])}")
    print(f"  Total Synthetic images: {len(dedup_df[dedup_df['label'] == 1])}")
    print(f"  Total Unique images:    {len(dedup_df)}")
    print("\nGenerators breakdown:")
    print(gen_summary[["generator", "generator_family", "count", "percentage"]])

    return dedup_df

if __name__ == "__main__":
    run_multi_dataset_pipeline()
