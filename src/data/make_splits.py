"""
SignalScope Stratified Dataset Splitter
Partitions data/raw into reproducible train/validation/test sets (70/15/15)
stratified across labels (real vs. synthetic) and generative models.
Produces data/metadata/dataset_inventory.csv and data/metadata/splits_summary.json.
"""

import os
import json
import random
import shutil
import hashlib
from pathlib import Path
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"
METADATA_DIR = DATA_DIR / "metadata"

SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
RANDOM_SEED = 42

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def detect_generator(filename: str, label: str) -> str:
    fname_lower = filename.lower()
    if label == "real":
        return "real_imagenet"
    if "adm" in fname_lower:
        return "adm"
    elif "stable_diffusion" in fname_lower or "sd15" in fname_lower:
        return "stable_diffusion_v1_5"
    elif "wukong" in fname_lower:
        return "wukong"
    return "unknown_synthetic"

def create_split_directories():
    for split in ["train", "validation", "test"]:
        for label in ["real", "synthetic"]:
            (SPLITS_DIR / split / label).mkdir(parents=True, exist_ok=True)

def generate_splits():
    random.seed(RANDOM_SEED)
    create_split_directories()
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    
    # 1. Gather all raw files
    for label in ["real", "synthetic"]:
        folder = RAW_DIR / label
        if not folder.exists():
            continue
        for img_path in sorted(folder.glob("*.*")):
            if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                continue
            generator = detect_generator(img_path.name, label)
            records.append({
                "filename": img_path.name,
                "label": label,
                "generator": generator,
                "raw_path": img_path
            })

    df = pd.DataFrame(records)
    if len(df) == 0:
        print("No raw images found to split!")
        return

    print(f"Found {len(df)} total raw images:")
    print(df.groupby(["label", "generator"]).size())

    # 2. Stratified Partitioning across (label, generator) groups
    assigned_records = []
    
    for (lbl, gen), group in df.groupby(["label", "generator"]):
        group_items = group.to_dict("records")
        random.shuffle(group_items)
        n = len(group_items)
        n_train = int(round(n * SPLIT_RATIOS["train"]))
        n_val = int(round(n * SPLIT_RATIOS["validation"]))
        
        # Ensure remaining go to test
        train_items = group_items[:n_train]
        val_items = group_items[n_train:n_train + n_val]
        test_items = group_items[n_train + n_val:]

        for item in train_items:
            item["split"] = "train"
            assigned_records.append(item)
        for item in val_items:
            item["split"] = "validation"
            assigned_records.append(item)
        for item in test_items:
            item["split"] = "test"
            assigned_records.append(item)

    # 3. Copy files to split directories and record metadata
    print("Writing split images and compiling inventory metadata...")
    inventory = []
    summary = {"train": {}, "validation": {}, "test": {}, "total": len(assigned_records)}

    for idx, item in enumerate(assigned_records):
        src_path = item["raw_path"]
        split = item["split"]
        label = item["label"]
        generator = item["generator"]
        fname = item["filename"]

        dest_folder = SPLITS_DIR / split / label
        dest_path = dest_folder / fname

        if not dest_path.exists():
            shutil.copy2(src_path, dest_path)

        # Image metadata
        try:
            with Image.open(src_path) as img:
                w, h = img.size
                fmt = img.format
                mode = img.mode
        except Exception:
            w, h, fmt, mode = 0, 0, "UNKNOWN", "UNKNOWN"

        file_hash = compute_sha256(src_path)

        inventory.append({
            "image_id": f"img_{idx:05d}",
            "filename": fname,
            "label": label,
            "generator": generator,
            "split": split,
            "raw_relpath": str(src_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "split_relpath": str(dest_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "width": w,
            "height": h,
            "channels": len(mode) if mode else 3,
            "format": fmt,
            "sha256": file_hash
        })

    # Save inventory CSV
    inventory_df = pd.DataFrame(inventory)
    csv_path = METADATA_DIR / "dataset_inventory.csv"
    inventory_df.to_csv(csv_path, index=False)
    print(f"Inventory saved: {csv_path} ({len(inventory_df)} records)")

    # Save summary JSON
    breakdown = {}
    for (split_name, lbl, gen), cnt in inventory_df.groupby(["split", "label", "generator"]).size().items():
        if split_name not in breakdown:
            breakdown[split_name] = {}
        if lbl not in breakdown[split_name]:
            breakdown[split_name][lbl] = {}
        breakdown[split_name][lbl][gen] = int(cnt)

    summary_data = {
        "dataset_name": "GenImage Benchmark Subset",
        "total_images": len(inventory_df),
        "class_counts": {k: int(v) for k, v in inventory_df["label"].value_counts().items()},
        "generator_counts": {k: int(v) for k, v in inventory_df["generator"].value_counts().items()},
        "split_counts": {k: int(v) for k, v in inventory_df["split"].value_counts().items()},
        "detailed_breakdown": breakdown
    }
    json_path = METADATA_DIR / "splits_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary saved: {json_path}")

if __name__ == "__main__":
    generate_splits()
