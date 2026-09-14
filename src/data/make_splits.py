"""
SignalScope Leakage-Safe Multi-Dataset Splitter
Partitions images from data/metadata/images.csv into reproducible 70/15/15 splits.
Supports generator holdouts (Phase 9) and enforces zero cross-split duplicate leakage.
Produces data/splits/{train,val,test}.csv and backward-compatible split directory structures.
"""

import os
import sys
import json

import random
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "dataset_config.json"
DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
SPLITS_DIR = DATA_DIR / "splits"
REPORTS_DIR = PROJECT_ROOT / "reports" / "dataset"

def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def create_split_folders():
    for split in ["train", "validation", "test"]:
        for label_str in ["real", "synthetic"]:
            (SPLITS_DIR / split / label_str).mkdir(parents=True, exist_ok=True)

def generate_multi_splits(config_path: Path = CONFIG_PATH):
    print("=" * 68)
    print("   SIGNALSCOPE: LEAKAGE-SAFE MULTI-DATASET SPLIT GENERATION")
    print("=" * 68)

    cfg = load_config(config_path)
    seed = cfg.get("splits", {}).get("random_seed", 42)
    random.seed(seed)

    images_csv_path = METADATA_DIR / "images.csv"
    if not images_csv_path.exists():
        raise FileNotFoundError(f"Master images catalog missing at: {images_csv_path}. Run multi_dataset.py first.")

    df = pd.read_csv(images_csv_path)
    print(f"Loaded master catalog with {len(df)} deduplicated records.")

    train_ratio = cfg.get("splits", {}).get("train_ratio", 0.70)
    val_ratio = cfg.get("splits", {}).get("val_ratio", 0.15)
    test_ratio = cfg.get("splits", {}).get("test_ratio", 0.15)

    holdout_cfg = cfg.get("generator_holdout", {})
    holdout_enabled = holdout_cfg.get("enabled", False)
    held_out_generators = set(holdout_cfg.get("generators", []))

    if holdout_enabled and held_out_generators:
        print(f"\n[Phase 9] Generator Holdout ENABLED for: {held_out_generators}")
        print("  -> All images from these generators will be placed strictly into the TEST split.")
    else:
        print("\n[Phase 9] Generator Holdout: Disabled (all generators stratified across train/val/test).")

    create_split_folders()
    assigned_records = []

    # Partition by (label, generator) groups
    for (label_int, generator), group in df.groupby(["label", "generator"]):
        group_items = group.to_dict("records")
        random.shuffle(group_items)

        # Check if held out
        if holdout_enabled and generator in held_out_generators:
            for item in group_items:
                item["split"] = "test"
                assigned_records.append(item)
            print(f"  -> Held-out group ({generator}): {len(group_items)} items -> TEST ONLY")
            continue

        n = len(group_items)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        
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

    split_df = pd.DataFrame(assigned_records)

    # Verify zero leakage across splits
    train_hashes = set(split_df[split_df["split"] == "train"]["sha256"])
    val_hashes = set(split_df[split_df["split"] == "validation"]["sha256"])
    test_hashes = set(split_df[split_df["split"] == "test"]["sha256"])

    leak_train_val = train_hashes.intersection(val_hashes)
    leak_train_test = train_hashes.intersection(test_hashes)
    leak_val_test = val_hashes.intersection(test_hashes)

    if leak_train_val or leak_train_test or leak_val_test:
        raise RuntimeError(f"FATAL: Cross-split leakage detected! Train-Val: {len(leak_train_val)}, Train-Test: {len(leak_train_test)}, Val-Test: {len(leak_val_test)}")
    print("\n[Verification] Cross-split leakage audit: PASS (0 overlapping hashes between train, validation, and test)")

    # Save split manifests
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(SPLITS_DIR / "all_splits.csv", index=False)
    split_df[split_df["split"] == "train"].to_csv(SPLITS_DIR / "train.csv", index=False)
    split_df[split_df["split"] == "validation"].to_csv(SPLITS_DIR / "val.csv", index=False)
    split_df[split_df["split"] == "test"].to_csv(SPLITS_DIR / "test.csv", index=False)
    split_df.to_csv(METADATA_DIR / "images.csv", index=False)

    # Populate split directories (backward-compatible with SignalScopeDataset)
    print("\nPopulating split directories with image files...")
    copied_counts = {"train": 0, "validation": 0, "test": 0}
    for idx, row in split_df.iterrows():
        split_name = row["split"]
        label_folder = "real" if row["label"] == 0 else "synthetic"
        src_path = PROJECT_ROOT / row["image_path"]
        dest_path = SPLITS_DIR / split_name / label_folder / row["filename"]
        
        if not dest_path.exists() and src_path.exists():
            shutil.copy2(src_path, dest_path)
            copied_counts[split_name] += 1

    # Generate split summary matrix
    split_matrix = split_df.groupby(["split", "label_str"]).size().unstack(fill_value=0)
    print("\nSplit Matrix (Counts):")
    print(split_matrix)

    # Save splits_summary.json
    splits_summary = {
        "total_images": len(split_df),
        "train_count": len(split_df[split_df["split"] == "train"]),
        "validation_count": len(split_df[split_df["split"] == "validation"]),
        "test_count": len(split_df[split_df["split"] == "test"]),
        "matrix": {
            "train": {
                "real": int(len(split_df[(split_df["split"] == "train") & (split_df["label"] == 0)])),
                "synthetic": int(len(split_df[(split_df["split"] == "train") & (split_df["label"] == 1)]))
            },
            "validation": {
                "real": int(len(split_df[(split_df["split"] == "validation") & (split_df["label"] == 0)])),
                "synthetic": int(len(split_df[(split_df["split"] == "validation") & (split_df["label"] == 1)]))
            },
            "test": {
                "real": int(len(split_df[(split_df["split"] == "test") & (split_df["label"] == 0)])),
                "synthetic": int(len(split_df[(split_df["split"] == "test") & (split_df["label"] == 1)]))
            }
        },
        "generator_holdout": {
            "enabled": holdout_enabled,
            "held_out_generators": list(held_out_generators)
        },
        "leakage_free": True
    }
    with open(METADATA_DIR / "splits_summary.json", "w", encoding="utf-8") as f:
        json.dump(splits_summary, f, indent=2)

    # Save split distribution report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    split_gen_df = split_df.groupby(["split", "generator", "label_str"]).size().reset_index(name="count")
    split_gen_df.to_csv(REPORTS_DIR / "split_distribution.csv", index=False)

    print(f"\nSplit generation complete! Files saved in {SPLITS_DIR}")
    return split_df

if __name__ == "__main__":
    generate_multi_splits()
