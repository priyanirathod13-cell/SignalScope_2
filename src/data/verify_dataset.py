"""
SignalScope Dataset Health & Verification Suite
Verifies directory structure, image counts, class & generator balance,
file readability via PIL, and strictly asserts zero split leakage.
"""

import sys
import json
import hashlib
from pathlib import Path
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"
METADATA_DIR = DATA_DIR / "metadata"

def check_structure():
    required_dirs = [
        RAW_DIR / "real",
        RAW_DIR / "synthetic",
        DATA_DIR / "processed",
        SPLITS_DIR / "train" / "real",
        SPLITS_DIR / "train" / "synthetic",
        SPLITS_DIR / "validation" / "real",
        SPLITS_DIR / "validation" / "synthetic",
        SPLITS_DIR / "test" / "real",
        SPLITS_DIR / "test" / "synthetic",
        METADATA_DIR
    ]
    all_exist = True
    print("Checking directory layout...")
    for d in required_dirs:
        rel = d.relative_to(PROJECT_ROOT)
        if not d.exists():
            print(f"  [MISSING] {rel}")
            all_exist = False
        else:
            print(f"  [OK]      {rel}")
    return all_exist

def verify_images():
    print("\nVerifying image file integrity and readability...")
    corrupt_files = []
    total_checked = 0

    for split in ["train", "validation", "test"]:
        for label in ["real", "synthetic"]:
            folder = SPLITS_DIR / split / label
            for img_p in folder.glob("*.*"):
                if img_p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    total_checked += 1
                    try:
                        with Image.open(img_p) as img:
                            img.verify()
                    except Exception as e:
                        corrupt_files.append((str(img_p), str(e)))

    print(f"Total images checked: {total_checked}")
    if corrupt_files:
        print(f"[FAIL] Found {len(corrupt_files)} corrupt images:")
        for cf, err in corrupt_files[:5]:
            print(f"  {cf}: {err}")
        return False
    else:
        print("[PASS] 100% of images verified valid and uncorrupted.")
        return True

def verify_anti_leakage():
    print("\nChecking for split leakage (disjointness check)...")
    split_hashes = {"train": set(), "validation": set(), "test": set()}

    for split in ["train", "validation", "test"]:
        for label in ["real", "synthetic"]:
            folder = SPLITS_DIR / split / label
            for img_p in folder.glob("*.*"):
                if img_p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    with open(img_p, "rb") as f:
                        h = hashlib.sha256(f.read()).hexdigest()
                        split_hashes[split].add(h)

    train_val_overlap = split_hashes["train"].intersection(split_hashes["validation"])
    train_test_overlap = split_hashes["train"].intersection(split_hashes["test"])
    val_test_overlap = split_hashes["validation"].intersection(split_hashes["test"])

    leaks = len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)
    if leaks > 0:
        print(f"[FAIL] Split leakage detected! Overlaps: Train-Val={len(train_val_overlap)}, Train-Test={len(train_test_overlap)}, Val-Test={len(val_test_overlap)}")
        return False
    else:
        print(f"[PASS] Zero split leakage verified! All partitions are mutually disjoint.")
        print(f"  Train hashes:      {len(split_hashes['train'])}")
        print(f"  Validation hashes: {len(split_hashes['validation'])}")
        print(f"  Test hashes:       {len(split_hashes['test'])}")
        return True

def print_summary():
    inv_path = METADATA_DIR / "dataset_inventory.csv"
    if not inv_path.exists():
        print(f"Metadata inventory {inv_path} not found.")
        return

    df = pd.read_csv(inv_path)
    print("\n" + "="*50)
    print("          SIGNALSCOPE DATASET AUDIT SUMMARY")
    print("="*50)
    print(f"Total Dataset Images: {len(df)}")
    print("\nClass Distribution:")
    for label, count in df["label"].value_counts().items():
        pct = (count / len(df)) * 100
        print(f"  - {label.capitalize():12s}: {count:5d} ({pct:.1f}%)")

    print("\nGenerator Distribution:")
    for gen, count in df["generator"].value_counts().items():
        pct = (count / len(df)) * 100
        print(f"  - {gen:20s}: {count:5d} ({pct:.1f}%)")

    print("\nSplit Distribution:")
    for split in ["train", "validation", "test"]:
        count = (df["split"] == split).sum()
        pct = (count / len(df)) * 100
        print(f"  - {split.capitalize():12s}: {count:5d} ({pct:.1f}%)")

    print("\nStratified Matrix (Split x Label):")
    print(pd.crosstab(df["split"], df["label"], margins=True))

    print("\nStratified Matrix (Split x Generator):")
    print(pd.crosstab(df["split"], df["generator"], margins=True))
    print("="*50)

def main():
    ok_struct = check_structure()
    ok_images = verify_images()
    ok_leaks = verify_anti_leakage()
    print_summary()

    if ok_struct and ok_images and ok_leaks:
        print("\n>>> [SUCCESS] All dataset validation checks passed! <<<")
        sys.exit(0)
    else:
        print("\n>>> [FAILURE] One or more validation checks failed! <<<")
        sys.exit(1)

if __name__ == "__main__":
    main()
