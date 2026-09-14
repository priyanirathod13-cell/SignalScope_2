"""
SignalScope V3.1 Dataset Preparation Pipeline
Builds a scientifically balanced, leakage-safe dataset specifically eliminating:
1. Aspect-Ratio / Letterbox Padding Shortcut:
   Symmetrically balances square, landscape (4:3), and portrait (3:4)
   aspect ratios across BOTH Real and Synthetic classes (0.0% padding difference).
2. Semantic Night/Astrophotography Shortcut:
   Balances real astrophotography and synthetic cosmic scenes exactly 1:1.
3. Strict Unseen Holdout:
   Wukong generator is 100% held out and never used for training or validation.
4. Diagnostic Failure Images:
   User failure images (WhatsApp 02.56.xx) are excluded from training/validation.
"""

import os
import sys
import shutil
from pathlib import Path
import hashlib
import json
import random
from collections import Counter
from PIL import Image
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(r"C:\SignalScope")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def classify_aspect_ratio(w, h):
    ratio = w / h
    if 0.95 <= ratio <= 1.05:
        return "square"
    elif ratio > 1.05:
        return "landscape"
    else:
        return "portrait"

def generate_crops(img, base_id, out_dir, prefix, label, gen, source):
    """
    Generates exactly 3 balanced aspect ratio variants (1 square, 1 landscape, 1 portrait)
    from a source image and saves them to out_dir.
    All 3 variants share the exact same base_group: f"{prefix}_{base_id}".
    """
    w, h = img.size
    records = []
    base_group = f"{prefix}_{base_id}"
    
    ratio = w / h
    if 0.95 <= ratio <= 1.05:
        # Original is square
        orig_p = out_dir / f"{prefix}_{base_id}_sq.jpg"
        img.save(orig_p, "JPEG", quality=95)
        records.append({
            "image_path": str(orig_p),
            "label": label,
            "label_str": "real" if label == 0 else "synthetic",
            "generator": gen,
            "source_dataset": source,
            "base_group": base_group,
            "sha256": get_sha256(orig_p),
            "aspect_ratio": "square",
            "width": w,
            "height": h
        })
        
        # Landscape 4:3 crop (w x 0.75w)
        lh = max(32, int(w * 0.75))
        top_l = max(0, (h - lh) // 2)
        img_l = img.crop((0, top_l, w, top_l + lh))
        p_l = out_dir / f"{prefix}_{base_id}_land.jpg"
        img_l.save(p_l, "JPEG", quality=95)
        records.append({
            "image_path": str(p_l),
            "label": label,
            "label_str": "real" if label == 0 else "synthetic",
            "generator": gen,
            "source_dataset": source,
            "base_group": base_group,
            "sha256": get_sha256(p_l),
            "aspect_ratio": "landscape",
            "width": w,
            "height": lh
        })
        
        # Portrait 3:4 crop (0.75h x h)
        pw = max(32, int(h * 0.75))
        left_p = max(0, (w - pw) // 2)
        img_p = img.crop((left_p, 0, left_p + pw, h))
        p_p = out_dir / f"{prefix}_{base_id}_port.jpg"
        img_p.save(p_p, "JPEG", quality=95)
        records.append({
            "image_path": str(p_p),
            "label": label,
            "label_str": "real" if label == 0 else "synthetic",
            "generator": gen,
            "source_dataset": source,
            "base_group": base_group,
            "sha256": get_sha256(p_p),
            "aspect_ratio": "portrait",
            "width": pw,
            "height": h
        })
        
    elif ratio > 1.05:
        # Original is landscape
        orig_p = out_dir / f"{prefix}_{base_id}_land.jpg"
        img.save(orig_p, "JPEG", quality=95)
        records.append({
            "image_path": str(orig_p),
            "label": label,
            "label_str": "real" if label == 0 else "synthetic",
            "generator": gen,
            "source_dataset": source,
            "base_group": base_group,
            "sha256": get_sha256(orig_p),
            "aspect_ratio": "landscape",
            "width": w,
            "height": h
        })
        
        # Square crop (h x h)
        left_s = max(0, (w - h) // 2)
        img_s = img.crop((left_s, 0, left_s + h, h))
        p_s = out_dir / f"{prefix}_{base_id}_sq.jpg"
        img_s.save(p_s, "JPEG", quality=95)
        records.append({
            "image_path": str(p_s),
            "label": label,
            "label_str": "real" if label == 0 else "synthetic",
            "generator": gen,
            "source_dataset": source,
            "base_group": base_group,
            "sha256": get_sha256(p_s),
            "aspect_ratio": "square",
            "width": h,
            "height": h
        })
        
        # Portrait crop (0.75h x h)
        pw = max(32, int(h * 0.75))
        left_p = max(0, (w - pw) // 2)
        img_p = img.crop((left_p, 0, left_p + pw, h))
        p_p = out_dir / f"{prefix}_{base_id}_port.jpg"
        img_p.save(p_p, "JPEG", quality=95)
        records.append({
            "image_path": str(p_p),
            "label": label,
            "label_str": "real" if label == 0 else "synthetic",
            "generator": gen,
            "source_dataset": source,
            "base_group": base_group,
            "sha256": get_sha256(p_p),
            "aspect_ratio": "portrait",
            "width": pw,
            "height": h
        })
        
    else:
        # Original is portrait
        orig_p = out_dir / f"{prefix}_{base_id}_port.jpg"
        img.save(orig_p, "JPEG", quality=95)
        records.append({
            "image_path": str(orig_p),
            "label": label,
            "label_str": "real" if label == 0 else "synthetic",
            "generator": gen,
            "source_dataset": source,
            "base_group": base_group,
            "sha256": get_sha256(orig_p),
            "aspect_ratio": "portrait",
            "width": w,
            "height": h
        })
        
        # Square crop (w x w)
        top_s = max(0, (h - w) // 2)
        img_s = img.crop((0, top_s, w, top_s + w))
        p_s = out_dir / f"{prefix}_{base_id}_sq.jpg"
        img_s.save(p_s, "JPEG", quality=95)
        records.append({
            "image_path": str(p_s),
            "label": label,
            "label_str": "real" if label == 0 else "synthetic",
            "generator": gen,
            "source_dataset": source,
            "base_group": base_group,
            "sha256": get_sha256(p_s),
            "aspect_ratio": "square",
            "width": w,
            "height": w
        })
        
        # Landscape crop (w x 0.75w)
        lh = max(32, int(w * 0.75))
        top_l = max(0, (h - lh) // 2)
        img_l = img.crop((0, top_l, w, top_l + lh))
        p_l = out_dir / f"{prefix}_{base_id}_land.jpg"
        img_l.save(p_l, "JPEG", quality=95)
        records.append({
            "image_path": str(p_l),
            "label": label,
            "label_str": "real" if label == 0 else "synthetic",
            "generator": gen,
            "source_dataset": source,
            "base_group": base_group,
            "sha256": get_sha256(p_l),
            "aspect_ratio": "landscape",
            "width": w,
            "height": lh
        })
        
    return records

def build_v3_1_dataset():
    print("[V3.1 Dataset Builder] Initializing...")
    
    v3_1_dir = PROJECT_ROOT / "data" / "v3_1"
    real_out = v3_1_dir / "real"
    synth_out = v3_1_dir / "synthetic"
    meta_dir = v3_1_dir / "metadata"
    splits_dir = v3_1_dir / "splits"
    
    # Re-initialize clean directories
    for d in [real_out, synth_out, meta_dir, splits_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        
    all_records = []
    
    # =========================================================================
    # 1. Real ImageNet (Diverse photographic subjects, portraits, nature, objects)
    # Target: 1,600 base images * 3 crops = 4,800 samples
    # =========================================================================
    print("Processing Real ImageNet (1,600 base images -> 4,800 crops)...")
    raw_real = PROJECT_ROOT / "data" / "raw" / "real"
    real_imagenet_files = sorted(list(raw_real.glob("*.*")))[:1600]
    for i, fp in enumerate(real_imagenet_files):
        try:
            with Image.open(fp) as im:
                im_rgb = im.convert("RGB")
                recs = generate_crops(im_rgb, f"imagenet_{i:04d}", real_out, "real", 0, "real_imagenet", "ImageNet")
                all_records.extend(recs)
        except Exception:
            pass
            
    # =========================================================================
    # 2. Real Astrophotography & Night Sky
    # Target: 150 base images * 3 crops = 450 samples (1:1 with Cosmic Fantasy)
    # =========================================================================
    print("Processing Real Astrophotography (150 base images -> 450 crops)...")
    v3_real = PROJECT_ROOT / "data" / "v3" / "real"
    astro_files = sorted(list(v3_real.glob("*astro*.*")))[:150]
    for i, fp in enumerate(astro_files):
        try:
            with Image.open(fp) as im:
                im_rgb = im.convert("RGB")
                recs = generate_crops(im_rgb, f"astro_{i:04d}", real_out, "real", 0, "real_night_sky", "Astrophotography")
                all_records.extend(recs)
        except Exception:
            pass
            
    # =========================================================================
    # 3. Synthetic Stable Diffusion v1.5 (High-res portraits, landscapes, architecture)
    # Target: 800 base images * 3 crops = 2,400 samples
    # =========================================================================
    print("Processing Synthetic Stable Diffusion v1.5 (800 base images -> 2,400 crops)...")
    raw_synth = PROJECT_ROOT / "data" / "raw" / "synthetic"
    sd15_files = sorted(list(raw_synth.glob("synthetic_stable_*.*")))[:800]
    for i, fp in enumerate(sd15_files):
        try:
            with Image.open(fp) as im:
                im_rgb = im.convert("RGB")
                recs = generate_crops(im_rgb, f"sd15_{i:04d}", synth_out, "synth", 1, "stable_diffusion_v1_5", "GenImage")
                all_records.extend(recs)
        except Exception:
            pass
            
    # =========================================================================
    # 4. Synthetic ADM Guided Diffusion (Diverse subjects, textures, lighting)
    # Target: 800 base images * 3 crops = 2,400 samples
    # =========================================================================
    print("Processing Synthetic ADM (800 base images -> 2,400 crops)...")
    adm_files = sorted(list(raw_synth.glob("synthetic_adm_*.*")))[:800]
    for i, fp in enumerate(adm_files):
        try:
            with Image.open(fp) as im:
                im_rgb = im.convert("RGB")
                recs = generate_crops(im_rgb, f"adm_{i:04d}", synth_out, "synth", 1, "adm", "GenImage")
                all_records.extend(recs)
        except Exception:
            pass
            
    # =========================================================================
    # 5. Synthetic Cosmic Fantasy (Balanced 1:1 with real astrophotography)
    # Target: 150 base images * 3 crops = 450 samples
    # =========================================================================
    print("Processing Synthetic Cosmic Fantasy (150 base images -> 450 crops)...")
    v3_synth = PROJECT_ROOT / "data" / "v3" / "synthetic"
    cosmic_files = sorted(list(v3_synth.glob("*cosmic*.*")))[:150]
    for i, fp in enumerate(cosmic_files):
        try:
            with Image.open(fp) as im:
                im_rgb = im.convert("RGB")
                recs = generate_crops(im_rgb, f"cosmic_{i:04d}", synth_out, "synth", 1, "synth_cosmic_fantasy", "FantasyDiffusion")
                all_records.extend(recs)
        except Exception:
            pass
            
    df = pd.DataFrame(all_records)
    print(f"\nTotal collected samples: {len(df)}")
    print("By Label:\n", df["label_str"].value_counts())
    print("\nBy Aspect Ratio:\n", pd.crosstab(df["label_str"], df["aspect_ratio"], margins=True))
    
    # =========================================================================
    # 6. Leakage-Safe Group Partitioning (70% Train, 15% Val, 15% Test)
    # Groups are unique source base_groups (e.g. real_imagenet_0042)
    # All 3 crops of any image stay together in the EXACT SAME split.
    # =========================================================================
    print("\nPartitioning into Train / Val / Test (70% / 15% / 15%) with Group-Aware Splitting...")
    unique_groups = df[["base_group", "label_str", "generator"]].drop_duplicates()
    
    splits = {}
    for (l_str, gen), g_df in unique_groups.groupby(["label_str", "generator"]):
        groups = g_df["base_group"].tolist()
        random.shuffle(groups)
        n = len(groups)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        
        train_g = set(groups[:n_train])
        val_g = set(groups[n_train:n_train + n_val])
        test_g = set(groups[n_train + n_val:])
        
        for g in train_g:
            splits[g] = "train"
        for g in val_g:
            splits[g] = "val"
        for g in test_g:
            splits[g] = "test"
            
    df["split"] = df["base_group"].map(splits)
    
    # Save master metadata
    meta_path = meta_dir / "images.csv"
    df.to_csv(meta_path, index=False)
    print(f"Saved master metadata: {meta_path}")
    
    # Save split CSVs
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)
    
    print(f"Splits saved: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return df

if __name__ == "__main__":
    build_v3_1_dataset()
