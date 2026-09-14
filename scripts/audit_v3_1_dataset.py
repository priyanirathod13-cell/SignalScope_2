import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(r"C:\SignalScope")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def audit_v3_1():
    meta_path = PROJECT_ROOT / "data" / "v3_1" / "metadata" / "images.csv"
    if not meta_path.exists():
        print(f"Error: {meta_path} does not exist yet.")
        return False
        
    df = pd.read_csv(meta_path)
    print("=" * 70)
    print("SIGNALSCOPE V3.1 DATASET STATISTICAL AUDIT")
    print("=" * 70)
    print(f"Total Samples: {len(df):,}")
    
    # 1. Class Balance
    print("\n--- 1. CLASS BALANCE ---")
    class_counts = df["label_str"].value_counts()
    for lbl, count in class_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {lbl.upper():12s}: {count:6d} ({pct:5.2f}%)")
    class_imbalance = abs(class_counts.get("real", 0) - class_counts.get("synthetic", 0)) / len(df)
    print(f"  Class Imbalance Ratio: {class_imbalance*100:.2f}% (Target: < 5%)")
    
    # 2. Aspect Ratio Balance & Crosstab
    print("\n--- 2. ASPECT RATIO BY CLASS ---")
    ct = pd.crosstab(df["label_str"], df["aspect_ratio"], normalize="index") * 100
    print(ct.round(2))
    
    # Absolute counts
    ct_counts = pd.crosstab(df["label_str"], df["aspect_ratio"], margins=True)
    print("\nAbsolute Counts:")
    print(ct_counts)
    
    # 3. Padding Frequency by Class
    # Non-square images require padding when letterboxed into a square (224x224)
    print("\n--- 3. PADDING FREQUENCY BY CLASS ---")
    df["requires_padding"] = df["aspect_ratio"] != "square"
    pad_ct = pd.crosstab(df["label_str"], df["requires_padding"], normalize="index") * 100
    pad_ct.columns = ["No Padding (Square)", "Requires Padding (Non-Square)"]
    print(pad_ct.round(2))
    
    pad_real = pad_ct.loc["real", "Requires Padding (Non-Square)"]
    pad_synth = pad_ct.loc["synthetic", "Requires Padding (Non-Square)"]
    pad_diff = abs(pad_real - pad_synth)
    print(f"\nPadding Frequency Difference (Real vs. Synthetic): {pad_diff:.2f}% (Target: < 5%)")
    if pad_diff < 5.0:
        print("  >> PADDING SHORTCUT SUCCESSFULLY ELIMINATED! <<")
    else:
        print("  >> WARNING: PADDING BIAS DETECTED! <<")
        
    # 4. Split Distribution & Zero Leakage Check
    print("\n--- 4. SPLIT DISTRIBUTION & LEAKAGE CHECK ---")
    split_counts = df["split"].value_counts()
    for sp, count in split_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {sp.upper():8s}: {count:6d} ({pct:5.2f}%)")
        
    # SHA-256 collision check across splits
    train_hashes = set(df[df["split"] == "train"]["sha256"])
    val_hashes = set(df[df["split"] == "val"]["sha256"])
    test_hashes = set(df[df["split"] == "test"]["sha256"])
    
    train_val_leak = train_hashes.intersection(val_hashes)
    train_test_leak = train_hashes.intersection(test_hashes)
    val_test_leak = val_hashes.intersection(test_hashes)
    
    print(f"  Train-Val SHA256 Overlap:  {len(train_val_leak)}")
    print(f"  Train-Test SHA256 Overlap: {len(train_test_leak)}")
    print(f"  Val-Test SHA256 Overlap:   {len(val_test_leak)}")
    assert len(train_val_leak) == 0, "Train-Val SHA256 leakage detected!"
    assert len(train_test_leak) == 0, "Train-Test SHA256 leakage detected!"
    assert len(val_test_leak) == 0, "Val-Test SHA256 leakage detected!"
    
    # Base group cross-split leakage check
    train_groups = set(df[df["split"] == "train"]["base_group"])
    val_groups = set(df[df["split"] == "val"]["base_group"])
    test_groups = set(df[df["split"] == "test"]["base_group"])
    
    grp_tv = train_groups.intersection(val_groups)
    grp_tt = train_groups.intersection(test_groups)
    grp_vt = val_groups.intersection(test_groups)
    
    print(f"  Train-Val Group Overlap:   {len(grp_tv)}")
    print(f"  Train-Test Group Overlap:  {len(grp_tt)}")
    print(f"  Val-Test Group Overlap:    {len(grp_vt)}")
    assert len(grp_tv) == 0, "Train-Val group leakage detected!"
    assert len(grp_tt) == 0, "Train-Test group leakage detected!"
    assert len(grp_vt) == 0, "Val-Test group leakage detected!"
    print("  >> ZERO DATA LEAKAGE CONFIRMED (100% CLEAN GROUP & HASH SPLITS) <<")
    
    # 5. Generator Distribution
    print("\n--- 5. GENERATOR DISTRIBUTION ---")
    gen_df = df.groupby(["split", "generator", "label_str"]).size().unstack(fill_value=0)
    print(gen_df)
    
    # 6. Night Sky vs Cosmic Fantasy Parity
    print("\n--- 6. NIGHT / ASTROPHOTOGRAPHY PARITY ---")
    astro_real = (df["generator"] == "real_night_sky").sum()
    cosmic_synth = (df["generator"] == "synth_cosmic_fantasy").sum()
    print(f"  Real Night/Astro Samples:   {astro_real}")
    print(f"  Synthetic Cosmic Samples:   {cosmic_synth}")
    print(f"  Ratio: {astro_real / max(1, cosmic_synth):.2f}:1.00 (Target: 1.00:1.00)")
    
    return True

if __name__ == "__main__":
    audit_v3_1()
