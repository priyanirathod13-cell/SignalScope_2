"""
SignalScope Dataset Downloader
Downloads the GenImage benchmark subset from Hugging Face and organizes
raw images into data/raw/real and data/raw/synthetic with generator provenance.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_REAL = RAW_DIR / "real"
RAW_SYNTHETIC = RAW_DIR / "synthetic"
HF_REPO = "https://huggingface.co/datasets/jhutter2/281_Genimage"

GENERATOR_DIRS = {
    "adm": {
        "synthetic": "adm/ai",
        "real": "adm/nature",
        "gen_label": "adm"
    },
    "stable_diffusion_v1_5": {
        "synthetic": "stable_diffusion_v_1_5/ai",
        "real": "stable_diffusion_v_1_5/nature",
        "gen_label": "stable_diffusion_v1_5"
    },
    "wukong": {
        "synthetic": "wukong/ai",
        "real": "wukong/nature",
        "gen_label": "wukong"
    }
}

def setup_raw_directories():
    RAW_REAL.mkdir(parents=True, exist_ok=True)
    RAW_SYNTHETIC.mkdir(parents=True, exist_ok=True)

def populate_from_source(source_dir: Path):
    """
    Populates data/raw/real and data/raw/synthetic from a downloaded clone of jhutter2/281_Genimage.
    """
    setup_raw_directories()
    copied_real = 0
    copied_synth = 0

    print("Populating raw images with generator namespaces...")
    for gen_key, info in GENERATOR_DIRS.items():
        # Synthetic images
        synth_src = source_dir / info["synthetic"]
        if synth_src.exists():
            for img_file in synth_src.glob("*.*"):
                if img_file.suffix.lower() in [".png", ".jpg", ".jpeg"] and img_file.stat().st_size > 0:
                    try:
                        with Image.open(img_file) as im:
                            im.verify()
                        target_name = f"synthetic_{info['gen_label']}_{img_file.name}"
                        target_path = RAW_SYNTHETIC / target_name
                        if not target_path.exists():
                            shutil.copy2(img_file, target_path)
                        copied_synth += 1
                    except Exception:
                        pass

        # Real images
        real_src = source_dir / info["real"]
        if real_src.exists():
            for img_file in real_src.glob("*.*"):
                if img_file.suffix.lower() in [".png", ".jpg", ".jpeg"] and img_file.stat().st_size > 0:
                    try:
                        with Image.open(img_file) as im:
                            im.verify()
                        target_name = f"real_imagenet_{info['gen_label']}_{img_file.name}"
                        target_path = RAW_REAL / target_name
                        if not target_path.exists():
                            shutil.copy2(img_file, target_path)
                        copied_real += 1
                    except Exception:
                        pass

    print(f"Total raw images populated:")
    print(f"  Real images:      {copied_real} -> {RAW_REAL}")
    print(f"  Synthetic images: {copied_synth} -> {RAW_SYNTHETIC}")
    return copied_real, copied_synth

def main():
    source_clone = PROJECT_ROOT / "test_clone"
    if source_clone.exists():
        populate_from_source(source_clone)
    else:
        print(f"Source directory {source_clone} not found. Please clone {HF_REPO} with Git LFS.")

if __name__ == "__main__":
    main()
