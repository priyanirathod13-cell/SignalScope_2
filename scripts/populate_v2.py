import os, shutil
from pathlib import Path

v2_dir = Path('data/v2')
v2_real = v2_dir / 'real'
v2_synth = v2_dir / 'synthetic'
v2_meta = v2_dir / 'metadata'
v2_splits = v2_dir / 'splits'
v2_reports = v2_dir / 'reports'

v2_real.mkdir(parents=True, exist_ok=True)
v2_synth.mkdir(parents=True, exist_ok=True)
v2_meta.mkdir(parents=True, exist_ok=True)
v2_splits.mkdir(parents=True, exist_ok=True)
v2_reports.mkdir(parents=True, exist_ok=True)

# 1. Hardlink real images
for r_path in [Path('data/raw/real'), Path('data/raw/cifake/real')]:
    for f in r_path.glob('*.*'):
        dst = v2_real / f.name
        if not dst.exists():
            os.link(f, dst)
real_count = len(list(v2_real.glob('*.*')))
print(f'Populated data/v2/real: {real_count} images')

# 2. Hardlink synthetic images
for s_path in [Path('data/raw/synthetic'), Path('data/raw/cifake/synthetic')]:
    for f in s_path.glob('*.*'):
        dst = v2_synth / f.name
        if not dst.exists():
            os.link(f, dst)
synth_count = len(list(v2_synth.glob('*.*')))
print(f'Populated data/v2/synthetic: {synth_count} images')

# 3. Copy metadata
for f in Path('data/metadata').glob('*.*'):
    if f.name in ['images.csv', 'datasets.csv', 'generators.csv', 'splits_summary.json']:
        shutil.copy2(f, v2_meta / f.name)
        print(f'Copied {f.name} -> data/v2/metadata/')

# 4. Copy split CSVs and directories
for f in Path('data/splits').glob('*.csv'):
    shutil.copy2(f, v2_splits / f.name)
    print(f'Copied {f.name} -> data/v2/splits/')

for s in ['train', 'validation', 'test']:
    for lbl in ['real', 'synthetic']:
        src_folder = Path(f'data/splits/{s}/{lbl}')
        dst_folder = v2_splits / s / lbl
        dst_folder.mkdir(parents=True, exist_ok=True)
        for img in src_folder.glob('*.*'):
            target_img = dst_folder / img.name
            if not target_img.exists():
                os.link(img, target_img)
    total_split = len(list((v2_splits / s / 'real').glob('*.*'))) + len(list((v2_splits / s / 'synthetic').glob('*.*')))
    print(f'Populated data/v2/splits/{s}: {total_split} images')

# 5. Copy reports
for f in Path('reports/dataset').glob('*.*'):
    shutil.copy2(f, v2_reports / f.name)
print('Copied reports -> data/v2/reports/')
