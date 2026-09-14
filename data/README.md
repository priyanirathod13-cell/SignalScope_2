# SignalScope Dataset Documentation

## 1. Multi-Dataset Overview & Provenance
SignalScope trains on a diverse, multi-benchmark collection of authentic photography and synthetic generative imagery:

1. **GenImage Benchmark**:
   * **Source**: Hugging Face [`jhutter2/281_Genimage`](https://huggingface.co/datasets/jhutter2/281_Genimage) (NeurIPS 2023).
   * **License**: MIT (Subset) / CC-BY-NC-SA 4.0 (Original Benchmark).
   * **Generators**: Stable Diffusion v1.5, ADM (Ablated Diffusion Model), Wukong Diffusion, and ImageNet-1K Real photography.

2. **CIFAKE Benchmark**:
   * **Source**: Hugging Face [`dragonintelligence/CIFAKE-image-dataset`](https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset) (Bird & Lotfi, IEEE Access 2023).
   * **License**: CC-BY 4.0 / Apache-2.0.
   * **Generators**: Stable Diffusion v1.4 and CIFAR-10 Real photography (vehicles, animals, airplanes, ships).

---

## 2. Sample Inventory & Class Distribution

| Class | Count | Percentage | Description |
| :--- | :---: | :---: | :--- |
| **Real Photography (0)** | **10,842** | **49.86%** | Authentic camera captures from ImageNet-1K (nature, objects) and CIFAR-10 (vehicles, animals) |
| **AI-Generated / Synthetic (1)** | **10,901** | **50.14%** | Diffusion synthesis from SD v1.5, SD v1.4, ADM, and Wukong |
| **Total Verified Images** | **21,743** | **100.0%** | Cryptographically validated, deduplicated, 100% readable images |

### Generator & Architecture Breakdown
* **`real_cifar10`**: 8,000 images (36.79%) — Camera photography (CIFAR-10)
* **`real_imagenet`**: 2,842 images (13.07%) — Camera photography (ImageNet-1K)
* **`stable_diffusion_v1_4`**: 7,979 images (36.70%) — Latent diffusion text-to-image
* **`stable_diffusion_v1_5`**: 996 images (4.58%) — Latent diffusion text-to-image
* **`adm`**: 968 images (4.45%) — Guided pixel-space diffusion
* **`wukong`**: 958 images (4.41%) — Large-scale multilingual diffusion

---

## 3. Data Cleansing, Deduplication & Splitting
1. **Pillow Validation**: Every image verified for readability, valid dimensions, and RGB channel compatibility.
2. **SHA-256 Deduplication**: 42 duplicate entries identified; 21 redundant images purged. Zero internal duplicates remain.
3. **Cross-Split Isolation**: Zero overlapping SHA-256 hashes between `train`, `validation`, and `test`.
4. **Partitioning Breakdown**:
   * **Train Split (70%)**: **15,220** images (7,589 Real, 7,631 Synthetic)
   * **Validation Split (15%)**: **3,261** images (1,626 Real, 1,635 Synthetic)
   * **Test Split (15%)**: **3,262** images (1,627 Real, 1,635 Synthetic)
5. **Generator Holdout Support**: Configurable in `config/dataset_config.json` for zero-shot unseen generator evaluations.

---

## 4. How to Reproduce the Dataset Pipeline
From the project root:
```powershell
# 1. Run multi-dataset ingestion & deduplication
python -m src.data.multi_dataset

# 2. Generate leakage-safe stratified 70/15/15 splits
python -m src.data.make_splits

# 3. Run complete verification and audit report generator
python -m src.data.verify_dataset
```

---

## 5. Official SIH Evaluation Guardrail
> [!IMPORTANT]
> **The official Smart India Hackathon (SIH) held-out test set remains strictly untouched and unaccessed.**
> All dataset ingestion and evaluation in this repository use public benchmark datasets only. The official held-out test set is reserved exclusively for jury scoring.
