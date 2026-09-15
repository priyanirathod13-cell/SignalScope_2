from pathlib import Path
import sys

# Add project root to Python path
PROJECT_DIR = Path(r"D:\SignalScope")
sys.path.insert(0, str(PROJECT_DIR))

import matplotlib.pyplot as plt
from PIL import Image

from model.predict import predict_image


# ============================================================
# Configuration
# ============================================================



AI_FOLDER = (
    PROJECT_DIR
    / "data"
    / "external_test"
    / "ai"
)

REAL_FOLDER = (
    PROJECT_DIR
    / "data"
    / "external_test"
    / "real"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "evaluation"
    / "external_error_analysis_v3.png"
)


# ============================================================
# Find wrong predictions
# ============================================================

errors = []


def check_folder(folder, expected_label):

    for extension in [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.webp"
    ]:

        for image_path in folder.glob(extension):

            result = predict_image(image_path)

            prediction = result["label"]

            if prediction != expected_label:

                errors.append({
                    "path": image_path,
                    "expected": expected_label,
                    "predicted": prediction,
                    "confidence": result["confidence"]
                })


# ============================================================
# Analyze
# ============================================================

print("Analyzing external AI images...")
check_folder(AI_FOLDER, "AI")

print("Analyzing external Real images...")
check_folder(REAL_FOLDER, "Real")


print()
print("=" * 70)
print("V3 EXTERNAL ERROR ANALYSIS")
print("=" * 70)

print(
    f"Total errors found: {len(errors)}"
)


# ============================================================
# Sort by confidence
# Highest-confidence mistakes first
# ============================================================

errors.sort(
    key=lambda x: x["confidence"],
    reverse=True
)


for error in errors:

    print()
    print(
        f"{error['path'].name}"
    )

    print(
        f"Expected : {error['expected']}"
    )

    print(
        f"Predicted: {error['predicted']}"
    )

    print(
        f"Confidence: "
        f"{error['confidence']:.2f}%"
    )


# ============================================================
# Create visualization
# ============================================================

if not errors:

    print("\nNo errors found!")
    exit()


MAX_IMAGES = min(
    len(errors),
    20
)

selected_errors = errors[:MAX_IMAGES]

columns = 4
rows = (
    MAX_IMAGES + columns - 1
) // columns


fig = plt.figure(
    figsize=(16, 4 * rows)
)


for index, error in enumerate(
    selected_errors,
    start=1
):

    image = Image.open(
        error["path"]
    ).convert("RGB")

    ax = fig.add_subplot(
        rows,
        columns,
        index
    )

    ax.imshow(image)

    ax.axis("off")

    ax.set_title(
        f"{error['path'].name}\n"
        f"Expected: {error['expected']} | "
        f"Predicted: {error['predicted']}\n"
        f"Confidence: {error['confidence']:.1f}%"
    )


plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print()
print(
    f"Error analysis saved to:\n"
    f"{OUTPUT_PATH}"
)