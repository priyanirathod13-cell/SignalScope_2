from pathlib import Path

from model.predict import predict_image


# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = Path(r"D:\SignalScope")

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


# ============================================================
# Get images
# ============================================================

def get_images(folder):

    images = []

    for extension in [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.webp"
    ]:
        images.extend(
            folder.glob(extension)
        )

    return sorted(images)


# ============================================================
# Test a folder
# ============================================================

def test_folder(
    folder,
    expected_label
):

    images = get_images(folder)

    if not images:
        raise FileNotFoundError(
            f"No images found in {folder}"
        )

    correct = 0

    total = len(images)

    confidence_values = []

    print()
    print("=" * 70)

    print(
        f"Testing: {expected_label} images"
    )

    print(
        f"Images: {total}"
    )

    print("=" * 70)

    for index, image_path in enumerate(
        images,
        start=1
    ):

        result = predict_image(
            image_path
        )

        prediction = result["label"]

        confidence = result["confidence"]

        is_correct = (
            prediction == expected_label
        )

        if is_correct:

            correct += 1

            status = "CORRECT"

        else:

            status = "WRONG"

        confidence_values.append(
            confidence
        )

        print()

        print(
            f"{index:02d}. "
            f"{image_path.name}"
        )

        print(
            f"    Prediction      : "
            f"{prediction}"
        )

        print(
            f"    Confidence      : "
            f"{confidence:.2f}%"
        )

        print(
            f"    AI probability  : "
            f"{result['ai_probability']:.2f}%"
        )

        print(
            f"    Real probability: "
            f"{result['real_probability']:.2f}%"
        )

        print(
            f"    Result          : "
            f"{status}"
        )

    accuracy = (
        correct / total
    ) * 100

    average_confidence = (
        sum(confidence_values)
        / len(confidence_values)
    )

    print()
    print("-" * 70)

    print(
        f"Correct: "
        f"{correct}/{total}"
    )

    print(
        f"Accuracy: "
        f"{accuracy:.2f}%"
    )

    print(
        f"Average confidence: "
        f"{average_confidence:.2f}%"
    )

    print("-" * 70)

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy
    }


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SignalScope V3 - External Generalization Test")
    print("=" * 70)

    print()
    print("AI folder:")
    print(AI_FOLDER)

    print()
    print("Real folder:")
    print(REAL_FOLDER)

    # --------------------------------------------------------
    # Test AI images
    # --------------------------------------------------------

    ai_results = test_folder(
        AI_FOLDER,
        "AI"
    )

    # --------------------------------------------------------
    # Test Real images
    # --------------------------------------------------------

    real_results = test_folder(
        REAL_FOLDER,
        "Real"
    )

    # --------------------------------------------------------
    # Overall results
    # --------------------------------------------------------

    total = (
        ai_results["total"]
        + real_results["total"]
    )

    correct = (
        ai_results["correct"]
        + real_results["correct"]
    )

    overall_accuracy = (
        correct / total
    ) * 100

    # --------------------------------------------------------
    # False positive rate
    #
    # False positive =
    # Real image predicted as AI
    # --------------------------------------------------------

    real_false_positives = (
        real_results["total"]
        - real_results["correct"]
    )

    false_positive_rate = (
        real_false_positives
        / real_results["total"]
    ) * 100

    # --------------------------------------------------------
    # AI detection rate
    # --------------------------------------------------------

    ai_detection_rate = (
        ai_results["correct"]
        / ai_results["total"]
    ) * 100

    print()
    print("=" * 70)
    print("EXTERNAL GENERALIZATION SUMMARY")
    print("=" * 70)

    print()

    print(
        f"AI images detected correctly : "
        f"{ai_results['correct']}/"
        f"{ai_results['total']}"
    )

    print(
        f"AI detection rate            : "
        f"{ai_detection_rate:.2f}%"
    )

    print()

    print(
        f"Real images classified correctly: "
        f"{real_results['correct']}/"
        f"{real_results['total']}"
    )

    print(
        f"Real detection rate             : "
        f"{real_results['accuracy']:.2f}%"
    )

    print()

    print(
        f"False positives "
        f"(Real → AI)                 : "
        f"{real_false_positives}"
    )

    print(
        f"False-positive rate            : "
        f"{false_positive_rate:.2f}%"
    )

    print()

    print(
        f"Overall external accuracy      : "
        f"{overall_accuracy:.2f}%"
    )

    print()

    print("=" * 70)


if __name__ == "__main__":
    main()