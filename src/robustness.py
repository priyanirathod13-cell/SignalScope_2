"""
SignalScope Robustness CLI: Execute Model Robustness and Generalization Suite
Usage:
  python -m src.robustness
  python src/robustness.py --config config/train_config.json
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.robustness import run_robustness_evaluation


def parse_args():
    parser = argparse.ArgumentParser(
        description="SignalScope: Robustness & Generalization Evaluation Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_config.json",
        help="Path to training config JSON"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="reports/robustness_report.json",
        help="Destination path for JSON report"
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="reports/robustness_report.md",
        help="Destination path for Markdown report"
    )
    parser.add_argument(
        "--samples-dir",
        type=str,
        default="reports/robustness_samples",
        help="Directory to save visual examples of transformations"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_robustness_evaluation(
        cfg_path=args.config,
        output_json=args.output_json,
        output_md=args.output_md,
        samples_dir=args.samples_dir
    )


if __name__ == "__main__":
    main()
