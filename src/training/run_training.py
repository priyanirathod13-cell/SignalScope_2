import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trainer import train_model
from src.evaluation.evaluate import evaluate_test_set

def load_config(config_path='config/train_config.json'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    cfg = load_config('config/train_config.json')

    # 1. Train model
    model, history, best_auc = train_model(cfg)

    # 2. Evaluate on development test set
    evaluate_test_set(cfg)

if __name__ == '__main__':
    main()
