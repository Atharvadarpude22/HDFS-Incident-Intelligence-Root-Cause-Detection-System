"""features.csv -> ModelTrainer -> models/anomaly_model.pkl

Usage: python scripts/train_model.py [--features data/processed/features.csv]
                                     [--out models/anomaly_model.pkl] [--contamination 0.05]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]   # project root
sys.path.insert(0, str(ROOT))

from src.data import DataPreprocessor
from src.ml import AnomalyDetector, ModelTrainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def train(features_path, model_path, contamination=0.05):
    features_path = Path(features_path)
    if not features_path.is_file():
        raise FileNotFoundError(
            f"{features_path} not found. Run scripts/prepare_data.py first."
        )
    df = pd.read_csv(features_path)

    trainer = ModelTrainer(AnomalyDetector(contamination=contamination), DataPreprocessor())
    X = trainer.prepare_training_data(df)
    summary = trainer.train(X)
    trainer.save_model(model_path)
    return summary, trainer.evaluate(X)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", default=str(ROOT / "data" / "processed" / "features.csv"))
    p.add_argument("--out", default=str(ROOT / "models" / "anomaly_model.pkl"))
    p.add_argument("--contamination", type=float, default=0.05)
    a = p.parse_args()
    summary, report = train(a.features, a.out, a.contamination)
    print(json.dumps({"training": summary, "evaluation": report}, indent=2))