"""Raw HDFS -> load -> validate -> preprocess -> data/processed/{logs,features}.csv

Usage: python scripts/prepare_data.py [--raw data/raw/hdfs] [--out data/processed] [--strict]
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # project root
sys.path.insert(0, str(ROOT))

from src.data import DataLoader, DataValidator, DataPreprocessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("prepare_data")


def prepare(raw_dir, out_dir, strict=False, window="5min"):
    data = DataLoader(raw_dir).load_all(strict=False)
    df = data["structured_logs"]

    report = DataValidator().validate(df)
    if not report["can_process"]:
        raise ValueError(f"Validation failed: {report['errors']}")
    if strict and not report["is_valid"]:
        raise ValueError(f"Strict validation failed: {report['warnings']}")
    for w in report["warnings"]:
        logger.warning("Validation warning: %s", w)

    preprocessor = DataPreprocessor()
    logs = preprocessor.preprocess_logs(df)
    features = preprocessor.create_features(logs)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs.to_csv(out_dir / "logs.csv", index=False)
    features.to_csv(out_dir / "features.csv", index=False)
    logger.info("Saved %d log rows and %d feature windows to %s",
                len(logs), len(features), out_dir)
    return logs, features, report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=str(ROOT / "data" / "raw" / "hdfs"))
    p.add_argument("--out", default=str(ROOT / "data" / "processed"))
    p.add_argument("--strict", action="store_true")
    p.add_argument("--window", default="5min", help="e.g. 5min, 1min")
    a = p.parse_args()
    prepare(a.raw, a.out, a.strict, a.window)