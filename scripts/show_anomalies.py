import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ml import ModelPredictor

features = pd.read_csv(ROOT / "data/processed/features.csv")
scores = pd.DataFrame(ModelPredictor(ROOT / "models/anomaly_model.pkl").predict(features))
out = pd.concat([features[["time_window", "logs_per_window", "error_count",
                           "block_errors", "heartbeat_failures"]], scores], axis=1)
print(out.sort_values("anomaly_score", ascending=False).head(10).to_string())