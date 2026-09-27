import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Historical logs -> features -> trained AnomalyDetector."""

    def __init__(self, detector, preprocessor):
        self.detector = detector
        self.preprocessor = preprocessor
        self.training_summary = {}

    def prepare_training_data(self, df):
        """
        Accepts either raw structured HDFS logs (preprocessed here) or an
        already aggregated feature table (used as is). Returns a feature frame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame.")
        if df.empty:
            raise ValueError("Training data is empty.")

        already_features = all(c in df.columns for c in self.detector.feature_columns)
        features = df.copy() if already_features else self.preprocessor.preprocess(df)

        if features.empty:
            raise ValueError(
                "No feature windows could be created (check timestamps, "
                "or use a smaller preprocessor window such as '1min')."
            )
        if len(features) < 30:
            logger.warning(
                "Only %d feature windows. Consider a smaller window "
                "(DataPreprocessor(window='1min')).", len(features)
            )
        return features

    def train(self, X):
        self.detector.train(X)
        flags = self.detector.predict(X)
        self.training_summary = {
            "n_samples": int(len(flags)),
            "n_features": len(self.detector.feature_columns),
            "contamination": self.detector.contamination,
            "train_anomaly_rate": float(np.mean(flags)),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        self.detector.metadata.update(self.training_summary)
        return self.training_summary

    def evaluate(self, X, y_true=None):
        """
        Unsupervised summary always; precision/recall/F1/accuracy too when
        y_true (1/True = anomaly) is given, one label per row of X.
        """
        result = self.detector.predict_with_scores(X)
        flags = result["is_anomaly"].to_numpy()
        scores = result["anomaly_score"].to_numpy()

        report = {
            "n_samples": int(len(flags)),
            "anomaly_rate": float(flags.mean()),
            "score_mean": float(scores.mean()),
            "score_max": float(scores.max()),
        }

        if y_true is not None:
            y = np.asarray(y_true).astype(int).ravel()
            if len(y) != len(flags):
                raise ValueError(f"y_true has {len(y)} labels but X has {len(flags)} rows.")
            if not set(np.unique(y)) <= {0, 1}:
                raise ValueError("y_true must contain only 0/1 (or False/True).")

            pred = flags.astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            report.update({
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)),
                "f1": float(f1_score(y, pred, zero_division=0)),
                "accuracy": float(accuracy_score(y, pred)),
                "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
            })
        return report

    def save_model(self, path):
        return self.detector.save(path)