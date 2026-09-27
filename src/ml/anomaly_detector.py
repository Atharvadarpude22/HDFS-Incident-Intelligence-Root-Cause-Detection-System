import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

# Numeric window features produced by DataPreprocessor.create_features()
DEFAULT_FEATURE_COLUMNS = [
    "logs_per_window",
    "error_count", "warning_count", "info_count", "fatal_count",
    "heartbeat_failures", "block_errors", "replication_errors",
    "network_errors", "disk_errors",
    "unique_components", "unique_event_ids", "unique_processes",
    "error_ratio", "warning_ratio", "fatal_ratio",
]


class ModelNotTrainedError(RuntimeError):
    """Raised when predict/score/save is called before train/load."""


class AnomalyDetector:
    """
    Isolation Forest anomaly detector.

    anomaly_score is in [0, 1]:  > 0.5 means anomalous, and
    is_anomaly is exactly (anomaly_score > 0.5).  The score is the Isolation
    Forest decision value squashed through a sigmoid calibrated on the
    training data, so 0.5 sits on the contamination threshold.
    """

    MIN_TRAIN_SAMPLES = 5

    def __init__(self, contamination=0.05, n_estimators=200, random_state=42,
                 feature_columns=None):
        if contamination != "auto" and not (0 < contamination <= 0.5):
            raise ValueError("contamination must be in (0, 0.5] or 'auto'.")
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")

        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.feature_columns = list(feature_columns or DEFAULT_FEATURE_COLUMNS)

        self.model = None
        self.medians_ = None       # used to fill NaN at predict time
        self.score_scale_ = None   # sigmoid calibration
        self.metadata = {}         # free-form info (filled by ModelTrainer)

    # ------------------------------------------------------------------
    @property
    def is_trained(self):
        return self.model is not None

    def _check_trained(self):
        if not self.is_trained:
            raise ModelNotTrainedError("Model is not trained. Call train() or load() first.")

    def _to_frame(self, X):
        """DataFrame / dict / list of dicts / Series / array -> numeric feature frame."""
        if isinstance(X, pd.DataFrame):
            df = X
        elif isinstance(X, dict):
            df = pd.DataFrame([X])
        elif isinstance(X, pd.Series):
            df = X.to_frame().T
        elif isinstance(X, (list, tuple)) and len(X) > 0 and isinstance(X[0], dict):
            df = pd.DataFrame(list(X))
        else:
            try:
                arr = np.asarray(X, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Unsupported input for anomaly detection: {exc}") from exc
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim != 2 or arr.shape[1] != len(self.feature_columns):
                raise ValueError(
                    f"Expected {len(self.feature_columns)} features per row, got shape {arr.shape}."
                )
            df = pd.DataFrame(arr, columns=self.feature_columns)

        if df.empty:
            raise ValueError("Input is empty.")

        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        return (
            df[self.feature_columns]
            .apply(pd.to_numeric, errors="coerce")       # bad values -> NaN
            .replace([np.inf, -np.inf], np.nan)
        )

    def _sigmoid_score(self, decision):
        z = np.clip(decision / self.score_scale_, -50, 50)
        return 1.0 / (1.0 + np.exp(z))

    # ------------------------------------------------------------------
    def train(self, X):
        df = self._to_frame(X)
        if len(df) < self.MIN_TRAIN_SAMPLES:
            raise ValueError(
                f"Need at least {self.MIN_TRAIN_SAMPLES} training rows, got {len(df)}."
            )
        if len(df) < 30:
            logger.warning("Only %d training rows; results will be unstable.", len(df))

        self.medians_ = df.median().fillna(0.0)
        data = df.fillna(self.medians_).to_numpy(dtype=float)

        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        model.fit(data)

        train_decision = model.decision_function(data)
        self.score_scale_ = max(float(np.std(train_decision)), 1e-6)
        self.model = model
        return self

    def _decision(self, X):
        self._check_trained()
        data = self._to_frame(X).fillna(self.medians_).to_numpy(dtype=float)
        return self.model.decision_function(data)  # < 0 => anomaly

    def score(self, X):
        """Anomaly score per row in [0, 1] (higher = more abnormal)."""
        return self._sigmoid_score(self._decision(X))

    def predict(self, X):
        """Boolean array: True = anomaly."""
        return self._decision(X) < 0

    def predict_with_scores(self, X):
        """DataFrame with anomaly_score and is_anomaly for every input row."""
        decision = self._decision(X)
        return pd.DataFrame({
            "anomaly_score": self._sigmoid_score(decision),
            "is_anomaly": decision < 0,
        })

    # ------------------------------------------------------------------
    def save(self, path):
        self._check_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_columns": self.feature_columns,
            "medians": self.medians_,
            "score_scale": self.score_scale_,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "metadata": self.metadata,
            "sklearn_version": sklearn.__version__,
        }, path)
        return path

    def load(self, path):
        # joblib/pickle can run code: only load model files you created yourself.
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")

        try:
            payload = joblib.load(path)
        except Exception as exc:
            raise ValueError(f"Could not read model file {path}: {exc}") from exc

        required = {"model", "feature_columns", "medians", "score_scale"}
        if not isinstance(payload, dict) or not required <= payload.keys():
            raise ValueError(f"{path} is not a valid AnomalyDetector model file.")

        saved_version = payload.get("sklearn_version")
        if saved_version and saved_version != sklearn.__version__:
            warnings.warn(
                f"Model saved with scikit-learn {saved_version}, "
                f"running {sklearn.__version__}."
            )

        self.model = payload["model"]
        self.feature_columns = list(payload["feature_columns"])
        self.medians_ = payload["medians"]
        self.score_scale_ = payload["score_scale"]
        self.contamination = payload.get("contamination", self.contamination)
        self.n_estimators = payload.get("n_estimators", self.n_estimators)
        self.random_state = payload.get("random_state", self.random_state)
        self.metadata = payload.get("metadata", {})
        return self