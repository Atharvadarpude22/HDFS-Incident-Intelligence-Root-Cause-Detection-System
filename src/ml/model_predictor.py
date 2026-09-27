from pathlib import Path

from .anomaly_detector import AnomalyDetector


class ModelPredictor:
    """Uses an already-trained model (loaded from disk) to score features."""

    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.detector = None

    def load_model(self):
        self.detector = AnomalyDetector().load(self.model_path)
        return self.detector

    def _ensure_loaded(self):
        if self.detector is None:
            self.load_model()  # lazy load

    def predict(self, features):
        """One dict per row: {'anomaly_score': float, 'is_anomaly': bool}."""
        self._ensure_loaded()
        result = self.detector.predict_with_scores(features)
        return [
            {"anomaly_score": float(s), "is_anomaly": bool(a)}
            for s, a in zip(result["anomaly_score"], result["is_anomaly"])
        ]

    def get_anomaly_score(self, features):
        """Anomaly score in [0, 1]; for several rows (windows) the highest score."""
        self._ensure_loaded()
        return float(self.detector.score(features).max())