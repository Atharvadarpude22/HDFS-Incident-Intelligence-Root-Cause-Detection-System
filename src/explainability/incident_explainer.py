"""
Explainable-AI layer for the anomaly model.

IncidentExplainer answers: "WHY did the ML model consider this window
abnormal?" - as opposed to IncidentAgent.determine_root_cause(), which
answers "WHAT is the likely failure?" using logs + RAG + this explainer.

Uses SHAP's TreeExplainer when the `shap` package is installed (exact,
game-theoretic feature attributions for Isolation Forest). Falls back to a
permutation-importance approximation with no extra dependency, so the
module works even if SHAP isn't installed. Both paths expose the same
output shape, so callers never need to know which one ran.
"""
import logging

import numpy as np
import pandas as pd

from src.ml.anomaly_detector import ModelNotTrainedError

logger = logging.getLogger(__name__)

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

STRENGTH_BINS = (
    (0.6, "Strong"),
    (0.3, "Medium"),
    (0.0, "Weak"),
)


def _strength(share):
    for cutoff, label in STRENGTH_BINS:
        if share >= cutoff:
            return label
    return "Weak"


class IncidentExplainer:
    """
    model: a trained AnomalyDetector (src.ml.AnomalyDetector) — NOT the raw
    sklearn IsolationForest, so this class can reuse the detector's own
    input validation, NaN handling and feature order.
    """

    def __init__(self, model, background_size=100, random_state=42):
        if model is None:
            raise ValueError("model is required.")
        if not getattr(model, "is_trained", False):
            raise ModelNotTrainedError("IncidentExplainer requires a trained AnomalyDetector.")
        if background_size < 1:
            raise ValueError("background_size must be >= 1.")

        self.model = model
        self.feature_columns = list(model.feature_columns)
        self.background_size = background_size
        self.random_state = random_state
        self._shap_explainer = None  # built lazily on first explain() call

    # ------------------------------------------------------------------
    def _frame(self, features):
        df = self.model._to_frame(features)           # reuses AnomalyDetector's validation
        return df.fillna(self.model.medians_)

    def _build_shap_explainer(self):
        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self.model.model)
        return self._shap_explainer

    def _shap_contributions(self, df):
        """Signed per-feature contribution to the anomaly score, one row per sample."""
        explainer = self._build_shap_explainer()
        raw = np.asarray(explainer.shap_values(df.to_numpy(dtype=float)))
        # sklearn IsolationForest: shap_values are already toward "more anomalous" = positive.
        return -raw if getattr(explainer, "model_output", None) == "raw" else raw

    def _permutation_contributions(self, df):
        """
        No-SHAP fallback: for each row, zero out one feature at a time (replace it
        with the training median, i.e. "remove its information") and see how much
        the anomaly score drops. A feature that mattered a lot drops the score more.
        """
        rng = np.random.default_rng(self.random_state)
        baseline = self.model.score(df)
        contributions = np.zeros((len(df), len(self.feature_columns)))

        for j, col in enumerate(self.feature_columns):
            perturbed = df.copy()
            perturbed[col] = self.model.medians_[col]
            contributions[:, j] = baseline - self.model.score(perturbed)
        return contributions

    # ------------------------------------------------------------------
    def explain(self, features):
        """
        One or more rows -> per-row explanation:
          {anomaly_score, is_anomaly, base_score, top_factors: [...]}
        top_factors: see get_top_factors().
        """
        df = self._frame(features)
        result = self.model.predict_with_scores(df)

        try:
            contributions = self._shap_contributions(df) if _HAS_SHAP else self._permutation_contributions(df)
            method = "shap" if _HAS_SHAP else "permutation"
        except Exception as exc:  # SHAP failed at runtime -> don't crash the report
            logger.warning("SHAP failed (%s); falling back to permutation importance.", exc)
            contributions = self._permutation_contributions(df)
            method = "permutation"

        explanations = []
        for i in range(len(df)):
            factors = self._rank_factors(df.iloc[i], contributions[i])
            explanations.append({
                "anomaly_score": float(result["anomaly_score"].iloc[i]),
                "is_anomaly": bool(result["is_anomaly"].iloc[i]),
                "method": method,
                "top_factors": factors,
            })
        return explanations[0] if len(explanations) == 1 else explanations

    def get_top_factors(self, features, top_n=5):
        """Convenience: just the ranked factor list (no score/verdict wrapper)."""
        if top_n < 1:
            raise ValueError("top_n must be >= 1.")
        explanation = self.explain(features)
        rows = explanation if isinstance(explanation, list) else [explanation]
        factors = [r["top_factors"][:top_n] for r in rows]
        return factors[0] if len(factors) == 1 else factors

    def _rank_factors(self, row, contributions):
        total = float(np.abs(contributions).sum()) or 1.0
        order = np.argsort(-np.abs(contributions))
        return [
            {
                "feature": self.feature_columns[j],
                "value": row[self.feature_columns[j]].item()
                if hasattr(row[self.feature_columns[j]], "item") else row[self.feature_columns[j]],
                "contribution": round(float(contributions[j]), 4),
                "direction": "increases" if contributions[j] > 0 else "decreases",
                "share": round(abs(float(contributions[j])) / total, 4),
                "strength": _strength(abs(float(contributions[j])) / total),
            }
            for j in order
        ]

    # ------------------------------------------------------------------
    def generate_explanation(self, factors, top_n=4):
        """Ranked factor list (as from get_top_factors) -> plain-English sentences."""
        if not isinstance(factors, list):
            raise TypeError("factors must be the list returned by get_top_factors().")
        if not factors:
            return "No contributing factors were found."

        rising = [f for f in factors if f["direction"] == "increases"][:top_n]
        if not rising:
            return "No factor pushed this window toward being flagged as abnormal."

        lines = [f"{f['feature'].replace('_', ' ')} = {f['value']} \u2192 {f['strength']}"
                for f in rising]
        return "Main factors pushing this window toward 'abnormal':\n" + \
            "\n".join(f"- {line}" for line in lines)