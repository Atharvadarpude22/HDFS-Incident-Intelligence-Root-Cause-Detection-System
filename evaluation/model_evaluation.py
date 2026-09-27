"""
Anomaly-detection evaluation for the trained AnomalyDetector.

HDFS_2k ships no ground-truth failure labels, so there are two ways to
evaluate here:

1. evaluate_with_labels(detector, X, y_true)
   Real precision / recall / F1 / accuracy / ROC-AUC / PR-AUC - use this once
   you have labels (a hand-labeled subset of windows, or another HDFS
   dataset that ships block-level anomaly labels you've mapped onto your
   time windows).

2. evaluate_synthetic(detector, X)
   No labels required. Injects clearly-abnormal synthetic windows (real rows
   with their incident-signal features scaled up) into a copy of your real
   feature data, then checks whether the detector ranks them above the real
   (presumably mostly-normal) windows. This is a SANITY CHECK of the
   detector's discriminative power on data shaped like yours - it is NOT a
   measurement of real-world accuracy, and the report says so explicitly.

Usage:
    python evaluation/model_evaluation.py
    python evaluation/model_evaluation.py --labels data/processed/labels.csv
    python evaluation/model_evaluation.py --out evaluation/model_report.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT))

from src.ml.anomaly_detector import AnomalyDetector, DEFAULT_FEATURE_COLUMNS, ModelNotTrainedError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("model_evaluation")

SYNTHETIC_CAVEAT = (
    "Synthetic evaluation: HDFS_2k has no ground-truth failure labels, so these "
    "metrics come from synthetic anomalies injected into real feature data. They "
    "measure whether the detector can separate clearly-abnormal windows from your "
    "real (mostly normal) ones - NOT real-world accuracy on actual failures."
)

# Features a real incident spikes; scaled up to build a synthetic anomaly.
INCIDENT_FEATURES = [
    "error_count", "fatal_count", "warning_count", "heartbeat_failures",
    "block_errors", "replication_errors", "network_errors", "disk_errors",
]


def _safe_roc_auc(y_true, scores):
    if len(set(y_true)) < 2:
        return None  # undefined with only one class present
    return float(roc_auc_score(y_true, scores))


def evaluate_with_labels(detector, X, y_true):
    """
    detector: a trained AnomalyDetector.
    X: feature rows (DataFrame / dict / list of dicts - anything AnomalyDetector accepts).
    y_true: one 0/1 (or False/True) label per row of X, 1 = anomaly.
    """
    if not detector.is_trained:
        raise ModelNotTrainedError("detector must be trained before evaluation.")

    result = detector.predict_with_scores(X)
    scores = result["anomaly_score"].to_numpy()
    pred = result["is_anomaly"].to_numpy().astype(int)

    y = np.asarray(y_true).astype(int).ravel()
    if len(y) != len(pred):
        raise ValueError(f"y_true has {len(y)} labels but X has {len(pred)} rows.")
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("y_true must contain only 0/1 (or False/True).")

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "mode": "labeled",
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": _safe_roc_auc(y, scores),
        "pr_auc": float(average_precision_score(y, scores)) if len(set(y)) > 1 else None,
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    }


def inject_synthetic_anomalies(X, feature_columns=None, n_synthetic=None, factor=10.0, random_state=42):
    """
    Returns (X_combined, y_synthetic): the original rows (label 0) plus
    n_synthetic new rows (label 1) built by taking real rows and scaling up
    their incident-signal features, simulating an obvious spike.
    """
    base = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X if not isinstance(X, dict) else [X])
    if base.empty:
        raise ValueError("X is empty; nothing to evaluate.")

    signal_cols = [c for c in INCIDENT_FEATURES if c in base.columns]
    if not signal_cols:
        raise ValueError("None of the expected incident-signal columns are present in X: "
                         f"{INCIDENT_FEATURES}")

    rng = np.random.default_rng(random_state)
    n_synthetic = n_synthetic or max(5, int(round(0.15 * len(base))))
    n_synthetic = max(1, min(n_synthetic, max(len(base), 5)))

    seed_rows = base.sample(n=n_synthetic, replace=n_synthetic > len(base), random_state=random_state)
    synthetic = seed_rows.reset_index(drop=True).copy()

    for col in signal_cols:
        baseline = max(float(base[col].median()), 1.0)
        synthetic[col] = (synthetic[col].astype(float)
                          + baseline * factor * rng.uniform(0.7, 1.3, size=len(synthetic)))
    if "logs_per_window" in synthetic.columns:
        synthetic["logs_per_window"] = (synthetic["logs_per_window"].astype(float)
                                        * rng.uniform(2.0, 4.0, size=len(synthetic)))
    for ratio_col, count_col in (("error_ratio", "error_count"), ("warning_ratio", "warning_count"),
                                 ("fatal_ratio", "fatal_count")):
        if ratio_col in synthetic.columns and count_col in synthetic.columns \
                and "logs_per_window" in synthetic.columns:
            synthetic[ratio_col] = (synthetic[count_col] / synthetic["logs_per_window"]).clip(upper=1.0)

    combined = pd.concat([base, synthetic], ignore_index=True)
    y = np.concatenate([np.zeros(len(base)), np.ones(len(synthetic))]).astype(int)
    return combined, y


def evaluate_synthetic(detector, X, n_synthetic=None, factor=10.0, top_k=None, random_state=42):
    if not detector.is_trained:
        raise ModelNotTrainedError("detector must be trained before evaluation.")

    combined, y = inject_synthetic_anomalies(X, detector.feature_columns, n_synthetic, factor, random_state)
    result = detector.predict_with_scores(combined)
    scores = result["anomaly_score"].to_numpy()
    pred = result["is_anomaly"].to_numpy().astype(int)

    n_synth = int(y.sum())
    top_k = min(top_k or n_synth, len(y))
    top_indices = np.argsort(-scores)[:top_k]
    recall_at_k = float(y[top_indices].sum() / n_synth) if n_synth else None

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "mode": "synthetic",
        "caveat": SYNTHETIC_CAVEAT,
        "n_real_samples": int(len(y) - n_synth),
        "n_synthetic_anomalies": n_synth,
        "roc_auc": _safe_roc_auc(y, scores),
        "pr_auc": float(average_precision_score(y, scores)) if n_synth else None,
        "top_k": top_k,
        "recall_at_top_k": recall_at_k,
        "precision_at_trained_threshold": float(precision_score(y, pred, zero_division=0)),
        "recall_at_trained_threshold": float(recall_score(y, pred, zero_division=0)),
        "confusion_matrix_at_trained_threshold": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    }


def unsupervised_summary(detector, X):
    if not detector.is_trained:
        raise ModelNotTrainedError("detector must be trained before evaluation.")
    result = detector.predict_with_scores(X)
    scores = result["anomaly_score"].to_numpy()
    return {
        "mode": "unsupervised",
        "n_samples": int(len(scores)),
        "anomaly_rate": float(result["is_anomaly"].mean()),
        "score_mean": float(scores.mean()),
        "score_p95": float(np.percentile(scores, 95)),
        "score_max": float(scores.max()),
        "contamination_setting": detector.contamination,
    }


def full_report(detector, X, y_true=None, **synthetic_kwargs):
    report = {
        "unsupervised": unsupervised_summary(detector, X),
        "synthetic": evaluate_synthetic(detector, X, **synthetic_kwargs),
    }
    if y_true is not None:
        report["labeled"] = evaluate_with_labels(detector, X, y_true)
    return report


def _load_labels(labels_path, features_df):
    """CSV with columns time_window,label -> (matched feature rows, y_true), inner-joined
    on time_window. Rows in features_df with no matching label are dropped (with a warning),
    rather than failing outright, since partial labeling is the common real case."""
    labels_df = pd.read_csv(labels_path)
    if not {"time_window", "label"} <= set(labels_df.columns):
        raise ValueError(f"{labels_path} must have columns: time_window,label")

    # Compare as strings: features_df may hold real Timestamps (in-memory) or strings
    # (round-tripped through CSV), and labels_df is always strings straight from CSV.
    left = features_df.assign(_key=features_df["time_window"].astype(str))
    right = labels_df.assign(_key=labels_df["time_window"].astype(str))[["_key", "label"]]
    merged = left.merge(right, on="_key", how="inner").drop(columns=["_key"])
    if merged.empty:
        raise ValueError(f"No feature windows matched any time_window in {labels_path}.")
    if len(merged) < len(features_df):
        logger.warning("Only %d/%d feature windows have a label (%.0f%% coverage).",
                       len(merged), len(features_df), 100 * len(merged) / len(features_df))
    return merged.drop(columns=["label"]), merged["label"].astype(int).to_numpy()


def run(features_path, model_path, labels_path=None, top_k=None, factor=10.0, out_path=None):
    features_path = Path(features_path)
    if not features_path.is_file():
        raise FileNotFoundError(f"{features_path} not found. Run scripts/prepare_data.py first.")
    features = pd.read_csv(features_path)
    if features.empty:
        raise ValueError(f"{features_path} has no rows.")

    detector = AnomalyDetector().load(model_path)

    report = {
        "unsupervised": unsupervised_summary(detector, features),
        "synthetic": evaluate_synthetic(detector, features, top_k=top_k, factor=factor),
    }
    if labels_path:
        labeled_features, y_true = _load_labels(labels_path, features)
        report["labeled"] = evaluate_with_labels(detector, labeled_features, y_true)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2))
        logger.info("Saved report to %s", out_path)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", default=str(ROOT / "data" / "processed" / "features.csv"))
    p.add_argument("--model", default=str(ROOT / "models" / "anomaly_model.pkl"))
    p.add_argument("--labels", default=None,
                  help="optional CSV with columns time_window,label (1=anomaly)")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--factor", type=float, default=10.0,
                  help="how much to scale incident-signal features for synthetic anomalies")
    p.add_argument("--out", default=None, help="optional path to save the report as JSON")
    args = p.parse_args()

    result = run(args.features, args.model, args.labels, args.top_k, args.factor, args.out)
    print(json.dumps(result, indent=2))