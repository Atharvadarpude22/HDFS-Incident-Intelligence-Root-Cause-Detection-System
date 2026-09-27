import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml import AnomalyDetector
from src.ml.anomaly_detector import ModelNotTrainedError
from src.explainability import IncidentExplainer
import src.explainability.incident_explainer as ie_module


def make_features(n_normal=60, n_spikes=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_normal):
        logs, errors = int(rng.integers(80, 120)), int(rng.integers(0, 3))
        rows.append({
            "logs_per_window": logs, "error_count": errors, "warning_count": int(rng.integers(0, 4)),
            "info_count": logs - errors, "fatal_count": 0, "heartbeat_failures": 0,
            "block_errors": 0, "replication_errors": 0, "network_errors": 0, "disk_errors": 0,
            "unique_components": 5, "unique_event_ids": int(rng.integers(8, 12)),
            "unique_processes": int(rng.integers(10, 15)),
            "error_ratio": errors / logs, "warning_ratio": 0.02, "fatal_ratio": 0.0,
        })
    for _ in range(n_spikes):     # heartbeat-dominated spike, by design
        rows.append({
            "logs_per_window": 900, "error_count": 300, "warning_count": 150,
            "info_count": 450, "fatal_count": 20, "heartbeat_failures": 200,
            "block_errors": 10, "replication_errors": 5, "network_errors": 5, "disk_errors": 2,
            "unique_components": 12, "unique_event_ids": 40, "unique_processes": 60,
            "error_ratio": 0.33, "warning_ratio": 0.17, "fatal_ratio": 0.02,
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def trained():
    df = make_features()
    return AnomalyDetector(contamination=0.05).train(df), df


@pytest.fixture(params=["shap", "permutation"])
def explainer(request, trained, monkeypatch):
    if request.param == "permutation":
        monkeypatch.setattr(ie_module, "_HAS_SHAP", False)
    det, _ = trained
    return IncidentExplainer(det), request.param


# ---------------- construction ----------------
def test_requires_trained_model():
    with pytest.raises(ModelNotTrainedError):
        IncidentExplainer(AnomalyDetector())
    with pytest.raises(ValueError):
        IncidentExplainer(None)


def test_uses_shap_when_available(trained):
    det, _ = trained
    assert ie_module._HAS_SHAP is True                     # installed in this environment
    result = IncidentExplainer(det).explain(make_features(1, 1).iloc[[-1]])
    assert result["method"] == "shap"


# ---------------- explain() ----------------
def test_explain_single_row_anomalous(explainer, trained):
    exp, method = explainer
    _, df = trained
    spike = df.iloc[[-1]]                                   # last row is an injected spike

    result = exp.explain(spike)
    assert result["is_anomaly"] and result["anomaly_score"] > 0.5
    assert result["method"] == method
    assert len(result["top_factors"]) == len(exp.feature_columns)

    top = result["top_factors"][0]
    assert {"feature", "value", "contribution", "direction", "share", "strength"} <= top.keys()
    assert top["direction"] == "increases"
    # the spike elevates several features at once; the top one should be one of them,
    # not an untouched feature such as unique_processes or info_count alone
    assert top["feature"] in ("heartbeat_failures", "error_count", "fatal_count",
                              "block_errors", "warning_count", "logs_per_window",
                              "error_ratio", "warning_ratio")
    shares = [f["share"] for f in result["top_factors"]]
    assert shares == sorted(shares, reverse=True)           # ranked descending
    assert sum(shares) == pytest.approx(1.0, abs=1e-3)       # shares are normalized by total |contribution|


def test_explain_normal_row(explainer, trained):
    exp, _ = explainer
    _, df = trained
    result = exp.explain(df.iloc[[0]])
    assert result["is_anomaly"] is False and result["anomaly_score"] < 0.5


def test_explain_multiple_rows(explainer, trained):
    exp, _ = explainer
    _, df = trained
    results = exp.explain(df.iloc[[0, -1]])
    assert isinstance(results, list) and len(results) == 2
    assert results[0]["is_anomaly"] is False and results[1]["is_anomaly"] is True


def test_explain_dict_input(explainer, trained):
    exp, _ = explainer
    _, df = trained
    result = exp.explain(df.iloc[-1].to_dict())
    assert result["is_anomaly"]


def test_explain_bad_input(explainer):
    exp, _ = explainer
    with pytest.raises(ValueError):
        exp.explain({})
    with pytest.raises(ValueError):
        exp.explain(pd.DataFrame({"only_one_col": [1]}))


def test_shap_runtime_failure_falls_back(trained, monkeypatch):
    det, df = trained
    exp = IncidentExplainer(det)
    monkeypatch.setattr(exp, "_shap_contributions", lambda df: (_ for _ in ()).throw(RuntimeError("boom")))
    result = exp.explain(df.iloc[[-1]])
    assert result["method"] == "permutation" and result["is_anomaly"]


# ---------------- get_top_factors ----------------
def test_get_top_factors_top_n(explainer, trained):
    exp, _ = explainer
    _, df = trained
    factors = exp.get_top_factors(df.iloc[[-1]], top_n=3)
    assert len(factors) == 3

    with pytest.raises(ValueError):
        exp.get_top_factors(df.iloc[[-1]], top_n=0)


def test_get_top_factors_multi_row(explainer, trained):
    exp, _ = explainer
    _, df = trained
    factors = exp.get_top_factors(df.iloc[[0, -1]], top_n=2)
    assert isinstance(factors, list) and len(factors) == 2
    assert all(len(f) == 2 for f in factors)


# ---------------- generate_explanation ----------------
def test_generate_explanation_text(explainer, trained):
    exp, _ = explainer
    _, df = trained
    factors = exp.get_top_factors(df.iloc[[-1]])
    text = exp.generate_explanation(factors)
    assert "abnormal" in text and any(f["feature"].replace("_", " ") in text for f in factors[:1])


def test_generate_explanation_edge_cases(explainer):
    exp, _ = explainer
    assert exp.generate_explanation([]) == "No contributing factors were found."
    all_decreasing = [{"feature": "x", "value": 1, "contribution": -0.5,
                       "direction": "decreases", "share": 1.0, "strength": "Strong"}]
    assert "No factor pushed" in exp.generate_explanation(all_decreasing)

    with pytest.raises(TypeError):
        exp.generate_explanation("not a list")


# ---------------- integration: same numbers the agent would show ----------------
def test_matches_detector_score(explainer, trained):
    exp, _ = explainer
    det, df = trained
    row = df.iloc[[-1]]
    assert exp.explain(row)["anomaly_score"] == pytest.approx(float(det.score(row)[0]), abs=1e-6)