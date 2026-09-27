import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import DataPreprocessor
from src.ml import (AnomalyDetector, ModelNotTrainedError, ModelPredictor,
                    ModelTrainer, DEFAULT_FEATURE_COLUMNS)
from scripts.train_model import train as train_script


def make_features(n_normal=60, n_spikes=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_normal):
        logs = int(rng.integers(80, 120))
        errors = int(rng.integers(0, 3))
        rows.append({
            "logs_per_window": logs, "error_count": errors, "warning_count": int(rng.integers(0, 4)),
            "info_count": logs - errors, "fatal_count": 0, "heartbeat_failures": 0,
            "block_errors": 0, "replication_errors": 0, "network_errors": 0, "disk_errors": 0,
            "unique_components": 5, "unique_event_ids": int(rng.integers(8, 12)),
            "unique_processes": int(rng.integers(10, 15)),
            "error_ratio": errors / logs, "warning_ratio": 0.02, "fatal_ratio": 0.0,
        })
    for _ in range(n_spikes):
        rows.append({
            "logs_per_window": 900, "error_count": 300, "warning_count": 150,
            "info_count": 450, "fatal_count": 20, "heartbeat_failures": 40,
            "block_errors": 90, "replication_errors": 30, "network_errors": 25, "disk_errors": 15,
            "unique_components": 12, "unique_event_ids": 40, "unique_processes": 60,
            "error_ratio": 0.33, "warning_ratio": 0.17, "fatal_ratio": 0.02,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def trained():
    df = make_features()
    return AnomalyDetector(contamination=0.05).train(df), df


# ---------------- AnomalyDetector ----------------
def test_spikes_flagged_and_scores_valid(trained):
    det, df = trained
    res = det.predict_with_scores(df)
    assert res["anomaly_score"].between(0, 1).all()
    assert res["is_anomaly"].iloc[-3:].all()                       # injected spikes
    assert res["anomaly_score"].iloc[-3:].min() > res["anomaly_score"].iloc[:-3].median()
    assert ((res["anomaly_score"] > 0.5) == res["is_anomaly"]).all()   # consistent
    assert res["is_anomaly"].mean() <= 0.15


def test_input_formats(trained):
    det, df = trained
    row = df.iloc[-1]
    assert det.predict(row.to_dict())[0]                              # dict
    assert det.predict([row.to_dict(), df.iloc[0].to_dict()]).tolist() == [True, False]
    assert det.predict(row[DEFAULT_FEATURE_COLUMNS].to_numpy()).shape == (1,)   # 1D array
    assert det.score(df.iloc[[0]]).shape == (1,)                      # 1-row frame


def test_nan_inf_and_bad_values_handled(trained):
    det, df = trained
    bad = df.iloc[[0, 1, 2]].copy().astype(object)
    bad.loc[bad.index[0], "error_count"] = np.nan
    bad.loc[bad.index[1], "logs_per_window"] = np.inf
    bad.loc[bad.index[2], "info_count"] = "abc"
    scores = det.score(bad)
    assert np.isfinite(scores).all()


def test_extra_columns_ignored(trained):
    det, df = trained
    df2 = df.copy(); df2["time_window"] = pd.Timestamp("2008-11-09"); df2["junk"] = "x"
    assert det.predict(df2).shape == (len(df),)


def test_errors(trained):
    det, df = trained
    with pytest.raises(ModelNotTrainedError):
        AnomalyDetector().predict(df)
    with pytest.raises(ModelNotTrainedError):
        AnomalyDetector().save("x.pkl")
    with pytest.raises(ValueError):
        det.predict(df.drop(columns=["error_count"]))
    with pytest.raises(ValueError):
        det.predict(df.iloc[0:0])
    with pytest.raises(ValueError):
        det.predict(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        AnomalyDetector().train(df.iloc[:3])
    with pytest.raises(ValueError):
        AnomalyDetector(contamination=0.9)


def test_save_load_roundtrip(trained, tmp_path):
    det, df = trained
    path = det.save(tmp_path / "nested" / "model.pkl")
    loaded = AnomalyDetector().load(path)
    np.testing.assert_allclose(det.score(df), loaded.score(df))
    assert loaded.feature_columns == det.feature_columns


def test_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        AnomalyDetector().load(tmp_path / "missing.pkl")
    bad = tmp_path / "bad.pkl"; bad.write_bytes(b"not a pickle")
    with pytest.raises(ValueError):
        AnomalyDetector().load(bad)
    import joblib
    wrong = tmp_path / "wrong.pkl"; joblib.dump({"foo": 1}, wrong)
    with pytest.raises(ValueError):
        AnomalyDetector().load(wrong)


# ---------------- ModelTrainer ----------------
def raw_logs(n=400):
    ts = pd.date_range("2008-11-09 20:00:00", periods=n, freq="10s")
    return pd.DataFrame({
        "LineId": range(1, n + 1),
        "Date": ts.strftime("%y%m%d"), "Time": ts.strftime("%H%M%S"),
        "Pid": np.arange(n) % 20, "Level": ["INFO"] * (n - 10) + ["ERROR"] * 10,
        "Component": ["dfs.A", "dfs.B"] * (n // 2),
        "Content": ["Receiving block blk_1"] * (n - 10) + ["heartbeat block timeout"] * 10,
        "EventId": [f"E{i % 5}" for i in range(n)], "EventTemplate": ["t"] * n,
    })


def test_trainer_from_raw_logs_and_features():
    trainer = ModelTrainer(AnomalyDetector(), DataPreprocessor())
    X = trainer.prepare_training_data(raw_logs())
    assert "logs_per_window" in X.columns and len(X) > 5
    summary = trainer.train(X)
    assert summary["n_samples"] == len(X)

    same = trainer.prepare_training_data(X)            # already features -> unchanged
    assert len(same) == len(X)

    with pytest.raises(ValueError):
        trainer.prepare_training_data(raw_logs().iloc[0:0])
    with pytest.raises(TypeError):
        trainer.prepare_training_data([1, 2])


def test_trainer_evaluate_with_labels():
    df = make_features()
    labels = [0] * 60 + [1] * 3
    trainer = ModelTrainer(AnomalyDetector(), DataPreprocessor())
    trainer.train(df)
    rep = trainer.evaluate(df, labels)
    assert rep["recall"] == 1.0 and 0 <= rep["precision"] <= 1
    assert set(rep["confusion"]) == {"tp", "fp", "fn", "tn"}
    assert "precision" not in trainer.evaluate(df)                    # unsupervised
    with pytest.raises(ValueError):
        trainer.evaluate(df, [0, 1])
    with pytest.raises(ValueError):
        trainer.evaluate(df, [0] * 62 + [5])


# ---------------- ModelPredictor + script ----------------
def test_predictor(tmp_path):
    df = make_features()
    trainer = ModelTrainer(AnomalyDetector(), DataPreprocessor())
    trainer.train(df)
    path = trainer.save_model(tmp_path / "m.pkl")

    pred = ModelPredictor(path)                                       # lazy load
    out = pred.predict(df.iloc[[0, -1]])
    assert out[0]["is_anomaly"] is False and out[1]["is_anomaly"] is True
    assert pred.get_anomaly_score(df) == max(pred.detector.score(df))
    assert pred.detector.metadata["n_samples"] == len(df)

    with pytest.raises(FileNotFoundError):
        ModelPredictor(tmp_path / "nope.pkl").predict(df)


def test_train_script(tmp_path):
    make_features().to_csv(tmp_path / "features.csv", index=False)
    summary, report = train_script(tmp_path / "features.csv", tmp_path / "models" / "m.pkl")
    assert (tmp_path / "models" / "m.pkl").exists() and summary["n_samples"] == 63
    with pytest.raises(FileNotFoundError):
        train_script(tmp_path / "missing.csv", tmp_path / "m.pkl")