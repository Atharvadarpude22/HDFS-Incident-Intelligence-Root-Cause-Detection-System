import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml import AnomalyDetector
from src.ml.anomaly_detector import ModelNotTrainedError
from src.rag import DocumentLoader, DocumentChunker, VectorStore
from evaluation import model_evaluation as me
from evaluation import rag_evaluation as re
from langchain_core.embeddings import DeterministicFakeEmbedding

KB_DIR = Path(__file__).resolve().parents[1] / "data" / "knowledge_base"


# ---------------------------------------------------------------- shared fixtures
def make_features(n_normal=80, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_normal):
        logs, errors = int(rng.integers(80, 120)), int(rng.integers(0, 3))
        rows.append({
            "time_window": pd.Timestamp("2008-11-09") + pd.Timedelta(minutes=len(rows)),
            "logs_per_window": logs, "error_count": errors, "warning_count": int(rng.integers(0, 4)),
            "info_count": logs - errors, "fatal_count": 0, "heartbeat_failures": 0,
            "block_errors": 0, "replication_errors": 0, "network_errors": 0, "disk_errors": 0,
            "unique_components": 5, "unique_event_ids": int(rng.integers(8, 12)),
            "unique_processes": int(rng.integers(10, 15)),
            "error_ratio": errors / logs, "warning_ratio": 0.02, "fatal_ratio": 0.0,
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def detector():
    return AnomalyDetector(contamination=0.05).train(make_features())


# ================================================================== model_evaluation
def test_inject_synthetic_anomalies_shapes():
    X = make_features(40)
    combined, y = me.inject_synthetic_anomalies(X, n_synthetic=6, factor=10.0)
    assert len(combined) == 46 and y.sum() == 6
    assert set(np.unique(y)) == {0, 1}
    # synthetic rows should have clearly higher error_count than the real median
    synth_errors = combined.loc[y == 1, "error_count"]
    assert synth_errors.min() > X["error_count"].median()


def test_inject_synthetic_anomalies_errors():
    with pytest.raises(ValueError):
        me.inject_synthetic_anomalies(pd.DataFrame({"only_col": [1, 2]}))
    with pytest.raises(ValueError):
        me.inject_synthetic_anomalies(make_features().iloc[0:0])


def test_evaluate_synthetic_separates_well(detector):
    report = me.evaluate_synthetic(detector, make_features(80, seed=1), factor=10.0)
    assert report["mode"] == "synthetic" and report["caveat"]
    assert report["roc_auc"] > 0.8          # a 10x spike should be easy to separate
    assert 0.0 <= report["recall_at_top_k"] <= 1.0
    assert report["n_synthetic_anomalies"] > 0


def test_evaluate_synthetic_requires_trained_model():
    with pytest.raises(ModelNotTrainedError):
        me.evaluate_synthetic(AnomalyDetector(), make_features())


def test_evaluate_with_labels(detector):
    X = make_features(60, seed=2)
    # label the 3 rows with the most errors as "anomalies" for this synthetic ground truth
    y = (X["error_count"] >= X["error_count"].nlargest(3).min()).astype(int).to_numpy()
    report = me.evaluate_with_labels(detector, X, y)
    assert report["mode"] == "labeled" and report["n_positive"] == int(y.sum())
    for key in ("precision", "recall", "f1", "accuracy"):
        assert 0.0 <= report[key] <= 1.0
    assert report["confusion_matrix"]["tp"] + report["confusion_matrix"]["fn"] == int(y.sum())


def test_evaluate_with_labels_validation(detector):
    X = make_features(10)
    with pytest.raises(ValueError):
        me.evaluate_with_labels(detector, X, [0, 1])                  # wrong length
    with pytest.raises(ValueError):
        me.evaluate_with_labels(detector, X, [2] * len(X))            # not binary
    with pytest.raises(ModelNotTrainedError):
        me.evaluate_with_labels(AnomalyDetector(), X, [0] * len(X))


def test_unsupervised_summary(detector):
    summary = me.unsupervised_summary(detector, make_features(30, seed=3))
    assert summary["mode"] == "unsupervised" and summary["n_samples"] == 30
    assert 0.0 <= summary["anomaly_rate"] <= 1.0


def test_full_report_with_and_without_labels(detector):
    X = make_features(50, seed=4)
    report = me.full_report(detector, X)
    assert set(report) == {"unsupervised", "synthetic"}

    y = np.zeros(len(X), dtype=int)
    y[:3] = 1
    report2 = me.full_report(detector, X, y_true=y)
    assert set(report2) == {"unsupervised", "synthetic", "labeled"}


def test_load_labels_partial_coverage_and_mismatch(tmp_path, caplog):
    X = make_features(20, seed=5)
    labels = pd.DataFrame({"time_window": X["time_window"].iloc[:10], "label": [0, 1] * 5})
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)

    merged, y = me._load_labels(labels_path, X)
    assert len(merged) == 10 and len(y) == 10

    empty_labels = tmp_path / "empty_labels.csv"
    pd.DataFrame({"time_window": [pd.Timestamp("1999-01-01")], "label": [0]}).to_csv(empty_labels, index=False)
    with pytest.raises(ValueError):
        me._load_labels(empty_labels, X)

    bad_cols = tmp_path / "bad.csv"
    pd.DataFrame({"time_window": [1], "wrong_col": [0]}).to_csv(bad_cols, index=False)
    with pytest.raises(ValueError):
        me._load_labels(bad_cols, X)


def test_model_evaluation_run_script(tmp_path):
    features_path = tmp_path / "features.csv"
    make_features(60, seed=6).to_csv(features_path, index=False)
    model_path = tmp_path / "m.pkl"
    AnomalyDetector(contamination=0.05).train(pd.read_csv(features_path)).save(model_path)

    out_path = tmp_path / "report.json"
    report = me.run(features_path, model_path, out_path=out_path)
    assert out_path.exists()
    assert json.loads(out_path.read_text()) == report

    with pytest.raises(FileNotFoundError):
        me.run(tmp_path / "missing.csv", model_path)


def test_model_evaluation_run_with_labels(tmp_path):
    X = make_features(40, seed=7)
    features_path = tmp_path / "features.csv"
    X.to_csv(features_path, index=False)
    model_path = tmp_path / "m.pkl"
    AnomalyDetector(contamination=0.05).train(X).save(model_path)

    labels_path = tmp_path / "labels.csv"
    pd.DataFrame({"time_window": X["time_window"], "label": 0}).to_csv(labels_path, index=False)

    report = me.run(features_path, model_path, labels_path=labels_path)
    assert "labeled" in report and report["labeled"]["n_samples"] == 40


# ================================================================== rag_evaluation
class FakeEmbeddingService:
    model = DeterministicFakeEmbedding(size=64)


@pytest.fixture(scope="module")
def rag_index(tmp_path_factory):
    """Real knowledge_base/*.md -> real chunker -> fake-embedded FAISS index,
    saved to a temp folder so load_retriever() can load it like a real one."""
    documents = DocumentLoader(KB_DIR).load_documents()
    chunks = DocumentChunker(chunk_size=400, overlap=40).chunk_documents(documents)
    store = VectorStore(FakeEmbeddingService())
    store.build(chunks)
    index_dir = tmp_path_factory.mktemp("rag_index")
    store.save(index_dir)
    return index_dir


def test_load_eval_set_default_and_custom(tmp_path):
    default = re.load_eval_set(None)
    assert len(default) == len(re.DEFAULT_EVAL_SET)
    assert all({"query", "expected_source"} <= row.keys() for row in default)

    custom_path = tmp_path / "custom.json"
    custom_path.write_text(json.dumps([{"query": "x", "expected_source": "y.md"}]))
    assert re.load_eval_set(custom_path) == [{"query": "x", "expected_source": "y.md"}]

    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps([{"query": "x"}]))       # missing expected_source
    with pytest.raises(ValueError):
        re.load_eval_set(bad_path)

    empty_path = tmp_path / "empty.json"
    empty_path.write_text("[]")
    with pytest.raises(ValueError):
        re.load_eval_set(empty_path)


def test_evaluate_retriever_on_real_knowledge_base(rag_index):
    # Dense scores are meaningless with a fake embedding model, so force pure BM25
    # ranking here - this evaluates the runbook CONTENT and chunking, not embeddings.
    retriever = re.load_retriever(rag_index, embedding_service=FakeEmbeddingService(),
                                  dense_weight=0.0, sparse_weight=1.0,
                                  max_distance=float("inf"), min_bm25_score=0.0)
    report = re.evaluate_retriever(retriever, re.DEFAULT_EVAL_SET, top_k=3)

    assert report["n_queries"] == len(re.DEFAULT_EVAL_SET)
    assert 0.0 <= report["top_1_accuracy"] <= 1.0
    assert 0.0 <= report["top_k_recall"] <= 1.0
    assert 0.0 <= report["mrr"] <= 1.0
    # real, keyword-rich runbooks against their own vocabulary should retrieve
    # reasonably well - loose bounds since these are genuine (not rigged) docs
    assert report["top_1_accuracy"] >= 0.5
    assert report["top_k_recall"] >= 0.8
    assert all({"query", "expected_source", "retrieved_sources", "rank"} <= r.keys()
              for r in report["per_query"])


def test_evaluate_retriever_validation(rag_index):
    retriever = re.load_retriever(rag_index, embedding_service=FakeEmbeddingService(), max_distance=float("inf"))
    with pytest.raises(ValueError):
        re.evaluate_retriever(retriever, [])
    with pytest.raises(ValueError):
        re.evaluate_retriever(retriever, re.DEFAULT_EVAL_SET, top_k=0)


def test_evaluate_retriever_handles_bad_query_gracefully(rag_index):
    retriever = re.load_retriever(rag_index, embedding_service=FakeEmbeddingService(),
                                  dense_weight=0.0, sparse_weight=1.0, max_distance=float("inf"))
    eval_set = [{"query": "", "expected_source": "hdfs_disk_failure.md"}]  # retrieve() rejects blank queries
    report = re.evaluate_retriever(retriever, eval_set, top_k=3)
    assert report["per_query"][0]["retrieved_sources"] == [] and report["per_query"][0]["rank"] is None


def test_rag_evaluation_run_script(rag_index, tmp_path):
    out_path = tmp_path / "rag_report.json"
    report = re.run(rag_index, top_k=3, out_path=out_path, embedding_service=FakeEmbeddingService())
    assert out_path.exists()
    assert json.loads(out_path.read_text())["n_queries"] == report["n_queries"]


def test_rag_evaluation_run_missing_index(tmp_path):
    with pytest.raises(FileNotFoundError):
        re.run(tmp_path / "no_such_index", embedding_service=FakeEmbeddingService())