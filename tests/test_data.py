import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DataLoader, DataValidator, DataPreprocessor
from scripts.prepare_data import prepare


def sample_df():
    return pd.DataFrame({
        "LineId": [1, 2, 3, 4],
        "Date": ["081109", "081109", "081109", "081109"],
        "Time": ["203615", "203616", "204010", "204011"],
        "Pid": [143, 143, 35, 36],
        "Level": ["INFO", "ERROR", "warning", "FATAL"],
        "Component": ["dfs.A", "dfs.B", "dfs.A", "dfs.C"],
        "Content": ["Receiving block blk_1", "Heartbeat block failed",
                    "Connection timeout", "disk volume failure"],
        "EventId": ["E1", "E2", "E3", "E4"],
        "EventTemplate": ["t1", "t2", "t3", "t4"],
    })


@pytest.fixture
def data_dir(tmp_path):
    sample_df().to_csv(tmp_path / "HDFS_2k.log_structured.csv", index=False)
    pd.DataFrame({"EventId": ["E1"], "EventTemplate": ["t1"]}).to_csv(
        tmp_path / "HDFS_2k.log_templates.csv", index=False)
    (tmp_path / "HDFS_2k.log").write_text("081109 203615 143 INFO x\n")
    return tmp_path


# ---------------- loader ----------------
def test_loader_keeps_leading_zeros(data_dir):
    df = DataLoader(data_dir).load_structured_logs()
    assert df["Date"].iloc[0] == "081109"

def test_loader_missing_dir_and_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        DataLoader(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        DataLoader(tmp_path).load_structured_logs()

def test_loader_empty_file(tmp_path):
    (tmp_path / "HDFS_2k.log_structured.csv").write_text("")
    with pytest.raises(ValueError):
        DataLoader(tmp_path).load_structured_logs()

def test_load_all_non_strict(data_dir):
    (data_dir / "HDFS_2k.log").unlink()
    with pytest.raises(FileNotFoundError):
        DataLoader(data_dir).load_all(strict=True)
    result = DataLoader(data_dir).load_all(strict=False)
    assert result["raw_logs"] is None and result["templates"] is not None


# ---------------- validator ----------------
def test_validator_ok_and_warnings():
    v = DataValidator()
    r = v.validate(sample_df())
    assert r["is_valid"] and r["can_process"]

    df = sample_df()
    df.loc[0, "Level"] = "BOGUS"
    df.loc[1, "Time"] = "999999"
    r = v.validate(df)
    assert not r["is_valid"] and r["can_process"]
    assert r["invalid_log_levels"] == 1 and r["invalid_timestamps"] == 1

def test_validator_blocking_errors():
    v = DataValidator()
    assert not v.validate(sample_df().drop(columns=["Level"]))["can_process"]
    assert not v.validate(sample_df().iloc[0:0])["can_process"]
    df = sample_df(); df["Date"] = "garbage"
    assert not v.validate(df)["can_process"]
    with pytest.raises(TypeError):
        v.validate([1, 2])

def test_validator_numeric_dates():
    df = sample_df()
    df["Date"] = [81109.0] * 4        # float, leading zero lost
    df["Time"] = [203615, 203616, 204010, 204011]
    assert DataValidator().validate(df)["invalid_timestamps"] == 0


# ---------------- preprocessor ----------------
def test_features_and_signals():
    p = DataPreprocessor()
    logs = p.preprocess_logs(sample_df())
    assert set(logs["Level"]) == {"INFO", "ERROR", "WARN", "FATAL"}   # normalized
    f = p.create_features(logs)
    assert len(f) == 2                                   # 20:35 and 20:40 windows
    w = f.iloc[0]
    assert w["logs_per_window"] == 2 and w["error_count"] == 1
    assert w["heartbeat_failures"] == 1 and w["block_errors"] == 1
    assert f.iloc[1]["fatal_count"] == 1 and f.iloc[1]["disk_errors"] == 1
    assert w["error_ratio"] == 0.5

def test_preprocess_edge_cases():
    p = DataPreprocessor()
    df = sample_df()
    df.loc[0, "Level"] = None                                    # NaN level
    df.loc[1, "Content"] = None                                  # NaN content
    df = pd.concat([df, df.iloc[[0]]])                           # duplicate row
    logs = p.preprocess_logs(df)
    assert len(logs) == 4 and "UNKNOWN" in set(logs["Level"])

def test_all_timestamps_invalid_returns_empty():
    df = sample_df(); df["Date"] = "bad"
    f = DataPreprocessor().preprocess(df)
    assert f.empty and "error_ratio" in f.columns

def test_preprocessor_input_errors():
    p = DataPreprocessor()
    with pytest.raises(ValueError):
        p.preprocess(sample_df().drop(columns=["Pid"]))
    with pytest.raises(TypeError):
        p.preprocess("nope")
    with pytest.raises(ValueError):
        p.preprocess(sample_df().iloc[0:0])

def test_input_not_mutated():
    df = sample_df(); before = df.copy()
    DataPreprocessor().preprocess(df)
    pd.testing.assert_frame_equal(df, before)


# ---------------- prepare_data script ----------------
def test_prepare_data_writes_outputs(data_dir, tmp_path):
    out = tmp_path / "processed"
    logs, features, report = prepare(data_dir, out)
    assert len(features) == 2 and len(logs) == 4
    assert (out / "logs.csv").exists() and (out / "features.csv").exists()

def test_prepare_data_strict_and_blocking(data_dir, tmp_path):
    df = sample_df(); df.loc[0, "Level"] = "BOGUS"
    df.to_csv(data_dir / "HDFS_2k.log_structured.csv", index=False)
    prepare(data_dir, tmp_path / "o1")                          # warns, continues
    with pytest.raises(ValueError):
        prepare(data_dir, tmp_path / "o2", strict=True)
    sample_df().drop(columns=["Level"]).to_csv(
        data_dir / "HDFS_2k.log_structured.csv", index=False)
    with pytest.raises(ValueError):
        prepare(data_dir, tmp_path / "o3")