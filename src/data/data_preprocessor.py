import logging

import pandas as pd

from .data_validator import parse_hdfs_timestamps

logger = logging.getLogger(__name__)


class DataPreprocessor:

    # Columns the preprocessing steps actually use
    REQUIRED_COLUMNS = ["LineId", "Date", "Time", "Pid", "Level",
                        "Component", "Content", "EventId"]

    # signal name -> (regex on lower-cased Content, error-column name)
    SIGNALS = {
        "heartbeat": ("heartbeat", "heartbeat_failure"),
        "block": ("block|blk_", "block_error"),
        "replication": ("replicat", "replication_error"),
        "network": ("network|connection|socket|timeout|timed out", "network_error"),
        "disk": ("disk|storage|volume", "disk_error"),
    }

    FEATURE_COLUMNS = [
        "time_window", "logs_per_window",
        "error_count", "warning_count", "info_count", "fatal_count",
        "heartbeat_failures", "block_errors", "replication_errors",
        "network_errors", "disk_errors",
        "unique_components", "unique_event_ids", "unique_processes",
        "error_ratio", "warning_ratio", "fatal_ratio",
    ]

    def __init__(self, window="5min"):
        """window: pandas frequency for time windows, e.g. "5min" or "1min"."""
        try:
            pd.tseries.frequencies.to_offset(window)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid window: {window!r}") from exc
        self.window = window

    def _require(self, df, columns):
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    # ------------------------------------------------------------------
    def clean_logs(self, df):
        """Drop empty rows, exact duplicate rows and repeated LineIds."""
        df = df.copy()
        df = df.dropna(how="all").drop_duplicates()
        if "LineId" in df.columns:
            df = df.drop_duplicates(subset="LineId", keep="first")
        return df

    def parse_timestamp(self, df):
        """YYMMDD + HHMMSS -> Timestamp, hour, minute, second (bad values -> NaT)."""
        self._require(df, ["Date", "Time"])
        df = df.copy()
        df["Timestamp"] = parse_hdfs_timestamps(df["Date"], df["Time"])
        df["hour"] = df["Timestamp"].dt.hour
        df["minute"] = df["Timestamp"].dt.minute
        df["second"] = df["Timestamp"].dt.second
        return df

    def normalize_log_level(self, df):
        """Upper-case levels; WARNING -> WARN; missing/blank -> UNKNOWN."""
        self._require(df, ["Level"])
        df = df.copy()
        df["Level"] = (
            df["Level"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
            .replace({"": "UNKNOWN", "WARNING": "WARN"})
        )
        return df

    def extract_log_features(self, df):
        """Row-level level flags and incident signals."""
        self._require(df, ["Level", "Content"])
        df = df.copy()

        level = df["Level"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
        is_severe = level.isin(["ERROR", "FATAL"])

        df["is_error"] = (level == "ERROR").astype(int)
        df["is_warning"] = level.isin(["WARN", "WARNING"]).astype(int)
        df["is_info"] = (level == "INFO").astype(int)
        df["is_fatal"] = (level == "FATAL").astype(int)

        content = df["Content"].fillna("").astype(str).str.lower()

        for name, (pattern, error_col) in self.SIGNALS.items():
            signal = content.str.contains(pattern, regex=True, na=False)
            df[f"{name}_signal"] = signal.astype(int)
            df[error_col] = (is_severe & signal).astype(int)  # signal + ERROR/FATAL

        return df

    def create_time_windows(self, df):
        """Time windows (default 5 min); rows without a valid timestamp are dropped."""
        if "Timestamp" not in df.columns:
            raise ValueError("Timestamp column not found. Run parse_timestamp() first.")
        df = df.dropna(subset=["Timestamp"]).copy()
        df["time_window"] = df["Timestamp"].dt.floor(self.window)
        return df

    def create_features(self, df):
        """Aggregate per time window. Returns an empty frame (same columns) if no rows."""
        if df.empty:
            return pd.DataFrame(columns=self.FEATURE_COLUMNS)

        self._require(df, ["time_window", "LineId", "Component", "EventId", "Pid",
                           "is_error", "is_warning", "is_info", "is_fatal",
                           "heartbeat_failure", "block_error", "replication_error",
                           "network_error", "disk_error"])

        features = (
            df.groupby("time_window")
            .agg(
                logs_per_window=("LineId", "size"),  # size: safe even if LineId is NaN
                error_count=("is_error", "sum"),
                warning_count=("is_warning", "sum"),
                info_count=("is_info", "sum"),
                fatal_count=("is_fatal", "sum"),
                heartbeat_failures=("heartbeat_failure", "sum"),
                block_errors=("block_error", "sum"),
                replication_errors=("replication_error", "sum"),
                network_errors=("network_error", "sum"),
                disk_errors=("disk_error", "sum"),
                unique_components=("Component", "nunique"),
                unique_event_ids=("EventId", "nunique"),
                unique_processes=("Pid", "nunique"),
            )
            .reset_index()
            .sort_values("time_window")
            .reset_index(drop=True)
        )

        for name in ("error", "warning", "fatal"):
            features[f"{name}_ratio"] = (
                features[f"{name}_count"] / features["logs_per_window"]
            ).fillna(0)

        return features

    # ------------------------------------------------------------------
    def preprocess_logs(self, df):
        """Row-level pipeline (clean -> timestamp -> level -> signals -> windows)."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame.")
        self._require(df, self.REQUIRED_COLUMNS)

        n_input = len(df)
        df = self.clean_logs(df)
        if df.empty:
            raise ValueError("No rows left after cleaning.")

        df = self.parse_timestamp(df)
        df = self.normalize_log_level(df)
        df = self.extract_log_features(df)
        df = self.create_time_windows(df)

        if df.empty:
            logger.warning("No rows with a valid timestamp; nothing to window.")
        elif len(df) < n_input:
            logger.info("Dropped %d of %d rows during preprocessing.",
                        n_input - len(df), n_input)
        return df

    def preprocess(self, df):
        """Full pipeline -> ML-ready window features."""
        return self.create_features(self.preprocess_logs(df))