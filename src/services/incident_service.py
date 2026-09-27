import re

import numpy as np
import pandas as pd

from src.data import DataPreprocessor, DataValidator


class IncidentNotFoundError(LookupError):
    """Raised when an incident id does not exist (API: HTTP 404)."""


class IncidentService:
    """
    Turns the HDFS logs into "incidents".

    One incident = one time window of logs. Incidents are numbered in
    chronological order: INC-001 is the first window, INC-002 the next, ...
    Use the same window size the model was trained with (default 1 minute).
    """

    ID_PREFIX = "INC-"
    LOG_COLUMNS = ["LineId", "Timestamp", "Level", "Component", "Pid", "EventId", "Content"]
    SEVERE_LEVELS = ["WARN", "ERROR", "FATAL"]
    _ID_PATTERN = re.compile(r"^\s*(?:INC[-_ ]?)?0*(\d+)\s*$", re.IGNORECASE)

    def __init__(self, data_loader, preprocessor=None, validator=None):
        if data_loader is None:
            raise ValueError("data_loader is required.")
        self.data_loader = data_loader
        self.preprocessor = preprocessor or DataPreprocessor(window="1min")
        self.validator = validator or DataValidator()

        self._logs = None
        self._features = None
        self._groups = None
        self._similarity_matrix = None

    # ------------------------------------------------------------------
    def load(self, force=False):
        """Load + validate + preprocess once (cached)."""
        if self._logs is not None and not force:
            return self

        raw = self.data_loader.load_structured_logs()
        report = self.validator.validate(raw)
        if not report["can_process"]:
            raise ValueError(f"Cannot build incidents: {report['errors']}")

        logs = self.preprocessor.preprocess_logs(raw)
        features = self.preprocessor.create_features(logs)
        if features.empty:
            raise ValueError("No incidents found: no log lines with a valid timestamp.")

        self._logs = logs.sort_values(["Timestamp", "LineId"]).reset_index(drop=True)
        self._features = features.reset_index(drop=True)
        self._groups = self._logs.groupby("time_window").indices  # window -> row positions
        self._similarity_matrix = None
        return self

    @property
    def n_incidents(self):
        self.load()
        return len(self._features)

    # ------------------------------------------------------------------
    def _index(self, incident_id):
        """'INC-003' / 'inc-3' / 3  ->  0-based row index."""
        self.load()
        if isinstance(incident_id, bool) or not isinstance(incident_id, (str, int)):
            raise TypeError("incident_id must be a string like 'INC-001' or an integer.")
        if isinstance(incident_id, str) and not incident_id.strip():
            raise ValueError("incident_id cannot be empty.")

        match = self._ID_PATTERN.match(str(incident_id))
        number = int(match.group(1)) if match else 0
        if not match or not 1 <= number <= self.n_incidents:
            raise IncidentNotFoundError(
                f"Incident {incident_id!r} not found "
                f"(valid: {self.ID_PREFIX}001 .. {self.ID_PREFIX}{self.n_incidents:03d})."
            )
        return number - 1

    def normalize_id(self, incident_id):
        return f"{self.ID_PREFIX}{self._index(incident_id) + 1:03d}"

    def _window_logs(self, index):
        window = self._features.at[index, "time_window"]
        return self._logs.iloc[self._groups[window]]

    # ------------------------------------------------------------------
    def get_incident(self, incident_id):
        index = self._index(incident_id)
        start = self._features.at[index, "time_window"]
        logs = self._window_logs(index)
        levels = logs["Level"].value_counts().to_dict()
        severe = logs["Level"].isin(self.SEVERE_LEVELS)

        signals = {
            name: int(((logs[f"{name}_signal"] == 1) & severe).sum())
            for name in DataPreprocessor.SIGNALS
        }
        return {
            "incident_id": f"{self.ID_PREFIX}{index + 1:03d}",
            "window_start": start.isoformat(),
            "window_end": (start + pd.to_timedelta(self.preprocessor.window)).isoformat(),
            "n_logs": int(len(logs)),
            "level_counts": {k: int(v) for k, v in levels.items()},
            "n_errors": int(levels.get("ERROR", 0)),
            "n_warnings": int(levels.get("WARN", 0)),
            "n_fatal": int(levels.get("FATAL", 0)),
            "components": sorted(logs["Component"].dropna().astype(str).unique())[:10],
            "signals": signals,   # WARN/ERROR/FATAL lines per signal
        }

    def get_logs(self, incident_id, limit=None):
        """Log lines of the incident window as a DataFrame (chronological)."""
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            raise ValueError("limit must be a positive integer.")
        logs = self._window_logs(self._index(incident_id))
        cols = [c for c in self.LOG_COLUMNS if c in logs.columns]
        logs = logs[cols].sort_values(["Timestamp", "LineId"])
        return logs.head(limit).copy() if limit else logs.copy()

    def get_features(self, incident_id):
        """The incident window's feature row as a plain dict."""
        index = self._index(incident_id)
        row = self._features.iloc[index]
        out = {}
        for key, value in row.items():
            if key == "time_window":
                out[key] = value.isoformat()
            else:
                out[key] = value.item() if hasattr(value, "item") else value
        return out

    def list_incidents(self, limit=None, sort_by="time"):
        """Summary of every incident. sort_by: 'time' or 'severity'."""
        self.load()
        if sort_by not in ("time", "severity"):
            raise ValueError("sort_by must be 'time' or 'severity'.")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            raise ValueError("limit must be a positive integer.")

        f = self._features
        rows = [
            {
                "incident_id": f"{self.ID_PREFIX}{i + 1:03d}",
                "window_start": r["time_window"].isoformat(),
                "n_logs": int(r["logs_per_window"]),
                "fatal_count": int(r["fatal_count"]),
                "error_count": int(r["error_count"]),
                "warning_count": int(r["warning_count"]),
            }
            for i, r in f.iterrows()
        ]
        if sort_by == "severity":
            rows.sort(key=lambda r: (r["fatal_count"], r["error_count"], r["warning_count"]),
                      reverse=True)
        return rows[:limit] if limit else rows

    def find_similar_incidents(self, incident_id, top_k=3):
        """Windows with the most similar feature profile (excluding itself)."""
        index = self._index(incident_id)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        if len(self._features) < 2:
            return []

        if self._similarity_matrix is None:
            cols = [c for c in DataPreprocessor.FEATURE_COLUMNS if c != "time_window"]
            x = self._features[cols].to_numpy(dtype=float)
            std = x.std(axis=0)
            std[std == 0] = 1.0
            self._similarity_matrix = (x - x.mean(axis=0)) / std

        z = self._similarity_matrix
        distance = np.linalg.norm(z - z[index], axis=1)
        distance[index] = np.inf
        order = np.argsort(distance)[: min(top_k, len(z) - 1)]

        return [
            {
                "incident_id": f"{self.ID_PREFIX}{int(i) + 1:03d}",
                "similarity": round(float(1.0 / (1.0 + distance[i])), 4),
                "window_start": self._features.at[int(i), "time_window"].isoformat(),
                "n_logs": int(self._features.at[int(i), "logs_per_window"]),
                "error_count": int(self._features.at[int(i), "error_count"]),
            }
            for i in order
        ]