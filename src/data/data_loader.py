import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads the HDFS_2k raw log, structured CSV and template CSV."""

    RAW_LOG_FILE = "HDFS_2k.log"
    STRUCTURED_FILE = "HDFS_2k.log_structured.csv"
    TEMPLATES_FILE = "HDFS_2k.log_templates.csv"

    def __init__(self, data_path):
        self.data_path = Path(data_path).expanduser()

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")
        if not self.data_path.is_dir():
            raise NotADirectoryError(f"Expected a directory: {self.data_path}")

    def _resolve(self, filename):
        path = self.data_path / filename
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"File is empty: {path}")
        return path

    def _read_csv(self, filename):
        path = self._resolve(filename)
        try:
            # Date/Time kept as strings so leading zeros survive (081109, not 81109)
            df = pd.read_csv(
                path,
                dtype={"Date": str, "Time": str},
                encoding="utf-8",
                encoding_errors="replace",
            )
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            raise ValueError(f"Could not parse {path}: {exc}") from exc

        if df.empty:
            logger.warning("%s contains a header but no rows.", path.name)
        return df

    def load_logs(self):
        """Raw log file as one string."""
        path = self._resolve(self.RAW_LOG_FILE)
        return path.read_text(encoding="utf-8", errors="replace")

    def load_structured_logs(self):
        """Structured logs as a DataFrame."""
        return self._read_csv(self.STRUCTURED_FILE)

    def load_templates(self):
        """Log templates as a DataFrame."""
        return self._read_csv(self.TEMPLATES_FILE)

    def load_all(self, strict=True):
        """
        strict=True : any missing/bad file raises.
        strict=False: structured logs are still required, but a missing or
                      unreadable raw log / templates file becomes None.
        """
        structured = self.load_structured_logs()  # always required
        result = {"structured_logs": structured, "raw_logs": None, "templates": None}

        for key, loader in (("raw_logs", self.load_logs),
                            ("templates", self.load_templates)):
            try:
                result[key] = loader()
            except (FileNotFoundError, ValueError) as exc:
                if strict:
                    raise
                logger.warning("Optional data '%s' unavailable: %s", key, exc)

        return result