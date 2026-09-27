import pandas as pd


def parse_hdfs_timestamps(date, time):
    """
    Combine HDFS Date (YYMMDD) and Time (HHMMSS) into datetimes.
    Tolerates ints, floats like 81109.0, stray spaces and NaN; bad values -> NaT.
    """
    def normalize(series):
        return (
            series.astype(str)
            .str.strip()
            .str.replace(r"\.0+$", "", regex=True)
            .str.zfill(6)
        )

    return pd.to_datetime(
        normalize(date) + normalize(time),
        format="%y%m%d%H%M%S",
        errors="coerce",
    )


class DataValidator:

    REQUIRED_COLUMNS = [
        "LineId", "Date", "Time", "Pid", "Level",
        "Component", "Content", "EventId", "EventTemplate",
    ]

    VALID_LOG_LEVELS = {"INFO", "WARN", "WARNING", "ERROR", "FATAL", "DEBUG"}

    def validate_logs(self, df):
        """
        Returns a report with:
          is_valid   : no errors and no warnings
          can_process: no blocking errors (warnings are fixed by the preprocessor)
          errors     : blocking problems (missing columns, empty data, no valid timestamps)
          warnings   : fixable problems (duplicates, bad levels, some bad timestamps, NaNs)
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame.")

        missing_columns = self.check_required_columns(df)
        errors = []

        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")

        if df.empty:
            errors.append("DataFrame is empty.")
            return self._report(errors, [], missing_columns, {}, 0, 0, 0, 0)

        missing_values = self.check_missing_values(df)
        duplicate_rows = self.check_duplicates(df)
        duplicate_line_ids = self.check_duplicate_line_ids(df)
        invalid_timestamps = self.check_timestamps(df)
        invalid_log_levels = self.check_log_levels(df)

        if invalid_timestamps == len(df) and {"Date", "Time"} <= set(df.columns):
            errors.append("No valid timestamps found.")

        warnings = []
        if missing_values:
            warnings.append(f"Missing values: {missing_values}")
        if duplicate_rows:
            warnings.append(f"{duplicate_rows} duplicate rows")
        if duplicate_line_ids:
            warnings.append(f"{duplicate_line_ids} duplicate LineIds")
        if invalid_timestamps and invalid_timestamps < len(df):
            warnings.append(f"{invalid_timestamps} invalid timestamps")
        if invalid_log_levels:
            warnings.append(f"{invalid_log_levels} invalid log levels")

        return self._report(errors, warnings, missing_columns, missing_values,
                            duplicate_rows, duplicate_line_ids,
                            invalid_timestamps, invalid_log_levels)

    @staticmethod
    def _report(errors, warnings, missing_columns, missing_values,
                duplicate_rows, duplicate_line_ids,
                invalid_timestamps, invalid_log_levels):
        return {
            "is_valid": not errors and not warnings,
            "can_process": not errors,
            "errors": errors,
            "warnings": warnings,
            "missing_columns": missing_columns,
            "missing_values": missing_values,
            "duplicate_rows": duplicate_rows,
            "duplicate_line_ids": duplicate_line_ids,
            "invalid_timestamps": invalid_timestamps,
            "invalid_log_levels": invalid_log_levels,
        }

    def check_required_columns(self, df):
        return [c for c in self.REQUIRED_COLUMNS if c not in df.columns]

    def check_missing_values(self, df):
        counts = df.isnull().sum()
        return {col: int(n) for col, n in counts[counts > 0].items()}

    def check_duplicates(self, df):
        return int(df.duplicated().sum())

    def check_duplicate_line_ids(self, df):
        if "LineId" not in df.columns:
            return 0
        return int(df["LineId"].duplicated().sum())

    def check_timestamps(self, df):
        if "Date" not in df.columns or "Time" not in df.columns:
            return 0
        return int(parse_hdfs_timestamps(df["Date"], df["Time"]).isna().sum())

    def check_log_levels(self, df):
        if "Level" not in df.columns:
            return 0
        levels = df["Level"].dropna().astype(str).str.strip().str.upper()
        return int((~levels.isin(self.VALID_LOG_LEVELS)).sum())

    def validate(self, df):
        return self.validate_logs(df)