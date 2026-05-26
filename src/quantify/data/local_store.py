from __future__ import annotations

from pathlib import Path

import pandas as pd


class LocalStore:
    """CSV-backed local data store for the first version.

    CSV keeps the project easy to move to the remote Windows machine without
    requiring pyarrow. Large production runs can later switch this behind the
    same interface to Parquet or a database.
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def path(self, name: str) -> Path:
        return self.data_dir / name

    def read_csv(self, name: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
        path = self.path(name)
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_csv(path, parse_dates=parse_dates, dtype={"code": "string"})
        if "code" in frame.columns:
            frame["code"] = frame["code"].astype(str).str.zfill(6)
        return frame

    def write_csv(self, name: str, frame: pd.DataFrame) -> Path:
        path = self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = frame.copy()
        if "code" in frame.columns:
            frame["code"] = frame["code"].astype(str).str.zfill(6)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def upsert_csv(
        self,
        name: str,
        frame: pd.DataFrame,
        key_columns: list[str],
        parse_dates: list[str] | None = None,
    ) -> Path:
        existing = self.read_csv(name, parse_dates=parse_dates)
        incoming = frame.copy()
        if "code" in incoming.columns:
            incoming["code"] = incoming["code"].astype(str).str.zfill(6)
        for column in parse_dates or []:
            if column in incoming.columns:
                incoming[column] = pd.to_datetime(incoming[column])
        combined = pd.concat([existing, incoming], ignore_index=True) if not existing.empty else incoming
        combined = combined.drop_duplicates(subset=key_columns, keep="last").sort_values(key_columns)
        return self.write_csv(name, combined)

    def required(self, name: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
        frame = self.read_csv(name, parse_dates=parse_dates)
        if frame.empty:
            raise FileNotFoundError(f"Missing or empty data file: {self.path(name)}")
        return frame
