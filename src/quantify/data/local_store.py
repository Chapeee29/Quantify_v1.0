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
        return pd.read_csv(path, parse_dates=parse_dates)

    def write_csv(self, name: str, frame: pd.DataFrame) -> Path:
        path = self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def required(self, name: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
        frame = self.read_csv(name, parse_dates=parse_dates)
        if frame.empty:
            raise FileNotFoundError(f"Missing or empty data file: {self.path(name)}")
        return frame
