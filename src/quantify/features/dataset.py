from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from quantify.config import AppConfig
from quantify.features.market import add_market_features, add_sector_features
from quantify.features.technical import add_technical_features


ID_COLUMNS = {"date", "code", "name", "sector", "list_date", "target_date"}
TARGET_COLUMNS = {"future_close", "next_return", "label"}


@dataclass
class SequenceArrays:
    x_sequence: np.ndarray
    x_static: np.ndarray
    y: np.ndarray | None
    meta: pd.DataFrame
    sequence_columns: list[str]
    static_columns: list[str]


def add_next_day_label(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    frame = frame.sort_values(["code", "date"]).copy()
    frame["future_close"] = frame.groupby("code")["close"].shift(-1)
    frame["target_date"] = frame.groupby("code")["date"].shift(-1)
    frame["next_return"] = frame["future_close"] / frame["close"] - 1
    frame["label"] = (frame["next_return"] >= threshold).astype(float)
    frame.loc[frame["future_close"].isna(), "label"] = np.nan
    return frame


def _pivot_fundamentals(fundamentals: pd.DataFrame) -> pd.DataFrame:
    if fundamentals is None or fundamentals.empty:
        return pd.DataFrame()
    f = fundamentals.copy()
    f["date"] = pd.to_datetime(f["date"])
    f["value"] = pd.to_numeric(f["value"], errors="coerce")
    f = f.dropna(subset=["date", "code", "feature", "value"])
    pivot = f.pivot_table(index=["code", "date"], columns="feature", values="value", aggfunc="last").reset_index()
    pivot.columns = [str(c) for c in pivot.columns]
    feature_cols = [c for c in pivot.columns if c not in {"code", "date"}]
    pivot = pivot.rename(columns={c: f"fund_{c}" for c in feature_cols})
    return pivot.sort_values(["code", "date"])


def merge_fundamentals_asof(frame: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    fund = _pivot_fundamentals(fundamentals)
    if fund.empty:
        return frame
    outputs = []
    frame = frame.sort_values(["code", "date"])
    for code, g in frame.groupby("code"):
        f = fund[fund["code"] == code].drop(columns=["code"])
        if f.empty:
            outputs.append(g)
            continue
        merged = pd.merge_asof(g.sort_values("date"), f.sort_values("date"), on="date", direction="backward")
        outputs.append(merged)
    return pd.concat(outputs, ignore_index=True).sort_values(["code", "date"])


def build_feature_frame(
    stock_daily: pd.DataFrame,
    index_daily: pd.DataFrame | None,
    sector_daily: pd.DataFrame | None,
    fundamentals: pd.DataFrame | None,
    config: AppConfig,
) -> pd.DataFrame:
    frame = stock_daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = add_technical_features(frame)
    frame = add_next_day_label(frame, config.label.next_day_return_threshold)

    if config.features.include_market and index_daily is not None and not index_daily.empty:
        market = add_market_features(index_daily, config.features.market_windows)
        frame = frame.merge(market, on="date", how="left")

    if config.features.include_sector and sector_daily is not None and not sector_daily.empty and "sector" in frame:
        sector = add_sector_features(sector_daily, config.features.market_windows)
        frame = frame.merge(sector, on=["date", "sector"], how="left")

    if config.features.include_fundamentals and fundamentals is not None and not fundamentals.empty:
        frame = merge_fundamentals_asof(frame, fundamentals)

    if "sector" in frame.columns:
        dummies = pd.get_dummies(frame["sector"], prefix="sector_is", dtype=float)
        frame = pd.concat([frame, dummies], axis=1)

    numeric_cols = [c for c in frame.columns if c not in ID_COLUMNS and pd.api.types.is_numeric_dtype(frame[c])]
    frame[numeric_cols] = frame.groupby("code")[numeric_cols].ffill()
    frame[numeric_cols] = frame[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame.sort_values(["code", "date"]).reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [c for c in frame.columns if c not in ID_COLUMNS | TARGET_COLUMNS and pd.api.types.is_numeric_dtype(frame[c])]
    static_prefixes = ("fund_", "sector_is_", "market_cap", "is_st")
    static_cols = [c for c in numeric if c.startswith(static_prefixes)]
    sequence_cols = [c for c in numeric if c not in static_cols]
    return sequence_cols, static_cols


def fit_scalers(frame: pd.DataFrame, train_end: str, sequence_cols: list[str], static_cols: list[str]) -> tuple[StandardScaler, StandardScaler]:
    train = frame[pd.to_datetime(frame["date"]) <= pd.Timestamp(train_end)]
    seq_scaler = StandardScaler().fit(train[sequence_cols].astype(float))
    static_scaler = StandardScaler().fit(train[static_cols].astype(float)) if static_cols else StandardScaler().fit(np.zeros((len(train), 1)))
    return seq_scaler, static_scaler


def apply_scalers(frame: pd.DataFrame, sequence_cols: list[str], static_cols: list[str], seq_scaler: StandardScaler, static_scaler: StandardScaler) -> pd.DataFrame:
    result = frame.copy()
    result[sequence_cols] = seq_scaler.transform(result[sequence_cols].astype(float))
    if static_cols:
        result[static_cols] = static_scaler.transform(result[static_cols].astype(float))
    return result


def build_sequence_arrays(
    frame: pd.DataFrame,
    sequence_cols: list[str],
    static_cols: list[str],
    sequence_length: int,
    require_label: bool,
) -> SequenceArrays:
    sequences: list[np.ndarray] = []
    static_values: list[np.ndarray] = []
    labels: list[float] = []
    meta_rows: list[dict[str, object]] = []

    for code, g in frame.sort_values(["code", "date"]).groupby("code"):
        g = g.reset_index(drop=True)
        seq_matrix = g[sequence_cols].astype(float).to_numpy()
        static_matrix = g[static_cols].astype(float).to_numpy() if static_cols else np.zeros((len(g), 1), dtype=float)
        for end_idx in range(sequence_length - 1, len(g)):
            label = g.loc[end_idx, "label"]
            if require_label and pd.isna(label):
                continue
            sequences.append(seq_matrix[end_idx - sequence_length + 1 : end_idx + 1])
            static_values.append(static_matrix[end_idx])
            if not pd.isna(label):
                labels.append(float(label))
            meta_rows.append(
                {
                    "date": g.loc[end_idx, "date"],
                    "target_date": g.loc[end_idx, "target_date"],
                    "code": code,
                    "name": g.loc[end_idx, "name"] if "name" in g else code,
                    "sector": g.loc[end_idx, "sector"] if "sector" in g else "",
                    "close": g.loc[end_idx, "close"],
                    "next_return": g.loc[end_idx, "next_return"],
                }
            )

    y = np.array(labels, dtype=np.float32) if labels else None
    return SequenceArrays(
        x_sequence=np.array(sequences, dtype=np.float32),
        x_static=np.array(static_values, dtype=np.float32),
        y=y,
        meta=pd.DataFrame(meta_rows),
        sequence_columns=sequence_cols,
        static_columns=static_cols,
    )


def split_by_time(arrays: SequenceArrays, train_end: str, validation_end: str) -> dict[str, np.ndarray]:
    dates = pd.to_datetime(arrays.meta["date"])
    train_mask = dates <= pd.Timestamp(train_end)
    validation_mask = (dates > pd.Timestamp(train_end)) & (dates <= pd.Timestamp(validation_end))
    test_mask = dates > pd.Timestamp(validation_end)
    return {
        "train": train_mask.to_numpy(),
        "validation": validation_mask.to_numpy(),
        "test": test_mask.to_numpy(),
    }
