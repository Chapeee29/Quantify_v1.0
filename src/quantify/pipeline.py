from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantify.config import AppConfig, ensure_runtime_dirs
from quantify.data.local_store import LocalStore
from quantify.features.dataset import (
    apply_scalers,
    build_feature_frame,
    build_sequence_arrays,
    feature_columns,
    fit_scalers,
)
from quantify.models.artifacts import load_preprocessor, save_preprocessor


def load_raw_frames(config: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    store = LocalStore(config.paths.data_dir)
    stock_daily = store.required("stocks/daily.csv", parse_dates=["date"])
    index_daily = store.read_csv("market/index_daily.csv", parse_dates=["date"])
    sector_daily = store.read_csv("market/sector_daily.csv", parse_dates=["date"])
    fundamentals = store.read_csv("fundamentals/features.csv", parse_dates=["date"])
    return stock_daily, index_daily, sector_daily, fundamentals


def prepare_training_arrays(config: AppConfig):
    ensure_runtime_dirs(config)
    stock_daily, index_daily, sector_daily, fundamentals = load_raw_frames(config)
    frame = build_feature_frame(stock_daily, index_daily, sector_daily, fundamentals, config)
    sequence_cols, static_cols = feature_columns(frame)
    seq_scaler, static_scaler = fit_scalers(frame, config.train.train_end, sequence_cols, static_cols)
    scaled = apply_scalers(frame, sequence_cols, static_cols, seq_scaler, static_scaler)
    arrays = build_sequence_arrays(scaled, sequence_cols, static_cols, config.features.sequence_length, require_label=True)
    save_preprocessor(
        Path(config.paths.model_dir) / "preprocessor.pkl",
        {
            "sequence_columns": sequence_cols,
            "static_columns": static_cols,
            "sequence_scaler": seq_scaler,
            "static_scaler": static_scaler,
            "sequence_length": config.features.sequence_length,
        },
    )
    return arrays


def prepare_prediction_arrays(config: AppConfig):
    ensure_runtime_dirs(config)
    stock_daily, index_daily, sector_daily, fundamentals = load_raw_frames(config)
    frame = build_feature_frame(stock_daily, index_daily, sector_daily, fundamentals, config)
    preprocessor = load_preprocessor(Path(config.paths.model_dir) / "preprocessor.pkl")
    sequence_cols = list(preprocessor["sequence_columns"])
    static_cols = list(preprocessor["static_columns"])
    for col in sequence_cols + static_cols:
        if col not in frame:
            frame[col] = 0.0
    scaled = apply_scalers(
        frame,
        sequence_cols,
        static_cols,
        preprocessor["sequence_scaler"],
        preprocessor["static_scaler"],
    )
    return build_sequence_arrays(
        scaled,
        sequence_cols,
        static_cols,
        int(preprocessor["sequence_length"]),
        require_label=False,
    )
