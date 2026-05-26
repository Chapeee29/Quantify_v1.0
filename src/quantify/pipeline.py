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

REPORT_CONTEXT_COLUMNS = [
    "market_up_ratio",
    "market_mean_ret_1",
    "market_median_ret_1",
    "market_total_amount",
    "market_amount_ratio_5",
    "mkt_sh000001_ret_1",
    "mkt_sh000300_ret_1",
]


def load_raw_frames(config: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    store = LocalStore(config.paths.data_dir)
    stock_daily = store.required("stocks/daily.csv", parse_dates=["date"])
    stock_info = store.read_csv("stocks/info.csv")
    if not stock_info.empty:
        merge_cols = [c for c in ["code", "name", "sector", "market_cap", "list_date"] if c in stock_info.columns]
        stock_daily = stock_daily.merge(stock_info[merge_cols].drop_duplicates("code"), on="code", how="left")
    sector_map = store.read_csv("stocks/sector_map.csv")
    if "sector" not in stock_daily.columns and not sector_map.empty:
        stock_daily = stock_daily.merge(sector_map[["code", "sector"]].drop_duplicates("code"), on="code", how="left")
    stock_list = store.read_csv("stocks/list.csv")
    if "name" not in stock_daily.columns and not stock_list.empty:
        stock_daily = stock_daily.merge(stock_list[["code", "name"]].drop_duplicates("code"), on="code", how="left")
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
    arrays = build_sequence_arrays(
        scaled,
        sequence_cols,
        static_cols,
        int(preprocessor["sequence_length"]),
        require_label=False,
    )
    return restore_report_context(arrays, frame)


def restore_report_context(arrays, raw_feature_frame: pd.DataFrame):
    available = [column for column in REPORT_CONTEXT_COLUMNS if column in raw_feature_frame.columns]
    if not available or arrays.meta.empty:
        return arrays
    context = raw_feature_frame[["date", "code", *available]].drop_duplicates(["date", "code"])
    report_names = {
        "mkt_sh000001_ret_1": "sh_index_ret_1",
        "mkt_sh000300_ret_1": "hs300_ret_1",
    }
    context = context.rename(columns=report_names)
    replace_columns = [report_names.get(column, column) for column in available]
    base = arrays.meta.drop(columns=[column for column in replace_columns if column in arrays.meta.columns])
    arrays.meta = base.merge(context, on=["date", "code"], how="left", sort=False)
    return arrays
