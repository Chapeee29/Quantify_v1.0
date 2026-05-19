from __future__ import annotations

import pandas as pd

from quantify.config import AppConfig
from quantify.features.dataset import (
    add_next_day_label,
    build_feature_frame,
    feature_columns,
    merge_fundamentals_asof,
)


def test_next_day_label_uses_future_close_only_as_target() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "code": ["000001", "000001", "000001"],
            "close": [10.0, 10.2, 10.0],
        }
    )
    result = add_next_day_label(frame, threshold=0.01)
    assert result.loc[0, "label"] == 1.0
    assert result.loc[1, "label"] == 0.0
    assert pd.isna(result.loc[2, "label"])


def test_feature_columns_exclude_future_target_fields() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "code": ["000001"],
            "close": [10.0],
            "future_close": [11.0],
            "next_return": [0.1],
            "label": [1.0],
            "ret_1": [0.02],
            "fund_roe": [0.1],
        }
    )
    sequence_cols, static_cols = feature_columns(frame)
    assert "future_close" not in sequence_cols + static_cols
    assert "next_return" not in sequence_cols + static_cols
    assert "label" not in sequence_cols + static_cols
    assert "ret_1" in sequence_cols
    assert "fund_roe" in static_cols


def test_fundamentals_are_merged_only_backward_by_effective_date() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-04-29", "2024-05-01", "2024-05-02"]),
            "code": ["000001", "000001", "000001"],
            "close": [10.0, 10.1, 10.2],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-05-01"]),
            "code": ["000001"],
            "feature": ["roe"],
            "value": [0.12],
            "source_sheet": ["test"],
        }
    )
    merged = merge_fundamentals_asof(prices, fundamentals)
    assert "fund_roe" in merged.columns
    assert pd.isna(merged.loc[0, "fund_roe"])
    assert merged.loc[1, "fund_roe"] == 0.12
    assert merged.loc[2, "fund_roe"] == 0.12


def test_build_feature_frame_adds_market_and_sector_without_target_leakage() -> None:
    config = AppConfig()
    config.features.sequence_length = 3
    stock = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
            "code": ["000001"] * 4,
            "name": ["A"] * 4,
            "sector": ["半导体"] * 4,
            "open": [10, 10.1, 10.2, 10.3],
            "high": [10.2, 10.3, 10.4, 10.5],
            "low": [9.9, 10.0, 10.1, 10.2],
            "close": [10, 10.2, 10.1, 10.4],
            "volume": [100, 120, 110, 140],
            "amount": [1000, 1224, 1111, 1456],
            "market_cap": [100] * 4,
        }
    )
    index = stock.assign(code="sh000001")
    sector = pd.DataFrame(
        {
            "date": stock["date"],
            "sector": ["半导体"] * 4,
            "close": [100, 101, 102, 104],
            "volume": [1000, 1200, 1100, 1400],
            "amount": [1e6, 1.2e6, 1.1e6, 1.4e6],
            "up_ratio": [0.5, 0.6, 0.4, 0.8],
        }
    )
    frame = build_feature_frame(stock, index, sector, pd.DataFrame(), config)
    sequence_cols, static_cols = feature_columns(frame)
    assert any(c.startswith("mkt_sh000001") for c in sequence_cols)
    assert any(c.startswith("sector_ret") for c in sequence_cols)
    assert "future_close" not in sequence_cols + static_cols
