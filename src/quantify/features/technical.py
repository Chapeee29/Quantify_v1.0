from __future__ import annotations

import numpy as np
import pandas as pd


def add_technical_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["code", "date"])

    def per_stock(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        close = g["close"].astype(float)
        high = g["high"].astype(float)
        low = g["low"].astype(float)
        open_ = g["open"].astype(float)
        volume = g["volume"].astype(float)

        g["ret_1"] = close.pct_change()
        for window in (5, 20, 60):
            g[f"ret_{window}"] = close.pct_change(window)
            ma = close.rolling(window, min_periods=max(2, window // 3)).mean()
            g[f"ma_gap_{window}"] = close / ma - 1
        for window in (5, 20):
            g[f"volatility_{window}"] = g["ret_1"].rolling(window, min_periods=2).std()
            vol_ma = volume.rolling(window, min_periods=2).mean()
            g[f"volume_ratio_{window}"] = volume / vol_ma - 1

        body = (close - open_).abs()
        full_range = (high - low).replace(0, np.nan)
        upper = high - np.maximum(open_, close)
        lower = np.minimum(open_, close) - low
        g["k_body_ratio"] = body / full_range
        g["k_upper_shadow_ratio"] = upper / full_range
        g["k_lower_shadow_ratio"] = lower / full_range
        g["close_position"] = (close - low) / full_range
        g["price_volume_corr_20"] = g["ret_1"].rolling(20, min_periods=8).corr(volume.pct_change())
        g["breakout_20"] = close / close.rolling(20, min_periods=5).max() - 1
        g["drawdown_20"] = close / close.rolling(20, min_periods=5).max() - 1
        g["amplitude"] = high / low.replace(0, np.nan) - 1
        return g

    return pd.concat([per_stock(group) for _, group in frame.groupby("code", sort=False)], ignore_index=True)
