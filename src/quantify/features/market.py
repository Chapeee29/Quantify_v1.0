from __future__ import annotations

import pandas as pd


def add_market_features(index_daily: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    if index_daily is None or index_daily.empty:
        return pd.DataFrame()
    frame = index_daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["code", "date"])
    outputs = []
    for code, g in frame.groupby("code"):
        g = g.copy()
        close = g["close"].astype(float)
        amount = g.get("amount", pd.Series(index=g.index, dtype=float)).astype(float)
        out = pd.DataFrame({"date": g["date"]})
        prefix = f"mkt_{code}"
        for window in windows:
            min_periods = min(2, window)
            out[f"{prefix}_ret_{window}"] = close.pct_change(window)
            out[f"{prefix}_amount_ratio_{window}"] = amount / amount.rolling(window, min_periods=min_periods).mean() - 1
        out[f"{prefix}_volatility_20"] = close.pct_change().rolling(20, min_periods=5).std()
        outputs.append(out)
    result = outputs[0]
    for item in outputs[1:]:
        result = result.merge(item, on="date", how="outer")
    return result.sort_values("date")


def add_sector_features(sector_daily: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    if sector_daily is None or sector_daily.empty:
        return pd.DataFrame()
    frame = sector_daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["sector", "date"])
    outputs = []
    for sector, g in frame.groupby("sector"):
        g = g.copy()
        close = g["close"].astype(float)
        amount = g.get("amount", pd.Series(index=g.index, dtype=float)).astype(float)
        out = pd.DataFrame({"date": g["date"], "sector": sector})
        for window in windows:
            min_periods = min(2, window)
            out[f"sector_ret_{window}"] = close.pct_change(window)
            out[f"sector_amount_ratio_{window}"] = amount / amount.rolling(window, min_periods=min_periods).mean() - 1
        out["sector_up_ratio"] = pd.to_numeric(g.get("up_ratio", 0), errors="coerce")
        outputs.append(out)
    result = pd.concat(outputs, ignore_index=True)
    if not result.empty and "sector_ret_20" in result.columns:
        result["sector_rank_20"] = result.groupby("date")["sector_ret_20"].rank(pct=True)
    return result.sort_values(["sector", "date"])
