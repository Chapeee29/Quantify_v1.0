from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quantify.data.local_store import LocalStore


def make_sample_data(data_dir: str | Path, seed: int = 42) -> None:
    """Create deterministic sample data for tests and smoke runs."""

    rng = np.random.default_rng(seed)
    store = LocalStore(data_dir)
    dates = pd.bdate_range("2024-01-02", periods=320)
    sectors = ["新能源", "半导体", "消费"]
    stocks = [f"00000{i}" for i in range(1, 7)]

    stock_rows = []
    info_rows = []
    for idx, code in enumerate(stocks):
        sector = sectors[idx % len(sectors)]
        drift = 0.0002 + idx * 0.00005
        noise = rng.normal(drift, 0.018, len(dates))
        if idx % 2 == 0:
            noise[120:160] += 0.004
        close = 10 + np.cumsum(noise)
        close = np.maximum(close, 1)
        open_ = close * (1 + rng.normal(0, 0.004, len(dates)))
        high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.02, len(dates)))
        low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.02, len(dates)))
        volume = rng.integers(80_000, 800_000, len(dates)) * (1 + idx * 0.1)
        amount = volume * close
        turnover = rng.uniform(0.5, 8.0, len(dates))
        for d, o, h, l, c, v, a, t in zip(dates, open_, high, low, close, volume, amount, turnover):
            stock_rows.append(
                {
                    "date": d,
                    "code": code,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                    "amount": a,
                    "turnover_rate": t,
                    "sector": sector,
                    "market_cap": float(50 + idx * 20),
                    "is_st": False,
                    "list_date": "2020-01-01",
                }
            )
        info_rows.append({"code": code, "name": f"样例{idx + 1}", "sector": sector})

    index_rows = []
    for symbol, base in [("sh000001", 3000), ("sz399001", 10000), ("sz399006", 2000)]:
        ret = rng.normal(0.0003, 0.009, len(dates))
        close = base + np.cumsum(ret * base / 10)
        for d, c in zip(dates, close):
            index_rows.append(
                {"date": d, "code": symbol, "open": c * 0.998, "high": c * 1.006, "low": c * 0.994, "close": c, "volume": 1e9, "amount": c * 1e8}
            )

    sector_rows = []
    stock_frame = pd.DataFrame(stock_rows)
    for (d, sector), g in stock_frame.groupby(["date", "sector"]):
        sector_rows.append(
            {
                "date": d,
                "sector": sector,
                "close": g["close"].mean(),
                "volume": g["volume"].sum(),
                "amount": g["amount"].sum(),
                "up_ratio": float((g["close"].pct_change().fillna(0) > 0).mean()),
            }
        )

    fundamental_rows = []
    for code in stocks:
        for d in pd.to_datetime(["2024-04-30", "2024-08-31", "2024-10-31"]):
            fundamental_rows.extend(
                [
                    {"date": d, "code": code, "feature": "roe", "value": rng.uniform(0.05, 0.2), "source_sheet": "sample"},
                    {"date": d, "code": code, "feature": "cash_quality", "value": rng.uniform(-0.2, 0.5), "source_sheet": "sample"},
                    {"date": d, "code": code, "feature": "debt_ratio", "value": rng.uniform(0.2, 0.8), "source_sheet": "sample"},
                ]
            )

    store.write_csv("stocks/daily.csv", stock_frame)
    store.write_csv("stocks/info.csv", pd.DataFrame(info_rows))
    store.write_csv("market/index_daily.csv", pd.DataFrame(index_rows))
    store.write_csv("market/sector_daily.csv", pd.DataFrame(sector_rows))
    store.write_csv("fundamentals/features.csv", pd.DataFrame(fundamental_rows))
