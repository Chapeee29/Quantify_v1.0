from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class AkShareSource:
    """Thin AkShare adapter.

    The exact public AkShare endpoints can drift, so imports and calls are kept
    here instead of leaking endpoint assumptions through the feature pipeline.
    """

    adjust: str = "qfq"

    def _ak(self):
        try:
            import akshare as ak  # type: ignore
        except ImportError as exc:
            raise RuntimeError("akshare is not installed. Run `pip install -r requirements.txt`.") from exc
        return ak

    def stock_list(self) -> pd.DataFrame:
        ak = self._ak()
        raw = ak.stock_info_a_code_name()
        return raw.rename(columns={"code": "code", "name": "name"})[["code", "name"]]

    def stock_daily(self, code: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
        ak = self._ak()
        start = start_date.replace("-", "")
        end = (end_date or pd.Timestamp.today().strftime("%Y-%m-%d")).replace("-", "")
        raw = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust=self.adjust)
        mapping = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover_rate",
        }
        frame = raw.rename(columns=mapping)
        frame["code"] = code
        keep = [c for c in ["date", "code", "open", "high", "low", "close", "volume", "amount", "turnover_rate"] if c in frame]
        frame = frame[keep].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        return frame

    def index_daily(self, symbol: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
        ak = self._ak()
        start = start_date.replace("-", "")
        end = (end_date or pd.Timestamp.today().strftime("%Y-%m-%d")).replace("-", "")
        raw = ak.stock_zh_index_daily_em(symbol=symbol, start_date=start, end_date=end)
        mapping = {
            "date": "date",
            "open": "open",
            "close": "close",
            "high": "high",
            "low": "low",
            "volume": "volume",
            "amount": "amount",
        }
        frame = raw.rename(columns=mapping)
        frame["code"] = symbol
        keep = [c for c in ["date", "code", "open", "high", "low", "close", "volume", "amount"] if c in frame]
        frame = frame[keep].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        return frame
