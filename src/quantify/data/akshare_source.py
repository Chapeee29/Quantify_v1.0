from __future__ import annotations

from dataclasses import dataclass
import re

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
        frame = raw.rename(columns={"code": "code", "name": "name"})[["code", "name"]].copy()
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        return frame

    def stock_daily(self, code: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
        try:
            return self._stock_daily_sina(code, start_date, end_date)
        except Exception:
            return self._stock_daily_eastmoney(code, start_date, end_date)

    def _stock_daily_eastmoney(self, code: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
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

    def _stock_daily_sina(self, code: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
        ak = self._ak()
        start = start_date.replace("-", "")
        end = (end_date or pd.Timestamp.today().strftime("%Y-%m-%d")).replace("-", "")
        symbol = self._exchange_symbol(code)
        raw = ak.stock_zh_a_daily(symbol=symbol, start_date=start, end_date=end, adjust=self.adjust)
        frame = raw.rename(columns={"turnover": "turnover_rate"})
        frame["code"] = str(code).zfill(6)
        keep = [c for c in ["date", "code", "open", "high", "low", "close", "volume", "amount", "turnover_rate"] if c in frame]
        frame = frame[keep].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        return frame

    @staticmethod
    def _exchange_symbol(code: str) -> str:
        code = str(code).zfill(6)
        if code.startswith(("6", "9")):
            return f"sh{code}"
        if code.startswith(("4", "8")):
            return f"bj{code}"
        return f"sz{code}"

    def industry_list(self) -> pd.DataFrame:
        ak = self._ak()
        try:
            raw = ak.stock_board_industry_name_ths()
            mapping = {"name": "sector", "code": "sector_code"}
        except Exception:
            raw = ak.stock_board_industry_name_em()
            mapping = {"板块名称": "sector", "板块代码": "sector_code"}
        frame = raw.rename(columns=mapping)
        keep = [c for c in ["sector_code", "sector"] if c in frame.columns]
        return frame[keep].dropna(subset=["sector"]).drop_duplicates()

    def industry_daily(self, sector: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
        try:
            return self._industry_daily_ths(sector, start_date, end_date)
        except Exception:
            return self._industry_daily_eastmoney(sector, start_date, end_date)

    def _industry_daily_eastmoney(self, sector: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
        ak = self._ak()
        start = start_date.replace("-", "")
        end = (end_date or pd.Timestamp.today().strftime("%Y-%m-%d")).replace("-", "")
        raw = ak.stock_board_industry_hist_em(symbol=sector, period="日k", start_date=start, end_date=end, adjust="")
        mapping = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
        }
        frame = raw.rename(columns=mapping)
        frame["sector"] = sector
        keep = [c for c in ["date", "sector", "open", "high", "low", "close", "volume", "amount"] if c in frame]
        frame = frame[keep].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        return frame

    def _industry_daily_ths(self, sector: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
        ak = self._ak()
        start = start_date.replace("-", "")
        end = (end_date or pd.Timestamp.today().strftime("%Y-%m-%d")).replace("-", "")
        raw = ak.stock_board_industry_index_ths(symbol=sector, start_date=start, end_date=end)
        mapping = {
            "日期": "date",
            "开盘价": "open",
            "收盘价": "close",
            "最高价": "high",
            "最低价": "low",
            "成交量": "volume",
            "成交额": "amount",
        }
        frame = raw.rename(columns=mapping)
        frame["sector"] = sector
        keep = [c for c in ["date", "sector", "open", "high", "low", "close", "volume", "amount"] if c in frame]
        frame = frame[keep].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        return frame

    def industry_constituents(self, sector: str) -> pd.DataFrame:
        ak = self._ak()
        raw = ak.stock_board_industry_cons_em(symbol=sector)
        mapping = {"代码": "code", "名称": "name"}
        frame = raw.rename(columns=mapping)
        frame["sector"] = sector
        keep = [c for c in ["code", "name", "sector"] if c in frame.columns]
        frame = frame[keep].copy()
        frame["code"] = frame["code"].astype(str).str.zfill(6)
        return frame.drop_duplicates(subset=["code", "sector"])

    def stock_individual_info(self, code: str) -> pd.DataFrame:
        ak = self._ak()
        raw = ak.stock_individual_info_em(symbol=str(code).zfill(6))
        values = dict(zip(raw["item"], raw["value"]))
        sector = self.normalize_sector(values.get("行业", ""))
        return pd.DataFrame(
            [
                {
                    "code": str(values.get("股票代码", code)).zfill(6),
                    "name": values.get("股票简称", ""),
                    "sector": sector,
                    "market_cap": pd.to_numeric(values.get("总市值"), errors="coerce"),
                    "float_market_cap": pd.to_numeric(values.get("流通市值"), errors="coerce"),
                    "list_date": values.get("上市时间", ""),
                }
            ]
        )

    @staticmethod
    def normalize_sector(value: object) -> str:
        sector = str(value or "").strip()
        sector = re.sub(r"[ⅠⅡⅢIVX]+$", "", sector).strip()
        return sector
