from __future__ import annotations

DATE = "date"
CODE = "code"
SECTOR = "sector"

PRICE_COLUMNS = [
    DATE,
    CODE,
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]

OPTIONAL_STOCK_COLUMNS = [
    "name",
    "sector",
    "market_cap",
    "turnover_rate",
    "is_st",
    "list_date",
]

INDEX_COLUMNS = [DATE, CODE, "open", "high", "low", "close", "volume", "amount"]
SECTOR_COLUMNS = [DATE, SECTOR, "close", "volume", "amount", "up_ratio"]

FUNDAMENTAL_COLUMNS = [DATE, CODE, "feature", "value", "source_sheet"]
