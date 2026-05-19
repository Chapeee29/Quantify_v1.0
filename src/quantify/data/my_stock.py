from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook


DEFAULT_FEATURE_SHEETS = [
    "汇总",
    "转化后的ROE",
    "现金流汇总2",
    "季度现金流",
    "季度利润表",
    "资产负债表",
    "利润表",
    "现金流量表",
    "估值",
    "校验风险",
    "周转率",
]


def list_std_sheets(std_path: str | Path) -> list[str]:
    wb = load_workbook(std_path, read_only=True, data_only=True)
    try:
        return list(wb.sheetnames)
    finally:
        wb.close()


def _stock_code_from_path(path: Path) -> str:
    match = re.search(r"(\d{6})", path.stem)
    if not match:
        raise ValueError(f"Cannot infer stock code from workbook name: {path.name}")
    return match.group(1)


def _safe_feature_name(raw: object) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value or value.lower() == "nan":
        return None
    return re.sub(r"\s+", "_", value)


def _date_columns(header_row: tuple[object, ...]) -> dict[int, pd.Timestamp]:
    result: dict[int, pd.Timestamp] = {}
    for idx, value in enumerate(header_row):
        if idx == 0 or value is None:
            continue
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.notna(parsed):
            result[idx] = pd.Timestamp(parsed).normalize()
    return result


def extract_features_from_workbook(
    workbook_path: str | Path,
    sheets: list[str] | None = None,
    report_lag_days_if_unknown: int = 90,
) -> pd.DataFrame:
    """Extract long-form fundamental features from a My_Stock final workbook.

    My_Stock's Excel files are excellent for one-stock deep analysis but are not
    leakage-safe by themselves. This extractor emits each report value with a
    conservative effective date. If a future version captures actual disclosure
    dates, those dates should replace the lag rule here.
    """

    path = Path(workbook_path)
    code = _stock_code_from_path(path)
    wb = load_workbook(path, read_only=True, data_only=True)
    rows: list[dict[str, object]] = []
    wanted = sheets or DEFAULT_FEATURE_SHEETS
    try:
        for sheet in wanted:
            if sheet not in wb.sheetnames:
                continue
            ws = wb[sheet]
            header_candidates = []
            for r in range(1, min(ws.max_row, 8) + 1):
                values = tuple(cell.value for cell in ws[r])
                date_cols = _date_columns(values)
                if date_cols:
                    header_candidates.append((r, date_cols))
            if not header_candidates:
                continue
            header_row, date_cols = max(header_candidates, key=lambda item: len(item[1]))
            for r in range(header_row + 1, ws.max_row + 1):
                feature = _safe_feature_name(ws.cell(r, 1).value)
                if not feature:
                    continue
                for col_idx, report_date in date_cols.items():
                    value = ws.cell(r, col_idx + 1).value
                    numeric = pd.to_numeric(value, errors="coerce")
                    if pd.isna(numeric):
                        continue
                    effective_date = report_date + pd.Timedelta(days=report_lag_days_if_unknown)
                    rows.append(
                        {
                            "date": effective_date,
                            "report_date": report_date,
                            "code": code,
                            "feature": f"{sheet}.{feature}",
                            "value": float(numeric),
                            "source_sheet": sheet,
                            "source_file": str(path),
                        }
                    )
    finally:
        wb.close()
    return pd.DataFrame(rows)


def extract_feature_dir(
    final_dir: str | Path,
    output_path: str | Path,
    report_lag_days_if_unknown: int = 90,
) -> pd.DataFrame:
    frames = []
    for workbook in Path(final_dir).glob("*.xlsx"):
        try:
            frame = extract_features_from_workbook(workbook, report_lag_days_if_unknown=report_lag_days_if_unknown)
        except Exception:
            continue
        if not frame.empty:
            frames.append(frame)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result
