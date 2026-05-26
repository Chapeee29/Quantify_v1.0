from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from quantify.config import AppConfig
from quantify.features.dataset import SequenceArrays
from quantify.models.deep_sequence import DeepSequenceModel
from quantify.models.train import resolve_device


def load_model(model_path: str | Path, config: AppConfig, sequence_dim: int, static_dim: int) -> DeepSequenceModel:
    checkpoint = torch.load(model_path, map_location="cpu")
    model_cfg = checkpoint.get("model_config", {})
    model = DeepSequenceModel(
        sequence_dim=sequence_dim,
        static_dim=static_dim,
        hidden_size=int(model_cfg.get("hidden_size", config.model.hidden_size)),
        num_layers=int(model_cfg.get("num_layers", config.model.num_layers)),
        dropout=float(model_cfg.get("dropout", config.model.dropout)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model


def score_candidates(
    arrays: SequenceArrays,
    model_path: str | Path,
    config: AppConfig,
    as_of_date: str | None = None,
    include_st: bool = False,
    include_stale: bool = False,
) -> pd.DataFrame:
    if arrays.x_sequence.size == 0:
        return pd.DataFrame()
    model = load_model(model_path, config, arrays.x_sequence.shape[-1], arrays.x_static.shape[-1])
    device = resolve_device(config.model.device)
    model.to(device)
    model.eval()

    meta = arrays.meta.copy()
    if as_of_date:
        meta = meta[pd.to_datetime(meta["date"]) <= pd.Timestamp(as_of_date)]
    latest_idx = meta.sort_values("date").groupby("code").tail(1).index.to_numpy()
    seq = torch.tensor(arrays.x_sequence[latest_idx], dtype=torch.float32).to(device)
    sta = torch.tensor(arrays.x_static[latest_idx], dtype=torch.float32).to(device)
    with torch.no_grad():
        scores = torch.sigmoid(model(seq, sta)).cpu().numpy()
    result = arrays.meta.iloc[latest_idx].copy()
    result["score"] = scores
    if not include_st and "name" in result.columns:
        result = result[~result["name"].astype(str).str.contains("ST", case=False, na=False)]
    if not include_stale and not result.empty:
        latest_date = pd.to_datetime(result["date"]).max()
        result = result[pd.to_datetime(result["date"]) == latest_date]
    result = result.sort_values("score", ascending=False).reset_index(drop=True)
    return result


def write_top_report(candidates: pd.DataFrame, report_dir: str | Path, top_n: int) -> Path:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    if candidates.empty:
        output = report_dir / "top_candidates_empty.csv"
        candidates.to_csv(output, index=False, encoding="utf-8-sig")
        return output
    as_of = pd.to_datetime(candidates["date"]).max().strftime("%Y%m%d")
    top = candidates.head(top_n).copy()
    output = report_dir / f"top_{top_n}_{as_of}.csv"
    keep = [c for c in ["date", "target_date", "code", "name", "sector", "close", "score", "next_return"] if c in top]
    top[keep].to_csv(output, index=False, encoding="utf-8-sig")
    _write_daily_markdown(candidates, top, report_dir / f"daily_{as_of}.md")
    return output


def _pct(value: float) -> str:
    return "-" if pd.isna(value) else f"{value:.2%}"


def _write_daily_markdown(candidates: pd.DataFrame, top: pd.DataFrame, output: Path) -> None:
    row = candidates.iloc[0]
    as_of = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
    amount = row.get("market_total_amount", np.nan)
    amount_text = "-" if pd.isna(amount) else f"{float(amount) / 1e8:.2f} 亿元"
    lines = [
        f"# 候选股日报 {as_of}",
        "",
        "## 当日环境",
        "",
        f"- 当前有效候选股票数：{len(candidates)}",
        f"- 股票池上涨比例：{_pct(float(row.get('market_up_ratio', np.nan)))}",
        f"- 股票池平均涨跌：{_pct(float(row.get('market_mean_ret_1', np.nan)))}",
        f"- 股票池成交额：{amount_text}",
        f"- 上证指数单日变化：{_pct(float(row.get('sh_index_ret_1', np.nan)))}",
        f"- 沪深300单日变化：{_pct(float(row.get('hs300_ret_1', np.nan)))}",
        "",
        "## Top 候选",
        "",
        "| 排名 | 代码 | 名称 | 评分 |",
        "| ---: | --- | --- | ---: |",
    ]
    for idx, item in top.reset_index(drop=True).iterrows():
        lines.append(f"| {idx + 1} | {item['code']} | {item.get('name', '')} | {item['score']:.4f} |")
    lines.extend(
        [
            "",
            "## 注意",
            "",
            "- 当前报告是模型研究输出，不是自动交易指令。",
            "- 若本次只抓取验证批次，市场环境统计仅反映该批次，不代表完整 00/60 股票池。",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
