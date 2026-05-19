from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quantify.config import load_config
from quantify.data.akshare_source import AkShareSource
from quantify.data.local_store import LocalStore
from quantify.data.my_stock import extract_feature_dir, list_std_sheets
from quantify.data.sample import make_sample_data
from quantify.pipeline import prepare_prediction_arrays, prepare_training_arrays


def _config(args: argparse.Namespace):
    return load_config(args.config)


def cmd_make_sample_data(args: argparse.Namespace) -> None:
    config = _config(args)
    make_sample_data(config.paths.data_dir)
    print(f"sample data written to {config.paths.data_dir}")


def cmd_fetch_stock_list(args: argparse.Namespace) -> None:
    config = _config(args)
    source = AkShareSource()
    frame = source.stock_list()
    LocalStore(config.paths.data_dir).write_csv("stocks/list.csv", frame)
    print(f"stock list rows: {len(frame)}")


def cmd_fetch_daily(args: argparse.Namespace) -> None:
    config = _config(args)
    source = AkShareSource()
    store = LocalStore(config.paths.data_dir)
    stock_list = store.read_csv("stocks/list.csv")
    if stock_list.empty:
        stock_list = source.stock_list()
    codes = stock_list["code"].astype(str).str.zfill(6).tolist()
    if args.limit:
        codes = codes[: args.limit]
    frames = []
    for code in codes:
        try:
            frame = source.stock_daily(code, config.data.start_date, config.data.end_date)
        except Exception as exc:
            print(f"skip {code}: {exc}")
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("No stock daily data fetched.")
    result = pd.concat(frames, ignore_index=True)
    store.write_csv("stocks/daily.csv", result)
    print(f"daily rows: {len(result)}, codes: {result['code'].nunique()}")


def cmd_fetch_index(args: argparse.Namespace) -> None:
    config = _config(args)
    source = AkShareSource()
    frames = []
    for symbol in config.data.index_symbols:
        try:
            frames.append(source.index_daily(symbol, config.data.start_date, config.data.end_date))
        except Exception as exc:
            print(f"skip {symbol}: {exc}")
    if not frames:
        raise RuntimeError("No index data fetched.")
    result = pd.concat(frames, ignore_index=True)
    LocalStore(config.paths.data_dir).write_csv("market/index_daily.csv", result)
    print(f"index rows: {len(result)}")


def cmd_extract_my_stock(args: argparse.Namespace) -> None:
    config = _config(args)
    if args.list_std:
        sheets = list_std_sheets(config.paths.my_stock_std_path)
        print(json.dumps(sheets, ensure_ascii=False, indent=2))
        return
    output = Path(config.paths.data_dir) / "fundamentals" / "features.csv"
    frame = extract_feature_dir(
        config.paths.my_stock_final_dir,
        output,
        report_lag_days_if_unknown=config.data.report_lag_days_if_unknown,
    )
    print(f"fundamental features rows: {len(frame)} -> {output}")


def cmd_train(args: argparse.Namespace) -> None:
    try:
        from quantify.models.train import train_deep_model
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit("PyTorch is not installed. Install the remote training environment with `pip install -r requirements.txt`.") from exc
        raise

    config = _config(args)
    arrays = prepare_training_arrays(config)
    result = train_deep_model(arrays, config, config.paths.model_dir)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(json.dumps({"baseline": result["baseline_metrics"]}, ensure_ascii=False, indent=2))


def cmd_predict(args: argparse.Namespace) -> None:
    try:
        from quantify.models.predict import score_candidates, write_top_report
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit("PyTorch is not installed. Install the remote training environment with `pip install -r requirements.txt`.") from exc
        raise

    config = _config(args)
    arrays = prepare_prediction_arrays(config)
    candidates = score_candidates(arrays, Path(config.paths.model_dir) / "deep_sequence.pt", config, args.as_of_date)
    output = write_top_report(candidates, config.paths.report_dir, config.report.top_n)
    print(f"report written: {output}")
    if not candidates.empty:
        print(candidates.head(config.report.top_n)[["date", "code", "name", "sector", "score"]].to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantify")
    parser.add_argument("--config", default="configs/config.example.yaml", help="YAML config path")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-sample-data")
    p.set_defaults(func=cmd_make_sample_data)

    p = sub.add_parser("fetch-stock-list")
    p.set_defaults(func=cmd_fetch_stock_list)

    p = sub.add_parser("fetch-daily")
    p.add_argument("--limit", type=int, default=0, help="Limit stocks for a smoke fetch")
    p.set_defaults(func=cmd_fetch_daily)

    p = sub.add_parser("fetch-index")
    p.set_defaults(func=cmd_fetch_index)

    p = sub.add_parser("extract-my-stock")
    p.add_argument("--list-std", action="store_true", help="Print sheets in the My_Stock std workbook")
    p.set_defaults(func=cmd_extract_my_stock)

    p = sub.add_parser("train")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict")
    p.add_argument("--as-of-date")
    p.set_defaults(func=cmd_predict)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
