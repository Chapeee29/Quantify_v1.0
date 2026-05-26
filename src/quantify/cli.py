from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

import pandas as pd

from quantify.config import load_config
from quantify.data.akshare_source import AkShareSource
from quantify.data.local_store import LocalStore
from quantify.data.my_stock import extract_feature_dir, list_std_sheets
from quantify.data.sample import make_sample_data


def _config(args: argparse.Namespace):
    return load_config(args.config)


def _disable_proxy_env() -> None:
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


def _prepare_network(args: argparse.Namespace) -> None:
    if not getattr(args, "use_proxy", False):
        _disable_proxy_env()


def _parse_prefixes(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _select_codes(codes: list[str], args: argparse.Namespace) -> list[str]:
    prefixes = _parse_prefixes(getattr(args, "prefixes", None))
    if prefixes:
        codes = [code for code in codes if any(code.startswith(prefix) for prefix in prefixes)]

    per_prefix_limit = getattr(args, "per_prefix_limit", 0)
    if prefixes and per_prefix_limit:
        selected: list[str] = []
        for prefix in prefixes:
            selected.extend([code for code in codes if code.startswith(prefix)][:per_prefix_limit])
        codes = selected

    limit = getattr(args, "limit", 0)
    if limit:
        codes = codes[:limit]
    return codes


def _try_fetch(call, retries: int):
    last_error = None
    for attempt in range(retries + 1):
        try:
            return call(), None
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    return pd.DataFrame(), last_error


def _fetch_stock_daily_process(payload):
    code, start_date, end_date, adjust, retries, use_proxy = payload
    if not use_proxy:
        _disable_proxy_env()
    source = AkShareSource(adjust=adjust)
    frame, error = _try_fetch(lambda: source.stock_daily(code, start_date, end_date), retries)
    return code, frame, error


def cmd_make_sample_data(args: argparse.Namespace) -> None:
    config = _config(args)
    make_sample_data(config.paths.data_dir)
    print(f"sample data written to {config.paths.data_dir}")


def cmd_fetch_stock_list(args: argparse.Namespace) -> None:
    _prepare_network(args)
    config = _config(args)
    source = AkShareSource()
    frame = source.stock_list()
    LocalStore(config.paths.data_dir).write_csv("stocks/list.csv", frame)
    print(f"stock list rows: {len(frame)}")


def cmd_fetch_daily(args: argparse.Namespace) -> None:
    _prepare_network(args)
    config = _config(args)
    source = AkShareSource()
    store = LocalStore(config.paths.data_dir)
    stock_list = store.read_csv("stocks/list.csv")
    if stock_list.empty:
        stock_list = source.stock_list()
    codes = stock_list["code"].astype(str).str.zfill(6).tolist()
    codes = _select_codes(codes, args)
    if not codes:
        raise RuntimeError("No stock codes selected.")
    print(f"selected daily codes: {len(codes)}, prefixes={','.join(_parse_prefixes(args.prefixes)) or 'all'}")
    frames = []
    failures = []
    existing = store.read_csv("stocks/daily.csv", parse_dates=["date"]) if args.incremental else pd.DataFrame()
    latest_by_code = existing.groupby("code")["date"].max().to_dict() if not existing.empty else {}
    flushed = 0
    successful = 0

    def payload_for(code: str):
        start_date = config.data.start_date
        if code in latest_by_code:
            start_date = (latest_by_code[code] - pd.Timedelta(days=args.overlap_days)).strftime("%Y-%m-%d")
        return (code, start_date, config.data.end_date, source.adjust, args.retries, args.use_proxy)

    # Sina history decoding uses MiniRacer, which is unsafe under shared-thread concurrency.
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {executor.submit(_fetch_stock_daily_process, payload_for(code)): code for code in codes}
        for idx, future in enumerate(as_completed(future_map), start=1):
            code, frame, exc = future.result()
            if exc is not None:
                failures.append((code, str(exc)))
                print(f"skip {code}: {exc}")
            elif not frame.empty:
                frames.append(frame)
                successful += 1
            if args.incremental and len(frames) >= args.checkpoint_every:
                store.upsert_csv("stocks/daily.csv", pd.concat(frames, ignore_index=True), ["code", "date"], parse_dates=["date"])
                flushed += len(frames)
                frames.clear()
            if idx % 50 == 0 or idx == len(codes):
                print(f"fetched daily {idx}/{len(codes)}, ok={successful}, failed={len(failures)}")
    if not frames and not flushed:
        raise RuntimeError("No stock daily data fetched.")
    if args.incremental:
        if frames:
            store.upsert_csv("stocks/daily.csv", pd.concat(frames, ignore_index=True), ["code", "date"], parse_dates=["date"])
        result = store.read_csv("stocks/daily.csv", parse_dates=["date"])
    else:
        result = pd.concat(frames, ignore_index=True)
        store.write_csv("stocks/daily.csv", result)
    print(f"daily rows: {len(result)}, codes: {result['code'].nunique()}, latest={pd.to_datetime(result['date']).max().date()}")


def cmd_fetch_index(args: argparse.Namespace) -> None:
    _prepare_network(args)
    config = _config(args)
    source = AkShareSource()
    frames = []
    store = LocalStore(config.paths.data_dir)
    existing = store.read_csv("market/index_daily.csv", parse_dates=["date"]) if args.incremental else pd.DataFrame()

    def fetch_one(symbol: str):
        frame, error = _try_fetch(lambda: source.index_daily(symbol, config.data.start_date, config.data.end_date), args.retries)
        return symbol, frame, error

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {executor.submit(fetch_one, symbol): symbol for symbol in config.data.index_symbols}
        for future in as_completed(future_map):
            symbol, frame, exc = future.result()
            if exc is not None:
                print(f"skip {symbol}: {exc}")
            elif not frame.empty:
                frames.append(frame)
    if not frames:
        if args.incremental and not existing.empty:
            print(f"index fetch failed; keep cached data through {existing['date'].max().date()}")
            return
        raise RuntimeError("No index data fetched.")
    result = pd.concat(frames, ignore_index=True)
    if args.incremental:
        store.upsert_csv("market/index_daily.csv", result, ["code", "date"], parse_dates=["date"])
        result = store.read_csv("market/index_daily.csv", parse_dates=["date"])
    else:
        store.write_csv("market/index_daily.csv", result)
    print(f"index rows: {len(result)}, latest={pd.to_datetime(result['date']).max().date()}")


def cmd_fetch_sector(args: argparse.Namespace) -> None:
    _prepare_network(args)
    config = _config(args)
    source = AkShareSource()
    store = LocalStore(config.paths.data_dir)
    sectors = source.industry_list()
    if args.limit:
        sectors = sectors.head(args.limit)
    sector_names = sectors["sector"].astype(str).tolist()
    daily_frames = []
    mapping_frames = []
    failures = []

    def fetch_one(sector: str):
        try:
            daily = source.industry_daily(sector, config.data.start_date, config.data.end_date)
            mapping = source.industry_constituents(sector)
            return sector, daily, mapping, None
        except Exception as exc:
            return sector, pd.DataFrame(), pd.DataFrame(), exc

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {executor.submit(fetch_one, sector): sector for sector in sector_names}
        for idx, future in enumerate(as_completed(future_map), start=1):
            sector, daily, mapping, exc = future.result()
            if exc is not None:
                failures.append((sector, str(exc)))
                print(f"skip sector {sector}: {exc}")
            else:
                if not daily.empty:
                    daily_frames.append(daily)
                if not mapping.empty:
                    mapping_frames.append(mapping)
            if idx % 10 == 0 or idx == len(sector_names):
                print(f"fetched sectors {idx}/{len(sector_names)}, ok={len(daily_frames)}, failed={len(failures)}")

    if not daily_frames:
        raise RuntimeError("No sector daily data fetched.")
    sector_daily = pd.concat(daily_frames, ignore_index=True)
    store.write_csv("market/sector_daily.csv", sector_daily)
    if mapping_frames:
        sector_map = pd.concat(mapping_frames, ignore_index=True).drop_duplicates(subset=["code"])
        store.write_csv("stocks/sector_map.csv", sector_map)
        print(f"sector map rows: {len(sector_map)}")
    print(f"sector daily rows: {len(sector_daily)}, sectors: {sector_daily['sector'].nunique()}")


def cmd_fetch_stock_info(args: argparse.Namespace) -> None:
    _prepare_network(args)
    config = _config(args)
    source = AkShareSource()
    store = LocalStore(config.paths.data_dir)
    stock_list = store.read_csv("stocks/list.csv")
    if stock_list.empty:
        stock_list = source.stock_list()
    codes = stock_list["code"].astype(str).str.zfill(6).tolist()
    codes = _select_codes(codes, args)
    if not codes:
        raise RuntimeError("No stock codes selected.")
    print(f"selected stock info codes: {len(codes)}, prefixes={','.join(_parse_prefixes(args.prefixes)) or 'all'}")
    frames = []
    failures = []

    def fetch_one(code: str):
        try:
            return code, source.stock_individual_info(code), None
        except Exception as exc:
            return code, pd.DataFrame(), exc

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {executor.submit(fetch_one, code): code for code in codes}
        for idx, future in enumerate(as_completed(future_map), start=1):
            code, frame, exc = future.result()
            if exc is not None:
                failures.append((code, str(exc)))
                print(f"skip info {code}: {exc}")
            elif not frame.empty:
                frames.append(frame)
            if idx % 50 == 0 or idx == len(codes):
                print(f"fetched stock info {idx}/{len(codes)}, ok={len(frames)}, failed={len(failures)}")
    if not frames:
        raise RuntimeError("No stock info data fetched.")
    result = pd.concat(frames, ignore_index=True)
    store.write_csv("stocks/info.csv", result)
    store.write_csv("stocks/sector_map.csv", result[["code", "name", "sector"]].dropna(subset=["sector"]))
    print(f"stock info rows: {len(result)}, sectors: {result['sector'].nunique()}")


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
    from quantify.pipeline import prepare_training_arrays

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
    from quantify.pipeline import prepare_prediction_arrays

    config = _config(args)
    arrays = prepare_prediction_arrays(config)
    candidates = score_candidates(
        arrays,
        Path(config.paths.model_dir) / "deep_sequence.pt",
        config,
        args.as_of_date,
        include_st=args.include_st,
        include_stale=args.include_stale,
    )
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
    p.add_argument("--use-proxy", action="store_true", help="Use current HTTP(S)_PROXY environment variables")
    p.set_defaults(func=cmd_fetch_stock_list)

    p = sub.add_parser("fetch-daily")
    p.add_argument("--limit", type=int, default=0, help="Limit stocks for a smoke fetch")
    p.add_argument("--prefixes", default="00,60", help="Comma-separated stock code prefixes, default: 00,60")
    p.add_argument("--per-prefix-limit", type=int, default=0, help="Fetch this many stocks for each requested prefix")
    p.add_argument("--workers", type=int, default=4, help="Concurrent fetch processes for MiniRacer-safe history retrieval")
    p.add_argument("--retries", type=int, default=2, help="Retry count for failed stock requests")
    p.add_argument("--incremental", action="store_true", help="Merge fetched rows into the existing daily data")
    p.add_argument("--overlap-days", type=int, default=7, help="Refetch recent calendar days during incremental update")
    p.add_argument("--checkpoint-every", type=int, default=100, help="Persist this many successful incremental fetches at a time")
    p.add_argument("--use-proxy", action="store_true", help="Use current HTTP(S)_PROXY environment variables")
    p.set_defaults(func=cmd_fetch_daily)

    p = sub.add_parser("fetch-stock-info")
    p.add_argument("--limit", type=int, default=0, help="Limit stocks for a smoke fetch")
    p.add_argument("--prefixes", default="00,60", help="Comma-separated stock code prefixes, default: 00,60")
    p.add_argument("--per-prefix-limit", type=int, default=0, help="Fetch this many stocks for each requested prefix")
    p.add_argument("--workers", type=int, default=8, help="Concurrent fetch workers")
    p.add_argument("--use-proxy", action="store_true", help="Use current HTTP(S)_PROXY environment variables")
    p.set_defaults(func=cmd_fetch_stock_info)

    p = sub.add_parser("fetch-index")
    p.add_argument("--workers", type=int, default=4, help="Concurrent fetch workers")
    p.add_argument("--incremental", action="store_true", help="Merge fetched rows into the existing index data")
    p.add_argument("--retries", type=int, default=2, help="Retry count for failed index requests")
    p.add_argument("--use-proxy", action="store_true", help="Use current HTTP(S)_PROXY environment variables")
    p.set_defaults(func=cmd_fetch_index)

    p = sub.add_parser("fetch-sector")
    p.add_argument("--limit", type=int, default=0, help="Limit sectors for a smoke fetch")
    p.add_argument("--workers", type=int, default=6, help="Concurrent fetch workers")
    p.add_argument("--use-proxy", action="store_true", help="Use current HTTP(S)_PROXY environment variables")
    p.set_defaults(func=cmd_fetch_sector)

    p = sub.add_parser("extract-my-stock")
    p.add_argument("--list-std", action="store_true", help="Print sheets in the My_Stock std workbook")
    p.set_defaults(func=cmd_extract_my_stock)

    p = sub.add_parser("train")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict")
    p.add_argument("--as-of-date")
    p.add_argument("--include-st", action="store_true", help="Include ST and *ST stocks in the recommendation report")
    p.add_argument("--include-stale", action="store_true", help="Include stocks whose latest local trading date is older than the max date")
    p.set_defaults(func=cmd_predict)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
