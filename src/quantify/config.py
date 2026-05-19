from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    data_dir: str = "data"
    model_dir: str = "models"
    report_dir: str = "reports"
    my_stock_std_path: str = r"D:\codex\My_Stock_v1.0\std.xlsx"
    my_stock_final_dir: str = r"D:\codex\My_Stock_v1.0\Stock_Final"


@dataclass
class DataConfig:
    start_date: str = "2023-01-01"
    end_date: str | None = None
    stock_universe: str = "a_share_all"
    index_symbols: list[str] = field(
        default_factory=lambda: ["sh000001", "sz399001", "sz399006", "sh000300", "sh000905"]
    )
    report_lag_days_if_unknown: int = 90


@dataclass
class FeatureConfig:
    sequence_length: int = 120
    sequence_lengths_supported: list[int] = field(default_factory=lambda: [60, 120, 240])
    return_windows: list[int] = field(default_factory=lambda: [1, 5, 20, 60])
    volatility_windows: list[int] = field(default_factory=lambda: [5, 20])
    market_windows: list[int] = field(default_factory=lambda: [1, 5, 20, 60])
    include_fundamentals: bool = True
    include_market: bool = True
    include_sector: bool = True


@dataclass
class LabelConfig:
    next_day_return_threshold: float = 0.01


@dataclass
class ModelConfig:
    hidden_size: int = 96
    num_layers: int = 1
    dropout: float = 0.15
    learning_rate: float = 0.001
    batch_size: int = 128
    epochs: int = 12
    device: str = "auto"
    random_seed: int = 42


@dataclass
class TrainConfig:
    train_end: str = "2025-06-30"
    validation_end: str = "2025-12-31"
    top_n: int = 20


@dataclass
class ReportConfig:
    top_n: int = 20
    min_history_days: int = 240


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    report: ReportConfig = field(default_factory=ReportConfig)


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _build_dataclass(cls: type, values: dict[str, Any]):
    return cls(**values)


def default_config_dict() -> dict[str, Any]:
    return asdict(AppConfig())


def load_config(path: str | Path | None = None) -> AppConfig:
    values = default_config_dict()
    if path:
        with Path(path).open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        _deep_update(values, loaded)

    return AppConfig(
        paths=_build_dataclass(PathsConfig, values["paths"]),
        data=_build_dataclass(DataConfig, values["data"]),
        features=_build_dataclass(FeatureConfig, values["features"]),
        label=_build_dataclass(LabelConfig, values["label"]),
        model=_build_dataclass(ModelConfig, values["model"]),
        train=_build_dataclass(TrainConfig, values["train"]),
        report=_build_dataclass(ReportConfig, values["report"]),
    )


def ensure_runtime_dirs(config: AppConfig) -> None:
    for value in (config.paths.data_dir, config.paths.model_dir, config.paths.report_dir):
        Path(value).mkdir(parents=True, exist_ok=True)
