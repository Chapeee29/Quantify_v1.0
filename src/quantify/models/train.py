from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from quantify.config import AppConfig
from quantify.features.dataset import SequenceArrays, split_by_time
from quantify.models.deep_sequence import DeepSequenceModel


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _subset(arrays: SequenceArrays, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y = arrays.y
    if y is None:
        raise ValueError("Training requires labels.")
    return (
        torch.tensor(arrays.x_sequence[mask], dtype=torch.float32),
        torch.tensor(arrays.x_static[mask], dtype=torch.float32),
        torch.tensor(y[mask], dtype=torch.float32),
    )


def _metrics(y_true: np.ndarray, score: np.ndarray, top_n: int, prefix: str) -> dict[str, float]:
    pred = (score >= 0.5).astype(float)
    result = {
        f"{prefix}_accuracy": float(accuracy_score(y_true, pred)),
        f"{prefix}_precision": float(precision_score(y_true, pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        result[f"{prefix}_auc"] = float(roc_auc_score(y_true, score))
    order = np.argsort(score)[::-1][: min(top_n, len(score))]
    if len(order):
        result[f"{prefix}_precision_at_{top_n}"] = float(np.mean(y_true[order]))
    return result


def _predict_scores(model: DeepSequenceModel, arrays: SequenceArrays, mask: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    seq = torch.tensor(arrays.x_sequence[mask], dtype=torch.float32)
    sta = torch.tensor(arrays.x_static[mask], dtype=torch.float32)
    loader = DataLoader(TensorDataset(seq, sta), batch_size=512, shuffle=False)
    scores = []
    with torch.no_grad():
        for batch_seq, batch_static in loader:
            logits = model(batch_seq.to(device), batch_static.to(device))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores) if scores else np.array([])


def train_deep_model(arrays: SequenceArrays, config: AppConfig, artifact_dir: str | Path) -> dict[str, object]:
    torch.manual_seed(config.model.random_seed)
    np.random.seed(config.model.random_seed)
    masks = split_by_time(arrays, config.train.train_end, config.train.validation_end)
    device = resolve_device(config.model.device)

    x_train, s_train, y_train = _subset(arrays, masks["train"])
    x_val, s_val, y_val = _subset(arrays, masks["validation"])
    if len(y_train) == 0 or len(y_val) == 0:
        raise ValueError("Train and validation splits must both contain samples.")

    train_loader = DataLoader(
        TensorDataset(x_train, s_train, y_train),
        batch_size=config.model.batch_size,
        shuffle=True,
    )
    model = DeepSequenceModel(
        sequence_dim=arrays.x_sequence.shape[-1],
        static_dim=arrays.x_static.shape[-1],
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.model.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    history: list[dict[str, float]] = []
    best_state = None
    best_val_auc = -1.0
    for epoch in range(1, config.model.epochs + 1):
        model.train()
        losses = []
        for batch_seq, batch_static, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_seq.to(device), batch_static.to(device))
            loss = criterion(logits, batch_y.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_score = _predict_scores(model, arrays, masks["validation"], device)
        val_y = arrays.y[masks["validation"]]
        epoch_metrics = _metrics(val_y, val_score, config.train.top_n, "validation")
        epoch_metrics["epoch"] = float(epoch)
        epoch_metrics["train_loss"] = float(np.mean(losses)) if losses else 0.0
        history.append(epoch_metrics)
        val_auc = epoch_metrics.get("validation_auc", epoch_metrics.get(f"validation_precision_at_{config.train.top_n}", 0.0))
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics: dict[str, float] = {}
    for split in ("train", "validation", "test"):
        mask = masks[split]
        if not mask.any():
            continue
        score = _predict_scores(model, arrays, mask, device)
        y_true = arrays.y[mask]
        metrics.update(_metrics(y_true, score, config.train.top_n, split))
        returns = pd.to_numeric(arrays.meta.loc[mask, "next_return"], errors="coerce").to_numpy()
        order = np.argsort(score)[::-1][: min(config.train.top_n, len(score))]
        if len(order):
            metrics[f"{split}_top_{config.train.top_n}_mean_next_return"] = float(np.nanmean(returns[order]))

    baseline_metrics = train_baseline(arrays, masks, config.train.top_n)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "sequence_columns": arrays.sequence_columns,
            "static_columns": arrays.static_columns,
            "model_config": {
                "hidden_size": config.model.hidden_size,
                "num_layers": config.model.num_layers,
                "dropout": config.model.dropout,
            },
        },
        artifact_dir / "deep_sequence.pt",
    )
    with (artifact_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"deep_sequence": metrics, "baseline": baseline_metrics, "history": history}, f, ensure_ascii=False, indent=2)
    return {"metrics": metrics, "baseline_metrics": baseline_metrics, "history": history}


def train_baseline(arrays: SequenceArrays, masks: dict[str, np.ndarray], top_n: int) -> dict[str, float]:
    y = arrays.y
    if y is None:
        return {}
    latest_sequence = arrays.x_sequence[:, -1, :]
    x = np.concatenate([latest_sequence, arrays.x_static], axis=1)
    model = HistGradientBoostingClassifier(max_iter=80, random_state=42)
    model.fit(x[masks["train"]], y[masks["train"]])
    metrics: dict[str, float] = {}
    for split in ("validation", "test"):
        mask = masks[split]
        if not mask.any():
            continue
        score = model.predict_proba(x[mask])[:, 1]
        metrics.update(_metrics(y[mask], score, top_n, f"baseline_{split}"))
    return metrics
