from __future__ import annotations

import pickle
from pathlib import Path


def save_preprocessor(path: str | Path, payload: dict[str, object]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("wb") as f:
        pickle.dump(payload, f)


def load_preprocessor(path: str | Path) -> dict[str, object]:
    with Path(path).open("rb") as f:
        return pickle.load(f)
