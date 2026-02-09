from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


def build_label_map(labels: Iterable[str]) -> Dict[str, int]:
    unique = sorted(set(labels))
    return {name: idx + 1 for idx, name in enumerate(unique)}


def save_label_map(label_map: Dict[str, int], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)


def load_label_map(path: str | Path) -> Dict[str, int] | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
