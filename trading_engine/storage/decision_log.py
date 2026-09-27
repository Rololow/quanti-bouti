"""Journal des décisions (README §41-42) : une décision par ligne JSONL, avec
tous ses diagnostics, pour pouvoir répondre à « pourquoi le modèle voulait
réduire cette position à 14:32 ? » sans relancer le moteur."""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path
from typing import IO, Any


def _default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (tuple, set, frozenset)):
        return list(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


class DecisionLogWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] | None = open(self.path, "a", encoding="utf-8")

    def write(self, decision: Any) -> None:
        if self._fh is None:
            raise ValueError("decision log is closed")
        record = dataclasses.asdict(decision)
        record["costs"]["total"] = decision.costs.total
        self._fh.write(json.dumps(record, default=_default, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def read_decisions(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
