from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, allow_nan=True), encoding="utf-8")
    return path


def sidecar_path(output_path: str | Path, kind: str) -> Path:
    output = Path(output_path)
    return output.with_suffix(output.suffix + f".{kind}.json")
