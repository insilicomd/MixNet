"""Strict JSON, deterministic serialization, and fresh output directories."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    return decode_json(path.read_bytes())


def decode_json(payload: bytes) -> Any:
    return json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True,
                       allow_nan=False) + "\n").encode("utf-8")


def digest(value: bytes) -> str:
    return sha256(value).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_bytes(json_bytes(value))


def append_json(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def fresh_directory(path: Path) -> None:
    # mkdir without exist_ok also rejects an existing empty directory or symlink.
    path.mkdir(parents=True, exist_ok=False)
