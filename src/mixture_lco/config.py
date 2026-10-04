"""Closed configuration schema and representation-independent random search."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
import math
from pathlib import Path
from typing import Any

import numpy as np

from mixture_lco.io import decode_json, digest
from mixture_lco.mixnet import MixNetConfig


@dataclass(frozen=True)
class Settings:
    payload: dict[str, Any]

    @property
    def seeds(self) -> list[int]:
        return list(self.payload["seeds"])

    def model(self, dimension: int, candidate: dict[str, Any]) -> MixNetConfig:
        values = dict(self.payload["architecture"])
        for name in ("gate_hidden", "message_hidden", "trunk_hidden", "head_hidden"):
            values[name] = tuple(values[name])
        return MixNetConfig(input_dimension=dimension,
                            n_message_layers=candidate["n_message_layers"], **values)

    def candidates(self, seed: int) -> list[dict[str, Any]]:
        # No ingredient, encoder, dimension, targets, or split enters the trial stream.
        rng = np.random.Generator(np.random.PCG64(seed))
        space = self.payload["search"]
        result = []
        for _ in range(self.payload["trials"]):
            result.append({
                "n_message_layers": int(rng.choice(space["n_message_layers"])),
                "learning_rate": float(math.exp(rng.uniform(
                    math.log(space["learning_rate"][0]), math.log(space["learning_rate"][1])))),
                "weight_decay": float(rng.choice(space["weight_decay"])),
                "batch_size": int(rng.choice(space["batch_size"])),
            })
        return result


def _integer(value: Any, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def parse_settings(raw: Any) -> Settings:
    fields = {"label", "seeds", "split_seed", "cv_folds", "eval_folds", "hpo_folds",
              "validation_fraction", "trials", "max_epochs", "patience", "architecture", "search"}
    if not isinstance(raw, dict) or set(raw) != fields:
        raise ValueError(f"config requires exactly these fields: {sorted(fields)}")
    if not isinstance(raw["label"], str) or not raw["label"].strip():
        raise ValueError("config label must be nonempty")
    seeds = raw["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a nonempty list")
    for seed in [*seeds, raw["split_seed"]]:
        _integer(seed, "seed", 0)
        if seed >= 2**63:
            raise ValueError("seed must be less than 2**63")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    for name in ("cv_folds", "eval_folds", "hpo_folds", "trials", "max_epochs", "patience"):
        _integer(raw[name], name, 2 if name.endswith("folds") else 1)
    fraction = raw["validation_fraction"]
    if type(fraction) not in (int, float) or not 0 < fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    architecture = raw["architecture"]
    if not isinstance(architecture, dict) or set(architecture) != {
        "feature_len", "message_heads", "pooling_heads", "gate_hidden", "message_hidden",
        "trunk_hidden", "head_hidden",
    }:
        raise ValueError("architecture fields are not exact; MVE and extra features are off")
    for name in ("gate_hidden", "message_hidden", "trunk_hidden", "head_hidden"):
        if not isinstance(architecture[name], list):
            raise ValueError(f"{name} must be a list")
    space = raw["search"]
    if not isinstance(space, dict) or set(space) != {
        "n_message_layers", "learning_rate", "weight_decay", "batch_size",
    }:
        raise ValueError("search fields are not exact")
    for name, values in space.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"search.{name} must be a nonempty list")
        for value in values:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"search.{name} values must be finite numbers")
        if name != "learning_rate" and len(set(values)) != len(values):
            raise ValueError(f"search.{name} values must be unique")
    if any(type(v) is not int or v not in (0, 1, 2) for v in space["n_message_layers"]):
        raise ValueError("n_message_layers choices must be 0, 1, or 2")
    if any(v < 0 for v in space["weight_decay"]):
        raise ValueError("weight_decay must be nonnegative")
    for value in space["batch_size"]:
        _integer(value, "batch_size")
    lr = space["learning_rate"]
    if len(lr) != 2 or not 0 < lr[0] <= lr[1]:
        raise ValueError("learning_rate must be [positive lower, upper]")
    settings = Settings(raw)
    settings.model(1, {"n_message_layers": 0})
    return settings


def load_settings(path: Path | None = None, preset: str = "paper"
                  ) -> tuple[Settings, str]:
    source = path if path is not None else Path(str(files("mixture_lco") / "presets" /
                                                   f"{preset}.json"))
    payload = source.read_bytes()
    return parse_settings(decode_json(payload)), digest(payload)
