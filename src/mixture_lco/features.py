"""Portable containers matching the original training module's input contract."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from mixture_lco.data import Dataset
from mixture_lco.io import decode_json, digest


@dataclass(frozen=True)
class ComponentSample:
    embeddings: NDArray[np.float32]
    mass_fractions: NDArray[np.float64]
    compound_ids: tuple[str, ...]


@dataclass(frozen=True)
class ComponentFeatureSet:
    sample_ids: tuple[str, ...]
    feature_family: str
    samples: tuple[ComponentSample, ...]
    max_components: int
    dimension: int


class InferenceTimingObserver(Protocol):
    """Optional timing interface retained for the import-adapted training module."""

    logical_rows: int
    physical_forwards: int

    def span(self, stage: str, device: str) -> AbstractContextManager[None]: ...


def build_features(dataset: Dataset, embeddings: Path | None = None
                   ) -> tuple[ComponentFeatureSet, dict[str, object]]:
    vocabulary = sorted(dataset.compounds)
    vectors: dict[str, NDArray[np.float32]] = {}
    metadata: dict[str, object] = {"compound_vocabulary": vocabulary}
    if embeddings is None:
        identity = np.eye(len(vocabulary), dtype=np.float32)
        vectors = {cid: identity[i] for i, cid in enumerate(vocabulary)}
        metadata["kind"] = "registry-one-hot"
    else:
        # Hash exactly the same bytes as the decoder reads, including whitespace.
        payload = embeddings.read_bytes()
        raw = decode_json(payload)
        if not isinstance(raw, dict) or set(raw) != set(vocabulary):
            raise ValueError("embedding keys must exactly match the compound registry")
        dimension = None
        for cid in vocabulary:
            values = raw[cid]
            if not isinstance(values, list) or not values or any(
                type(value) not in (int, float) for value in values
            ):
                raise ValueError("embeddings must be nonempty numeric vectors (no booleans)")
            with np.errstate(over="ignore", invalid="ignore"):
                vector = np.asarray(values, dtype=np.float32)
            if vector.ndim != 1 or not bool(np.isfinite(vector).all()):
                raise ValueError("embedding vectors must be finite float32 values")
            if dimension is not None and len(vector) != dimension:
                raise ValueError("embedding vectors must have equal length")
            dimension = len(vector)
            vectors[cid] = vector
        metadata.update(kind="supplied-frozen-vectors", input_sha256=digest(payload))
    dimension = len(vectors[vocabulary[0]])
    samples = []
    for sample in dataset.samples:
        parts = dataset.formulations[sample.formulation_id].components
        samples.append(ComponentSample(
            np.stack([vectors[cid] for cid, _ in parts]),
            np.asarray([fraction for _, fraction in parts], dtype=np.float64),
            tuple(cid for cid, _ in parts)))
    metadata["dimension"] = dimension
    return ComponentFeatureSet(tuple(s.sample_id for s in dataset.samples), str(metadata["kind"]),
                               tuple(samples), max(len(s.compound_ids) for s in samples),
                               dimension), metadata


def subset(features: ComponentFeatureSet, rows: list[int]) -> ComponentFeatureSet:
    samples = tuple(features.samples[i] for i in rows)
    return ComponentFeatureSet(tuple(features.sample_ids[i] for i in rows), features.feature_family,
                               samples, max(len(s.compound_ids) for s in samples),
                               features.dimension)
