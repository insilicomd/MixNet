"""Explicit, chemistry-toolkit-free CSV input boundary."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import io
import math
from pathlib import Path
import struct

from mixture_lco.io import digest, json_bytes

TARGETS = ("DS", "DE", "CS", "CE")
FRACTION_TOLERANCE = Decimal("0.000001")
CSV_HEADERS = {
    "compounds.csv": ("compound_id", "canonical_smiles"),
    "formulation_components.csv": ("formulation_id", "compound_id", "mass_fraction"),
    "measurements.csv": ("sample_id", "formulation_id", *TARGETS),
}


@dataclass(frozen=True)
class Formulation:
    formulation_id: str
    components: tuple[tuple[str, Fraction], ...]
    ingredient_set_id: str


@dataclass(frozen=True)
class Sample:
    sample_id: str
    formulation_id: str
    targets: tuple[float, ...]


@dataclass(frozen=True)
class Dataset:
    compounds: dict[str, str]
    formulations: dict[str, Formulation]
    samples: tuple[Sample, ...]
    input_hashes: dict[str, str]

    def rows(self, formulation_ids: set[str]) -> list[int]:
        return [i for i, sample in enumerate(self.samples)
                if sample.formulation_id in formulation_ids]

    def vocabulary(self, rows: list[int]) -> set[str]:
        return {compound for i in rows
                for compound, _ in self.formulations[self.samples[i].formulation_id].components}

    def validation_report(self) -> dict[str, object]:
        violations = [sample.sample_id for sample in self.samples
                      if sample.targets[1] < sample.targets[0]
                      or sample.targets[2] < sample.targets[3]]
        return {
            "compounds": len(self.compounds), "formulations": len(self.formulations),
            "samples": len(self.samples),
            "ingredient_sets": len({f.ingredient_set_id for f in self.formulations.values()}),
            "order_violation_sample_ids": violations,
            "input_sha256": self.input_hashes,
        }


def _text(value: str, label: str) -> str:
    if not value or value != value.strip() or any(not c.isprintable() for c in value):
        raise ValueError(f"{label} must be nonempty, trimmed, printable text")
    return value


def _rows(payload: bytes, name: str) -> list[dict[str, str]]:
    reader = csv.reader(io.StringIO(payload.decode("utf-8"), newline=""), strict=True)
    header = next(reader, [])
    if tuple(header) != CSV_HEADERS[name]:
        raise ValueError(f"{name}: expected exact header {','.join(CSV_HEADERS[name])}")
    records = []
    for line, row in enumerate(reader, start=2):
        if len(row) != len(header):
            raise ValueError(f"{name}:{line}: wrong number of columns (blank rows forbidden)")
        records.append(dict(zip(header, row, strict=True)))
    if not records:
        raise ValueError(f"{name} must contain data rows")
    return records


def load_dataset(directory: Path) -> Dataset:
    payloads = {name: (directory / name).read_bytes() for name in CSV_HEADERS}
    compounds: dict[str, str] = {}
    smiles_seen: set[str] = set()
    for row in _rows(payloads["compounds.csv"], "compounds.csv"):
        cid = _text(row["compound_id"], "compound_id")
        smiles = _text(row["canonical_smiles"], "canonical_smiles")
        if cid in compounds:
            raise ValueError(f"duplicate compound_id: {cid}")
        if smiles in smiles_seen:
            raise ValueError("duplicate canonical_smiles: merge same-molecule aliases first")
        compounds[cid] = smiles
        smiles_seen.add(smiles)

    components: dict[str, dict[str, Fraction]] = {}
    for row in _rows(payloads["formulation_components.csv"], "formulation_components.csv"):
        fid = _text(row["formulation_id"], "formulation_id")
        cid = row["compound_id"]
        if cid not in compounds:
            raise ValueError(f"unknown compound_id: {cid}")
        try:
            value = Decimal(row["mass_fraction"])
        except InvalidOperation as exc:
            raise ValueError("mass_fraction must be a decimal number") from exc
        if not value.is_finite() or not 0 < value <= 1:
            raise ValueError("mass_fraction must be finite and in (0, 1]; percentages forbidden")
        if float(value) < 1.1754943508222875e-38:
            raise ValueError("mass_fraction must be representable as a positive normal float32")
        parts = components.setdefault(fid, {})
        if cid in parts:
            raise ValueError(f"duplicate component: {fid}, {cid}")
        parts[cid] = Fraction(value)

    formulations: dict[str, Formulation] = {}
    compositions: dict[tuple[tuple[str, Fraction], ...], str] = {}
    model_compositions: dict[tuple[tuple[str, bytes], ...], str] = {}
    for fid, parts in sorted(components.items()):
        total = sum(parts.values(), Fraction())
        if abs(total - 1) > Fraction(FRACTION_TOLERANCE):
            raise ValueError(f"{fid}: mass fractions must sum to 1 within 1e-6")
        # Exact normalized rational identity catches decimal spelling and scaling aliases.
        identity = tuple((cid, fraction / total) for cid, fraction in sorted(parts.items()))
        if identity in compositions:
            raise ValueError(f"duplicate composition: {fid} and {compositions[identity]}")
        model_identity = tuple((cid, struct.pack("<f", float(fraction)))
                               for cid, fraction in sorted(parts.items()))
        if model_identity in model_compositions:
            raise ValueError(f"duplicate float32 composition: {fid} and "
                             f"{model_compositions[model_identity]}")
        compositions[identity] = fid
        model_compositions[model_identity] = fid
        set_id = digest(json_bytes(sorted(parts)))
        formulations[fid] = Formulation(fid, tuple(sorted(parts.items())), set_id)

    samples: dict[str, Sample] = {}
    for row in _rows(payloads["measurements.csv"], "measurements.csv"):
        sid = _text(row["sample_id"], "sample_id")
        fid = row["formulation_id"]
        if sid in samples:
            raise ValueError(f"duplicate sample_id: {sid}")
        if fid not in formulations:
            raise ValueError(f"unknown formulation_id: {fid}")
        try:
            targets = tuple(float(row[target]) for target in TARGETS)
        except ValueError as exc:
            raise ValueError(f"{sid}: all four targets must be numeric and complete") from exc
        if any(not math.isfinite(value) or abs(value) > 3.4028234663852886e38
               for value in targets):
            raise ValueError(f"{sid}: targets must be finite and representable as float32")
        samples[sid] = Sample(sid, fid, targets)
    if {s.formulation_id for s in samples.values()} != set(formulations):
        raise ValueError("every formulation must have at least one measurement")
    return Dataset(dict(sorted(compounds.items())), formulations,
                   tuple(samples[sid] for sid in sorted(samples)),
                   {name: digest(payload) for name, payload in payloads.items()})
