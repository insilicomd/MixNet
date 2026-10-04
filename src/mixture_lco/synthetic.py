"""Invented software fixtures using arbitrary simple public molecules."""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

from mixture_lco.data import CSV_HEADERS
from mixture_lco.io import fresh_directory, write_json


def generate(output: Path) -> None:
    fresh_directory(output)
    # These molecules and numbers are not a thermochromic formulation or a scientific dataset.
    smiles = ("O", "C", "CC", "CCC", "CO", "CCO", "CC(=O)O", "N")
    registry = [(f"mol-{i:02d}", value) for i, value in enumerate(smiles, 1)]
    free_sets = list(combinations(range(2, 7), 2))[:8]
    evaluation_sets = [(1, i) for i in range(2, 8)]
    components: list[tuple[str, str, str]] = []
    measurements: list[tuple[object, ...]] = []
    for i, (left, right) in enumerate(free_sets + evaluation_sets, 1):
        for j, fraction in enumerate((0.08, 0.3), 1):
            fid = f"toy-f-{i:02d}-{j}"
            components.extend(((fid, f"mol-{left:02d}", str(fraction)),
                               (fid, f"mol-{right:02d}", str(1 - fraction))))
            for repeat in range(2 + (i == 1 and j == 1)):
                value = 2 * left * fraction + right * (1 - fraction) + 0.25 * repeat
                measurements.append((f"toy-s-{i:02d}-{j}-{repeat}", fid,
                                     value + 10, value + 15, value + 8, value + 3))
    tables: dict[str, list[tuple[object, ...]]] = {
        "compounds.csv": list(registry), "formulation_components.csv": list(components),
        "measurements.csv": measurements}
    for name, rows in tables.items():
        with (output / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(CSV_HEADERS[name])
            writer.writerows(rows)
    write_json(output / "embeddings.json", {
        cid: [float(((i + 1) * (j + 3)) % 11) / 10 for j in range(5)]
        for i, (cid, _) in enumerate(registry)})
