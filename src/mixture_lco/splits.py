"""Target-free grouped split construction with explicit memberships."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math

from mixture_lco.config import Settings
from mixture_lco.data import Dataset
from mixture_lco.io import digest, json_bytes


@dataclass(frozen=True)
class Fold:
    train: list[int]
    validation: list[int]


def grouped_folds(dataset: Dataset, rows: list[int], count: int, seed: int,
                  *, ingredient_sets: bool) -> list[Fold]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        fid = dataset.samples[row].formulation_id
        key = dataset.formulations[fid].ingredient_set_id if ingredient_sets else fid
        groups[key].append(row)
    if len(groups) < count:
        raise ValueError(f"need at least {count} distinct split groups; got {len(groups)}")
    ordered = sorted(groups, key=lambda key: digest(json_bytes([seed, key])))
    buckets: list[list[int]] = [[] for _ in range(count)]
    for key in ordered:
        bucket = min(range(count), key=lambda i: (len(buckets[i]), i))
        buckets[bucket].extend(groups[key])
    folds = [Fold(sorted(set(rows) - set(bucket)), sorted(bucket)) for bucket in buckets]
    if any(len(fold.train) < 2 for fold in folds):
        raise ValueError("each training fold needs at least two measurement records")
    return folds


def lco_memberships(dataset: Dataset, ingredient: str, settings: Settings) -> dict[str, object]:
    if ingredient not in dataset.compounds:
        raise ValueError(f"held-out ingredient is not in registry: {ingredient}")
    evaluation_fids = {fid for fid, formulation in dataset.formulations.items()
                       if ingredient in dict(formulation.components)}
    free = dataset.rows(set(dataset.formulations) - evaluation_fids)
    evaluation = dataset.rows(evaluation_fids)
    if not evaluation:
        raise ValueError("held-out ingredient has no measured formulations")
    cfg = settings.payload
    groups: dict[str, list[int]] = defaultdict(list)
    for row in free:
        groups[dataset.formulations[dataset.samples[row].formulation_id].ingredient_set_id].append(row)
    ordered = sorted(groups, key=lambda key: digest(json_bytes([cfg["split_seed"], key])))
    validation: list[int] = []
    minimum = math.ceil(len(free) * cfg["validation_fraction"])
    for key in ordered:
        if len(validation) >= minimum:
            break
        validation.extend(groups[key])
    train = sorted(set(free) - set(validation))
    if not validation or len(train) < 2:
        raise ValueError("X-free split needs nonempty validation and at least two training records")
    crossfit = grouped_folds(dataset, evaluation, cfg["eval_folds"], cfg["split_seed"],
                             ingredient_sets=True)
    # HPO uses all X-free rows, including the eventual early-stopping validation partition.
    hpo = {str(seed): [asdict(fold) for fold in grouped_folds(
        dataset, free, cfg["hpo_folds"], seed, ingredient_sets=True)] for seed in settings.seeds}
    return {"ingredient": ingredient, "free": free, "train": train,
            "validation": sorted(validation), "evaluation": evaluation,
            "crossfit": [asdict(fold) for fold in crossfit], "hpo": hpo}


def sample_membership(dataset: Dataset, rows: list[int]) -> list[str]:
    return [dataset.samples[i].sample_id for i in rows]
