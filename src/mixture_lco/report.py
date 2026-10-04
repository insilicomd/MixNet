"""Descriptive errors with explicit formulation and membership weighting."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from mixture_lco.data import TARGETS


@dataclass(frozen=True)
class Prediction:
    run_id: str
    representation: str
    sample_id: str
    formulation_id: str
    ingredient_set_id: str
    ingredient: str
    arm: str
    seed: int
    fold: int
    target: str
    y_true: float
    y_pred: float
    held_out_mass: float
    unseen_mass: float
    id_all_ingredients_seen: bool


def mass_band(value: float) -> str:
    return "below-0.10" if value < 0.1 else "0.10-to-below-0.20" if value < 0.2 else "at-least-0.20"


def summarize(predictions: list[Prediction]) -> dict[str, Any]:
    if not predictions:
        raise ValueError("cannot summarize empty predictions")
    identities = [(p.ingredient, p.arm, p.seed, p.sample_id, p.target) for p in predictions]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate prediction identity")
    groups: dict[tuple[str, str, str], list[Prediction]] = defaultdict(list)
    for row in predictions:
        groups[row.ingredient, row.formulation_id, row.arm].append(row)
    formulations: list[dict[str, Any]] = []
    for (ingredient, fid, arm), rows in sorted(groups.items()):
        cells: dict[tuple[int, str], list[float]] = defaultdict(list)
        for row in rows:
            cells[row.seed, row.target].append(abs(row.y_pred - row.y_true))
        seeds = sorted({r.seed for r in rows})
        sample_ids = {r.sample_id for r in rows}
        if set(cells) != {(seed, target) for seed in seeds for target in TARGETS} or any(
            len(values) != len(sample_ids) for values in cells.values()
        ):
            raise ValueError("incomplete sample/seed/target prediction grid")
        per_target = {target: mean(mean(cells[seed, target]) for seed in seeds)
                      for target in TARGETS}
        formulations.append({
            "ingredient": ingredient, "formulation_id": fid, "arm": arm,
            "record_count": len(sample_ids), "seed_count": len(seeds),
            "target_mae": per_target, "macro_mae": mean(per_target.values()),
            "held_out_mass": rows[0].held_out_mass,
            "unseen_mass_min": min(r.unseen_mass for r in rows),
            "unseen_mass_max": max(r.unseen_mass for r in rows),
            "id_all_ingredients_seen": all(r.id_all_ingredients_seen for r in rows),
        })
    result: dict[str, Any] = {
        "statistical_status": "descriptive only; no significance tests, intervals, or winner",
        "weighting": "absolute error -> mean over records per formulation/seed/target -> "
                     "mean over targets and seed errors -> equal formulation membership weights",
        "formulations": formulations,
    }
    cv = [r for r in formulations if r["arm"] == "cv"]
    if cv:
        result["cv"] = {
            "status": "selection-aware: the same folds tune, early-stop, and score",
            "formulation_count": len(cv), "macro_mae": mean(r["macro_mae"] for r in cv),
            "target_mae": {t: mean(r["target_mae"][t] for r in cv) for t in TARGETS},
        }
    else:
        indexed = {(r["ingredient"], r["formulation_id"], r["arm"]): r for r in formulations}
        memberships = []
        for lco in formulations:
            if lco["arm"] != "lco":
                continue
            matched = indexed.get((lco["ingredient"], lco["formulation_id"], "id"))
            if matched is None:
                raise ValueError("missing ID arm for LCO membership")
            if lco["unseen_mass_min"] != lco["unseen_mass_max"]:
                raise ValueError("LCO mass strata require seed-independent training vocabulary")
            memberships.append({
                "ingredient": lco["ingredient"], "formulation_id": lco["formulation_id"],
                "band": mass_band(lco["unseen_mass_min"]),
                "pooled_eligible": matched["id_all_ingredients_seen"],
                "id_macro_mae": matched["macro_mae"], "lco_macro_mae": lco["macro_mae"],
                "increase": lco["macro_mae"] - matched["macro_mae"],
            })

        def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                "membership_count": len(rows),
                "distinct_formulation_count": len({r["formulation_id"] for r in rows}),
                **{key: mean(r[key] for r in rows) if rows else None
                   for key in ("id_macro_mae", "lco_macro_mae", "increase")},
            }

        result["memberships"] = memberships
        result["all_ingredient_memberships"] = {
            ingredient: aggregate([r for r in memberships if r["ingredient"] == ingredient])
            for ingredient in sorted({r["ingredient"] for r in memberships})}
        eligible = [r for r in memberships if r["pooled_eligible"]]
        result["paired_pooled_eligible"] = {
            "all": aggregate(eligible),
            "bands": {band: aggregate([r for r in eligible if r["band"] == band])
                      for band in ("below-0.10", "0.10-to-below-0.20", "at-least-0.20")},
            "excluded_membership_count": len(memberships) - len(eligible),
            "eligibility": "every test ingredient occurs in ID training in every seed",
            "weighting": "equal ingredient-formulation memberships; repeated formulations "
                         "across ingredients are counted repeatedly, not independent observations",
        }
    result["prediction_order_violation_note"] = "No measured or predicted targets are reordered."
    return result


def prediction_dicts(predictions: list[Prediction]) -> list[dict[str, Any]]:
    return [asdict(row) for row in predictions]
