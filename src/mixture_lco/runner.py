"""Offline CPU orchestration; training never receives held-out evaluation labels."""

from __future__ import annotations

from dataclasses import asdict
import csv
from importlib.metadata import version
from pathlib import Path
import platform
from typing import Any

import numpy as np
import torch

from mixture_lco import __version__
from mixture_lco.config import Settings, load_settings
from mixture_lco.data import Dataset, TARGETS, load_dataset
from mixture_lco.features import ComponentFeatureSet, build_features, subset
from mixture_lco.io import append_json, digest, fresh_directory, json_bytes, write_json
from mixture_lco.mixnet import MixNet
from mixture_lco.report import Prediction, prediction_dicts, summarize
from mixture_lco.splits import grouped_folds, lco_memberships, sample_membership
from mixture_lco.train import (TrainResult, ensure_torch_runtime, load_checkpoint,
                               predict_component_targets, serialize_checkpoint, train_run)


def source_environment() -> dict[str, Any]:
    root = Path(__file__).parent
    source = {p.relative_to(root).as_posix(): digest(p.read_bytes())
              for p in sorted(root.rglob("*")) if p.suffix in (".py", ".json")}
    environment = {
        "python": platform.python_version(), "system": platform.system(),
        "release": platform.release(), "machine": platform.machine(),
        "dependencies": {name: version(name) for name in ("numpy", "torch")},
        "torch_build_sha256": digest(torch.__config__.show().encode()),
        "device": "cpu", "threads": 1, "interop_threads": 1,
        "deterministic_algorithms": True,
    }
    return {"package_version": __version__, "source_files": source,
            "source_sha256": digest(json_bytes(source)), "environment": environment,
            "environment_sha256": digest(json_bytes(environment))}


class Experiment:
    def __init__(self, dataset: Dataset, features: ComponentFeatureSet, settings: Settings,
                 output: Path, run_id: str, representation: str) -> None:
        self.dataset = dataset
        self.features = features
        self.settings = settings
        self.output = output
        self.run_id = run_id
        self.representation = representation
        self.fit_count = 0

    def fit(self, train: list[int], validation: list[int], candidate: dict[str, Any],
            seed: int, context: dict[str, Any], *, retain: bool = False) -> TrainResult:
        if set(train) & set(validation):
            raise ValueError("training and validation overlap")
        rows = sorted(set(train) | set(validation))
        lookup = {row: i for i, row in enumerate(rows)}
        # Subsetting is the label boundary: no other rows enter train_run, including its
        # target validation and normalizer preparation. Prediction takes features only.
        features = subset(self.features, rows)
        targets = np.asarray([self.dataset.samples[i].targets for i in rows], dtype=np.float64)
        model_config = self.settings.model(features.dimension, candidate)
        torch.manual_seed(seed)
        initial_model = MixNet(model_config)
        init_hash = digest(serialize_checkpoint(
            initial_model, model_config=model_config, best_epoch=0,
            target_names=TARGETS, target_means=(0.0,) * 4, target_stds=(1.0,) * 4))
        del initial_model
        self.fit_count += 1
        receipt: dict[str, Any] = {
            **context, "fit_id": self.fit_count, "seed": seed, "settings": candidate,
            "model_config": model_config.to_dict(), "initial_state_sha256": init_hash,
            "train_samples": sample_membership(self.dataset, train),
            "validation_samples": sample_membership(self.dataset, validation),
        }
        try:
            trained = train_run(
                model_config=model_config, features=features, targets=targets,
                target_names=TARGETS, train_indices=[lookup[i] for i in train],
                val_indices=[lookup[i] for i in validation], seed=seed,
                learning_rate=candidate["learning_rate"], weight_decay=candidate["weight_decay"],
                batch_size=candidate["batch_size"], max_epochs=self.settings.payload["max_epochs"],
                patience=self.settings.payload["patience"],
            )
            header = load_checkpoint(trained.checkpoint_bytes)
            receipt.update(status="completed", best_epoch=trained.best_epoch,
                           validation_macro_mae=trained.best_val_macro_mae,
                           validation_history=list(trained.val_macro_mae_history),
                           checkpoint_sha256=digest(trained.checkpoint_bytes),
                           target_means=list(header.target_means), target_stds=list(header.target_stds))
            if retain:
                name = f"fit-{self.fit_count:06d}.weights"
                (self.output / "checkpoints" / name).write_bytes(trained.checkpoint_bytes)
                receipt["checkpoint"] = f"checkpoints/{name}"
        except (ValueError, RuntimeError) as exc:
            receipt.update(status="failed", error=str(exc))
            append_json(self.output / "fits.jsonl", receipt)
            raise
        append_json(self.output / "fits.jsonl", receipt)
        return trained

    def tune(self, folds: list[dict[str, list[int]]], seed: int,
             ingredient: str) -> dict[str, Any]:
        best: tuple[float, int, dict[str, Any]] | None = None
        for trial, candidate in enumerate(self.settings.candidates(seed)):
            record: dict[str, Any] = {"ingredient": ingredient, "seed": seed,
                                      "trial": trial, "settings": candidate}
            scores: list[float] = []
            failures: list[dict[str, Any]] = []
            # Every trial attempts every fold, including after a failed fold. No retry
            # or replacement trials silently increase one representation's fit budget.
            for fold, assignment in enumerate(folds):
                try:
                    fitted = self.fit(assignment["train"], assignment["validation"], candidate,
                                      seed, {"phase": "hpo", "ingredient": ingredient,
                                             "trial": trial, "fold": fold})
                    scores.append(fitted.best_val_macro_mae)
                except (ValueError, RuntimeError) as exc:
                    failures.append({"fold": fold, "error": str(exc)})
            if failures:
                record.update(status="failed", failures=failures, fold_scores=scores)
            else:
                score = float(np.mean(scores))
                record.update(status="completed", score=score, fold_scores=scores)
                if best is None or (score, trial) < best[:2]:
                    best = (score, trial, candidate)
            append_json(self.output / "trials.jsonl", record)
        if best is None:
            raise ValueError("all HPO trials failed; inspect trials.jsonl and fits.jsonl")
        append_json(self.output / "chosen.jsonl", {
            "ingredient": ingredient, "seed": seed, "trial": best[1], "score": best[0],
            "settings": best[2], "objective": "equal fold mean of record-weighted validation macro-MAE",
        })
        return best[2]

    def predict(self, fitted: TrainResult, rows: list[int], train: list[int], *,
                ingredient: str, arm: str, seed: int, fold: int,
                id_train: list[int], batch_size: int) -> list[Prediction]:
        inferred = predict_component_targets(
            load_checkpoint(fitted.checkpoint_bytes), features=subset(self.features, rows),
            row_indices=list(range(len(rows))), batch_size=batch_size)
        vocabulary = self.dataset.vocabulary(train)
        id_vocabulary = self.dataset.vocabulary(id_train)
        predictions = []
        for local, index in enumerate(rows):
            sample = self.dataset.samples[index]
            formulation = self.dataset.formulations[sample.formulation_id]
            parts = dict(formulation.components)
            # Sum exact input fractions before the final float conversion, so decimal
            # boundary masses such as 0.09 + 0.01 stay in the correct reporting band.
            unseen = float(sum(w for cid, w in parts.items() if cid not in vocabulary))
            for target_index, target in enumerate(TARGETS):
                predictions.append(Prediction(
                    self.run_id, self.representation, sample.sample_id, sample.formulation_id,
                    formulation.ingredient_set_id, ingredient, arm, seed, fold, target,
                    sample.targets[target_index], float(inferred.y_pred[target][local]),
                    float(parts.get(ingredient, 0)), unseen, set(parts) <= id_vocabulary))
        return predictions


def run(directory: Path, output: Path, *, protocol: str, ingredients: list[str] | None = None,
        embeddings: Path | None = None, config: Path | None = None,
        preset: str = "paper") -> dict[str, Any]:
    if protocol not in ("cv", "lco"):
        raise ValueError("protocol must be cv or lco")
    ingredients = sorted(ingredients or [])
    if (protocol == "lco" and not ingredients) or (protocol == "cv" and ingredients):
        raise ValueError("LCO requires explicit ingredient IDs; CV accepts none")
    if len(set(ingredients)) != len(ingredients):
        raise ValueError("held-out ingredient IDs must be unique")
    dataset = load_dataset(directory)
    features, representation = build_features(dataset, embeddings)
    settings, config_hash = load_settings(config, preset)
    splits: dict[str, Any] = {
        "row_identity": [{"row": i, "sample_id": sample.sample_id,
                          "formulation_id": sample.formulation_id,
                          "ingredient_set_id": dataset.formulations[
                              sample.formulation_id].ingredient_set_id}
                         for i, sample in enumerate(dataset.samples)],
    }
    if protocol == "cv":
        splits["cv"] = {str(seed): [asdict(fold) for fold in grouped_folds(
            dataset, list(range(len(dataset.samples))), settings.payload["cv_folds"], seed,
            ingredient_sets=False)] for seed in settings.seeds}
    else:
        splits["lco"] = {x: lco_memberships(dataset, x, settings) for x in ingredients}
    ensure_torch_runtime()
    provenance = {**source_environment(), "input_sha256": dataset.input_hashes,
                  "config_file_sha256": config_hash, "config": settings.payload,
                  "representation": representation, "protocol": protocol, "ingredients": ingredients,
                  "split_sha256": digest(json_bytes(splits))}
    run_id = digest(json_bytes(provenance))
    manifest = {**provenance, "run_id": run_id, "status": "running"}
    fresh_directory(output)
    (output / "checkpoints").mkdir()
    write_json(output / "manifest.json", manifest)
    write_json(output / "splits.json", splits)
    write_json(output / "validation.json", dataset.validation_report())
    experiment = Experiment(dataset, features, settings, output, run_id, str(representation["kind"]))
    predictions = []
    try:
        for seed in settings.seeds:
            if protocol == "cv":
                folds = splits["cv"][str(seed)]
                candidate = experiment.tune(folds, seed, "")
                for fold, assignment in enumerate(folds):
                    train, validation = assignment["train"], assignment["validation"]
                    fitted = experiment.fit(train, validation, candidate, seed,
                                            {"phase": "cv", "fold": fold, "ingredient": ""},
                                            retain=True)
                    predictions.extend(experiment.predict(
                        fitted, validation, train, ingredient="", arm="cv", seed=seed,
                        fold=fold, id_train=train, batch_size=candidate["batch_size"]))
            else:
                for ingredient in ingredients:
                    assignment = splits["lco"][ingredient]
                    candidate = experiment.tune(assignment["hpo"][str(seed)], seed, ingredient)
                    train, validation = assignment["train"], assignment["validation"]
                    lco = experiment.fit(train, validation, candidate, seed,
                                         {"phase": "lco", "ingredient": ingredient, "fold": None},
                                         retain=True)
                    for fold, crossfit in enumerate(assignment["crossfit"]):
                        id_train = sorted(train + crossfit["train"])
                        evaluation = crossfit["validation"]
                        matched = experiment.fit(id_train, validation, candidate, seed,
                                                 {"phase": "id", "ingredient": ingredient,
                                                  "fold": fold}, retain=True)
                        for arm, fitted, training in (("lco", lco, train), ("id", matched, id_train)):
                            predictions.extend(experiment.predict(
                                fitted, evaluation, training, ingredient=ingredient, arm=arm,
                                seed=seed, fold=fold, id_train=id_train,
                                batch_size=candidate["batch_size"]))
        predictions.sort(key=lambda p: (p.ingredient, p.arm, p.seed, p.fold, p.sample_id, p.target))
        records = prediction_dicts(predictions)
        with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
        summary = summarize(predictions)
        write_json(output / "summary.json", summary)
        manifest.update(status="completed", fit_count=experiment.fit_count)
        manifest["artifacts_sha256"] = {
            p.relative_to(output).as_posix(): digest(p.read_bytes())
            for p in sorted(output.rglob("*")) if p.is_file() and p.name != "manifest.json"}
        write_json(output / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="failed", error=str(exc), fit_count=experiment.fit_count)
        write_json(output / "manifest.json", manifest)
        raise
    return manifest
