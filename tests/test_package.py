"""Behavioral contract tests; all inputs are invented software fixtures."""

from __future__ import annotations

import csv
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from mixture_lco.config import load_settings, parse_settings
from mixture_lco.data import TARGETS, load_dataset
from mixture_lco.features import build_features
from mixture_lco.io import read_json, write_json
from mixture_lco.report import Prediction, summarize
from mixture_lco.runner import run
from mixture_lco.splits import grouped_folds, lco_memberships
from mixture_lco.synthetic import generate

ROOT = Path(__file__).resolve().parents[1]


def json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name) / "data"
        generate(self.data)

    def mutate(self, filename, change):
        path = self.data / filename
        with path.open(newline="") as stream:
            rows = list(csv.reader(stream))
        change(rows)
        with path.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)

    def test_generated_fixture_matches_checked_in_inputs(self):
        for path in self.data.iterdir():
            self.assertEqual(path.read_bytes(), (ROOT / "examples/tiny" / path.name).read_bytes())
        data = load_dataset(self.data)
        self.assertEqual(len(data.samples), 57)
        self.assertEqual(len(data.formulations), 28)
        self.assertEqual(data.validation_report()["ingredient_sets"], 14)

    def test_invalid_inputs_fail_closed(self):
        changes = [
            ("formulation_components.csv", lambda r: r[1].__setitem__(1, "unknown")),
            ("formulation_components.csv", lambda r: r.append(r[1])),
            ("formulation_components.csv", lambda r: r.extend(
                [["duplicate-f", *r[1][1:]], ["duplicate-f", *r[2][1:]]])),
            ("formulation_components.csv", lambda r: r[1].__setitem__(2, "-0.08")),
            ("formulation_components.csv", lambda r: r[1].__setitem__(2, "8")),
            ("formulation_components.csv", lambda r: r[1].__setitem__(2, "0")),
            ("formulation_components.csv", lambda r: r[1].__setitem__(2, "NaN")),
            ("formulation_components.csv", lambda r: r[1].__setitem__(2, "0.1")),
            ("measurements.csv", lambda r: r[1].__setitem__(2, "nan")),
            ("measurements.csv", lambda r: r[1].__setitem__(3, "inf")),
            ("measurements.csv", lambda r: r[1].__setitem__(4, "")),
            ("measurements.csv", lambda r: r[1].__setitem__(1, "unknown-f")),
            ("measurements.csv", lambda r: r.append(r[1])),
            ("measurements.csv", lambda r: r[0].append("lot")),
            ("compounds.csv", lambda r: r.append(r[1])),
            ("compounds.csv", lambda r: r[2].__setitem__(1, r[1][1])),
        ]
        for filename, change in changes:
            with self.subTest(filename=filename, change=changes.index((filename, change))):
                original = (self.data / filename).read_bytes()
                self.mutate(filename, change)
                with self.assertRaises(ValueError):
                    load_dataset(self.data)
                (self.data / filename).write_bytes(original)

    def test_duplicate_composition_decimal_spellings_and_scaled_sum(self):
        def add(rows):
            rows.extend([["alias", rows[1][1], "0.080000008"],
                         ["alias", rows[2][1], "0.920000092"]])
        self.mutate("formulation_components.csv", add)
        with self.assertRaisesRegex(ValueError, "duplicate composition"):
            load_dataset(self.data)

    def test_float32_composition_collision_is_rejected(self):
        def add(rows):
            rows.extend([["alias", rows[1][1], "0.08000000000001"],
                         ["alias", rows[2][1], "0.91999999999999"]])
        self.mutate("formulation_components.csv", add)
        with self.assertRaisesRegex(ValueError, "duplicate float32 composition"):
            load_dataset(self.data)

    def test_csv_fraction_tolerance_edges_train_without_renormalization(self):
        for suffix, fractions in (("lower", ["0.333333"] * 3),
                                  ("upper", ["0.333333", "0.333333", "0.333335"])):
            def replace_first(rows):
                first = rows[1][0]
                rows[:] = [rows[0], *[r for r in rows[1:] if r[0] != first],
                           *[[first, f"mol-{i + 1:02}", value]
                             for i, value in enumerate(fractions)]]
            self.mutate("formulation_components.csv", replace_first)
            data = load_dataset(self.data)
            features, _ = build_features(data)
            actual = next(s.mass_fractions for s in features.samples if len(s.compound_ids) == 3)
            np.testing.assert_array_equal(actual, np.asarray(fractions, dtype=np.float64))
            output = Path(self.temp.name) / f"cv-{suffix}"
            run(self.data, output, protocol="cv", preset="smoke")
            self.assertEqual(read_json(output / "manifest.json")["status"], "completed")
            # Restore the fixture before exercising the opposite exact tolerance boundary.
            shutil.copyfile(ROOT / "examples/tiny/formulation_components.csv",
                            self.data / "formulation_components.csv")
        def outside(rows):
            first = rows[1][0]
            rows[:] = [rows[0], *[r for r in rows[1:] if r[0] != first],
                       [first, "mol-01", "0.333333"], [first, "mol-02", "0.333333"],
                       [first, "mol-03", "0.333335000001"]]
        self.mutate("formulation_components.csv", outside)
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            load_dataset(self.data)

    def test_order_violations_are_reported_without_correction(self):
        self.mutate("measurements.csv", lambda rows: rows[1].__setitem__(2, "100"))
        data = load_dataset(self.data)
        self.assertEqual(data.samples[0].targets[0], 100)
        self.assertIn(data.samples[0].sample_id, data.validation_report()["order_violation_sample_ids"])

    def test_embedding_validation_and_fixed_one_hot_vocabulary(self):
        data = load_dataset(self.data)
        features, info = build_features(data)
        self.assertEqual(info["compound_vocabulary"], sorted(data.compounds))
        self.assertEqual(features.dimension, 8)  # Includes unused mol-08 and withheld mol-01.
        x = next(s for s in features.samples if "mol-01" in s.compound_ids)
        np.testing.assert_array_equal(x.embeddings[0], np.eye(8)[0])
        path = self.data / "embeddings.json"
        original = read_json(path)
        variants = [dict(original, unknown=[1] * 5), {k: v for k, v in original.items() if k != "mol-01"},
                    dict(original, **{"mol-01": [1, 2]}),
                    dict(original, **{"mol-01": [True] * 5}),
                    dict(original, **{"mol-01": [1e100] * 5})]
        for raw in variants:
            write_json(path, raw)
            with self.assertRaises(ValueError):
                build_features(data, path)
        path.write_text('{"mol-01": [1], "mol-01": [2]}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            build_features(data, path)

    def test_repeated_record_and_ingredient_set_grouping(self):
        data = load_dataset(self.data)
        settings, _ = load_settings(preset="smoke")
        folds = grouped_folds(data, list(range(len(data.samples))), 3, 42, ingredient_sets=False)
        for fold in folds:
            train_fids = {data.samples[i].formulation_id for i in fold.train}
            val_fids = {data.samples[i].formulation_id for i in fold.validation}
            self.assertFalse(train_fids & val_fids)
        split = lco_memberships(data, "mol-01", settings)
        self.assertNotIn("mol-01", data.vocabulary(split["train"] + split["validation"]))

        def sets(rows):
            return {data.formulations[data.samples[i].formulation_id].ingredient_set_id for i in rows}
        self.assertFalse(sets(split["train"]) & sets(split["validation"]))
        for fold in split["crossfit"]:
            self.assertFalse(sets(fold["train"]) & sets(fold["validation"]))
        for fold in split["hpo"]["17"]:
            self.assertFalse(sets(fold["train"]) & sets(fold["validation"]))
            self.assertFalse(set(fold["train"] + fold["validation"]) & set(split["evaluation"]))
        with self.assertRaises(ValueError):
            grouped_folds(data, [0, 1], 3, 42, ingredient_sets=True)

    def test_config_schema_and_trial_budget(self):
        settings, _ = load_settings()
        self.assertEqual(len(settings.candidates(17)), 40)
        self.assertEqual(settings.payload["max_epochs"], 60)
        self.assertEqual(settings.payload["patience"], 8)
        self.assertEqual(settings.candidates(17), settings.candidates(17))
        for key, value in (("seeds", [True]), ("trials", 0), ("validation_fraction", 1),
                           ("unexpected", 42)):
            raw = dict(settings.payload, **{key: value})
            with self.assertRaises(ValueError):
                parse_settings(raw)


class ReportingTests(unittest.TestCase):
    def test_known_answer_record_target_seed_and_membership_weights(self):
        rows = []
        for fid, errors, seen in (("f1", [1.0, 3.0], True), ("f2", [10.0], False)):
            for arm in ("id", "lco"):
                for seed in (17, 42):
                    for record, error in enumerate(errors):
                        for ti, target in enumerate(TARGETS):
                            # f1 = 2 + 1.5(target) + 1(seed) = 4.5; f2 = 12.5.
                            value = error + ti + (2 if seed == 42 else 0) + (3 if arm == "lco" else 0)
                            rows.append(Prediction("run", "toy", f"{fid}-{record}", fid, "set", "X",
                                                   arm, seed, 0, target, 0, value, .3,
                                                   .3 if arm == "lco" else 0, seen))
        report = summarize(rows)
        all_members = report["all_ingredient_memberships"]["X"]
        self.assertEqual(all_members["id_macro_mae"], 8.5)  # Equal formulations, not 43/6.
        self.assertEqual(all_members["lco_macro_mae"], 11.5)
        self.assertEqual(all_members["increase"], 3)
        pooled = report["paired_pooled_eligible"]
        self.assertEqual(pooled["all"]["membership_count"], 1)
        self.assertEqual(pooled["all"]["id_macro_mae"], 4.5)
        self.assertEqual(pooled["excluded_membership_count"], 1)
        # Two ingredients count twice as memberships but once as distinct formulation.
        report = summarize(rows + [replace(r, ingredient="Y") for r in rows])
        self.assertEqual(report["paired_pooled_eligible"]["all"]["membership_count"], 2)
        self.assertEqual(report["paired_pooled_eligible"]["all"]["distinct_formulation_count"], 1)
        with self.assertRaises(ValueError):
            summarize(rows + [rows[0]])


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.data = cls.root / "data"
        generate(cls.data)
        cls.outputs = {}
        for protocol in ("cv", "lco"):
            for representation in ("one-hot", "supplied"):
                output = cls.root / f"{protocol}-{representation}"
                run(cls.data, output, protocol=protocol, preset="smoke",
                    ingredients=["mol-01"] if protocol == "lco" else None,
                    embeddings=cls.data / "embeddings.json" if representation == "supplied" else None)
                cls.outputs[protocol, representation] = output

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_cpu_cv_and_lco_both_representations(self):
        for (protocol, representation), output in self.outputs.items():
            manifest = read_json(output / "manifest.json")
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["fit_count"], 6 if protocol == "cv" else 7)
            for name, checksum in manifest["artifacts_sha256"].items():
                self.assertEqual(sha256((output / name).read_bytes()).hexdigest(), checksum)
            with (output / "predictions.csv").open() as stream:
                predictions = list(csv.DictReader(stream))
            self.assertEqual(len(predictions), 57 * 4 if protocol == "cv" else 24 * 4 * 2)
            self.assertEqual({p["target"] for p in predictions}, set(TARGETS))
            self.assertIn("source_sha256", manifest)
            self.assertIn("environment_sha256", manifest)
            self.assertTrue(read_json(output / "summary.json")["formulations"])

    def test_equal_budget_identical_trials_and_matched_initialization(self):
        output = self.outputs["lco", "one-hot"]
        fits = json_lines(output / "fits.jsonl")
        arms = [row for row in fits if row["phase"] != "hpo"]
        self.assertEqual(len({r["initial_state_sha256"] for r in arms}), 1)
        self.assertEqual(len({json.dumps(r["settings"], sort_keys=True) for r in arms}), 1)
        self.assertEqual(len({tuple(r["validation_samples"]) for r in arms}), 1)
        split = read_json(output / "splits.json")
        evaluation_ids = {split["row_identity"][i]["sample_id"]
                          for i in split["lco"]["mol-01"]["evaluation"]}
        for fit in fits:
            self.assertFalse(set(fit["validation_samples"]) & evaluation_ids)
            if fit["phase"] != "id":
                self.assertFalse(set(fit["train_samples"]) & evaluation_ids)
        other = json_lines(self.outputs["lco", "supplied"] / "trials.jsonl")
        self.assertEqual([r["settings"] for r in json_lines(output / "trials.jsonl")],
                         [r["settings"] for r in other])

    def test_actual_unseen_mass_and_pooled_exclusions(self):
        output = self.outputs["lco", "one-hot"]
        with (output / "predictions.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        extra_unseen = [r for r in rows if r["formulation_id"].startswith("toy-f-14")]
        self.assertTrue(extra_unseen)
        self.assertTrue(all(r["id_all_ingredients_seen"] == "False" for r in extra_unseen))
        self.assertTrue(all(float(r["unseen_mass"]) == 1 for r in extra_unseen if r["arm"] == "lco"))
        self.assertTrue(all(float(r["held_out_mass"]) < 1 for r in extra_unseen))
        report = read_json(output / "summary.json")
        self.assertEqual(report["all_ingredient_memberships"]["mol-01"]["membership_count"], 12)
        self.assertGreaterEqual(report["paired_pooled_eligible"]["excluded_membership_count"], 2)

    def test_multiple_seeds_and_ingredients_preserve_paired_memberships(self):
        settings, _ = load_settings(preset="smoke")
        config = self.root / "multi-config.json"
        write_json(config, dict(settings.payload, seeds=[17, 42]))
        output = self.root / "multi-run"
        run(self.data, output, protocol="lco", config=config, ingredients=["mol-01", "mol-02"])
        fits = json_lines(output / "fits.jsonl")
        for ingredient in ("mol-01", "mol-02"):
            for seed in (17, 42):
                arms = [r for r in fits if r["ingredient"] == ingredient
                        and r["seed"] == seed and r["phase"] != "hpo"]
                self.assertEqual(len(arms), 3)
                self.assertEqual(len({r["initial_state_sha256"] for r in arms}), 1)
        summary = read_json(output / "summary.json")
        self.assertEqual(set(summary["all_ingredient_memberships"]), {"mol-01", "mol-02"})
        self.assertTrue(all(r["seed_count"] == 2 for r in summary["formulations"]))

    def test_decimal_unseen_mass_boundary_uses_exact_sum(self):
        modified = self.root / "boundary-data"
        shutil.copytree(self.data, modified)
        path = modified / "formulation_components.csv"
        with path.open(newline="") as stream:
            rows = list(csv.reader(stream))
        fid = "toy-f-14-1"
        for row in rows[1:]:
            if row[0] == fid:
                row[2] = "0.09" if row[1] == "mol-01" else "0.01"
        rows.append([fid, "mol-02", "0.90"])
        with path.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
        output = self.root / "boundary-run"
        run(modified, output, protocol="lco", preset="smoke", ingredients=["mol-01"])
        report = read_json(output / "summary.json")
        member = next(r for r in report["memberships"] if r["formulation_id"] == fid)
        self.assertEqual(member["band"], "0.10-to-below-0.20")
        formulation = next(r for r in report["formulations"]
                           if r["formulation_id"] == fid and r["arm"] == "lco")
        self.assertEqual(formulation["unseen_mass_min"], 0.1)

    def test_held_out_labels_cannot_select_hpo_or_same_fold_checkpoint(self):
        output = self.outputs["lco", "one-hot"]
        splits = read_json(output / "splits.json")
        heldout = splits["lco"]["mol-01"]["crossfit"][0]["validation"]
        ids = {splits["row_identity"][i]["sample_id"] for i in heldout}
        modified = self.root / "changed-targets"
        shutil.copytree(self.data, modified)
        path = modified / "measurements.csv"
        with path.open(newline="") as stream:
            rows = list(csv.reader(stream))
        for row in rows[1:]:
            if row[0] in ids:
                row[2:] = [str(float(value) + 1000) for value in row[2:]]
        with path.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
        changed = self.root / "changed-run"
        run(modified, changed, protocol="lco", preset="smoke", ingredients=["mol-01"])
        self.assertEqual((output / "chosen.jsonl").read_bytes(), (changed / "chosen.jsonl").read_bytes())
        before, after = json_lines(output / "fits.jsonl"), json_lines(changed / "fits.jsonl")
        for old, new in zip(before, after, strict=True):
            if old["phase"] != "id" or old["fold"] == 0:
                self.assertEqual(old["checkpoint_sha256"], new["checkpoint_sha256"])

    def test_deterministic_run_and_fresh_output_requirement(self):
        previous = self.outputs["lco", "one-hot"]
        output = self.root / "rerun"
        run(self.data, output, protocol="lco", preset="smoke", ingredients=["mol-01"])
        for path in previous.rglob("*"):
            if path.is_file():
                self.assertEqual(path.read_bytes(), (output / path.relative_to(previous)).read_bytes())
        with self.assertRaises(FileExistsError):
            run(self.data, output, protocol="lco", preset="smoke", ingredients=["mol-01"])

    def test_failed_trials_and_run_are_preserved(self):
        output = self.root / "failed-run"
        with patch("mixture_lco.runner.train_run", side_effect=ValueError("synthetic numeric failure")):
            with self.assertRaisesRegex(ValueError, "all HPO trials failed"):
                run(self.data, output, protocol="lco", preset="smoke", ingredients=["mol-01"])
        self.assertEqual(read_json(output / "manifest.json")["status"], "failed")
        self.assertEqual(len(json_lines(output / "trials.jsonl")), 2)
        self.assertEqual(len(json_lines(output / "fits.jsonl")), 4)
        self.assertTrue(all(r["status"] == "failed" for r in json_lines(output / "fits.jsonl")))


class DistributionTests(unittest.TestCase):
    def test_recorded_extraction_hashes(self):
        provenance = read_json(ROOT / "SOURCE_PROVENANCE.json")
        for entry in provenance["files"]:
            checksum = sha256((ROOT / entry["package_file"]).read_bytes()).hexdigest()
            self.assertEqual(checksum, entry["extracted_sha256"])
            if entry["package_file"].endswith("mixnet.py"):
                self.assertEqual(checksum, entry["original_sha256"])

    def test_reproducible_allowlist_export_excludes_unlisted_files(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp) / "source"
            for name in (ROOT / "FILES.txt").read_text().splitlines():
                dest = work / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, dest)
            (work / "results").mkdir()
            (work / "results/private.csv").write_text("must not ship")
            (work / "src/mixture_lco/unlisted.py").write_text("must not ship")
            archives = []
            for i in range(2):
                output = Path(temp) / f"export-{i}"
                subprocess.run([sys.executable, str(work / "scripts/export.py"), "--output", str(output)],
                               check=True, capture_output=True)
                archives.append(next(output.glob("*.zip")))
            self.assertEqual(archives[0].read_bytes(), archives[1].read_bytes())
            with zipfile.ZipFile(archives[0]) as archive:
                names = {name.split("/", 1)[1] for name in archive.namelist()}
                expected = set((ROOT / "FILES.txt").read_text().splitlines()) | {"SHA256SUMS"}
                self.assertEqual(names, expected)
                for line in archive.read("mixture-lco-0.1.0/SHA256SUMS").decode().splitlines():
                    checksum, name = line.split("  ", 1)
                    self.assertEqual(sha256(archive.read("mixture-lco-0.1.0/" + name)).hexdigest(), checksum)


if __name__ == "__main__":
    unittest.main()
