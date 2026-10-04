# MixNet: mixture-lco

Portable MixNet training and **descriptive matched leave-compound-out (LCO)** evaluation
for four mixture properties, named DS, DE, CS, and CE.

This repository hosts the **mixture-lco 0.1.0** package: the MixNet model and portable
evaluation workflows. Repository: [insilicomd/MixNet](https://github.com/insilicomd/MixNet).
The Python package and command-line executable are named `mixture-lco`.

Supply your own compound registry, mass-fraction formulations, and measurements. The
examples are invented software fixtures using arbitrary simple molecules. They have
**no scientific or thermochromic validity** and do not reproduce confidential paper results.
No manufacturing records, learned weights, proprietary embeddings, or external encoder
implementation are included.

## Install and run

Python 3.11 or newer, NumPy 2.x, and PyTorch 2.x (at least 2.4) are required. In an environment
with those dependencies already installed, install this directory without network access:

```bash
python -m pip install --no-deps --no-build-isolation .
mixture-lco validate --data examples/tiny
mixture-lco cv --data examples/tiny --preset smoke --output /tmp/toy-cv-new
mixture-lco lco --data examples/tiny --preset smoke --ingredient mol-01 --output /tmp/toy-lco-new
mixture-lco lco --data examples/tiny --preset smoke --ingredient mol-01 \
  --embeddings examples/tiny/embeddings.json --output /tmp/toy-vectors-new
```

`python -m mixture_lco` is equivalent to `mixture-lco`. The executable runs on **CPU** and
does not download anything. Install dependencies separately if your environment lacks them.
The export includes no dependency wheels. In the source directory, commands can also be
run with `PYTHONPATH=src python -m mixture_lco ...` before installation.

Every output path must be **new**, including the synthetic generator's destination:

```bash
mixture-lco synthetic --output /tmp/new-invented-mixtures
```

The checked-in example has 8 registered molecules, 28 formulations, 14 ingredient sets,
and 57 invented measurement records. It deliberately includes repeated measurements,
an unused registry molecule, and an evaluation-only co-ingredient. `mol-01` has six
evaluation ingredient sets and eight X-free sets. The toy is designed to exercise
grouping and ineligible pooled memberships, not to yield meaningful errors.

## Inputs and embeddings

The exact CSV headers, validation rules, units, identity policy, and embedding JSON schema
are in [DATA_FORMAT.md](DATA_FORMAT.md). Without `--embeddings`, the representation is
one-hot over the **entire explicit sorted registry**. The withheld molecule retains its
column; it is not zeroed, replaced by UNK, or removed.

To use a real frozen encoder, canonicalize and merge your molecules first, generate vectors
in your own environment with your chosen encoder, and export one finite numeric vector per
registry ID. Keep the encoder name, immutable code/model revision, weight checksum,
tokenizer/preprocessing settings, canonicalization version, vector dimension and dtype,
and failures with your embedding-generation records. Hash and freeze that mapping before
comparing representations. All compound IDs must be present, including unused registry
entries. Do not train or fine-tune representations using evaluation labels. This package
checks the vector file's schema and checksum; it cannot verify its upstream lineage or
that it was generated without leakage. The example vectors are arbitrary arithmetic
fixtures, **not a pretrained chemical encoder**.

## Paper settings and smoke settings

The default is `--preset paper`: 40 random trials, 60 maximum epochs, patience 8, seeds
17/42/73, five formulation CV folds, five X-free ingredient-set HPO folds, three evaluation
cross-fit folds, and at least 15% of X-free records in the shared early-stopping validation
set. The architecture and hyperparameter ranges follow the manuscript method. These
defaults can be expensive on CPU. No full benchmark execution is part of this package's
smoke validation.

`--preset smoke` explicitly reduces the architecture, epochs, seeds, folds, and trials.
It is for software checks only. Inspect or customize the full closed configuration:

```bash
mixture-lco config --preset paper > my-config.json
mixture-lco lco --data my-data --config my-config.json --ingredient my-X \
  --output /tmp/my-new-run
```

`--ingredient` can be repeated to evaluate multiple explicitly chosen ingredients in one
run. The historical paper's ingredient selection and eligibility ranking are not automated.
Each ingredient/representation/seed receives the same configured number of candidate
trials and HPO folds. Candidate settings depend only on the training seed and search space,
so they match across ingredients and representations. Equal budgets refer to trial/fold
counts and maximum epochs, not identical optimizer steps, elapsed time, or model sizes.

Grouped CV deliberately follows the manuscript's **selection-aware** benchmark: the same
folds tune hyperparameters, early-stop, and report predictions. It is not nested CV and
does not provide an unbiased generalization estimate. LCO uses X-free ingredient-set folds
for tuning; both final arms share the winner, initialization seed and X-free validation
partition. Each arm fits its own train-only normalizer. ID additionally learns from other
evaluation cross-fit folds, so the arm comparison includes a training-size difference.
See [docs/METHODS.md](docs/METHODS.md) for the exact data flow and limitations.

## Outputs

Each completed run contains:

| File | Contents |
|---|---|
| `manifest.json` | Run status and identity; exact input, config, source, split, environment and output hashes |
| `validation.json` | Input counts and measured target-order violations |
| `splits.json` | Stable row-to-sample/formulation/set identities and every training, validation, HPO and evaluation assignment |
| `trials.jsonl`, `chosen.jsonl` | Every attempted trial, failures, fold scores, selected settings and HPO weighting |
| `fits.jsonl` | Every fit's memberships, initialization hash, normalizer, best epoch/history and checkpoint hash |
| `checkpoints/` | Final-arm weights plus normalizer in the original safe JSON-header/float32-buffer format; no pickle |
| `predictions.csv` | Sample, formulation, ingredient set, held-out ingredient, arm, seed, fold, target, truth and prediction; mass and eligibility fields |
| `summary.json` | Per-formulation errors, all per-ingredient memberships and paired eligible pooled errors, with denominators |

Errors average the individual seed models' absolute errors. They are not errors of an
ensemble-average prediction. Repeated measurement records first receive equal weight
within each formulation, so heavily measured formulations do not dominate the final mean.
Mass bands use actual LCO unseen mass, not an assumption that it always equals X's fraction.
Pooled summaries include only memberships whose **ID training vocabulary contains every
test ingredient in every seed**. The all-ingredient summaries retain the others. Across
ingredients, one formulation can count in multiple memberships. No significance winner,
p-value, confidence interval, or guarantee for a new ingredient is produced.

Failed runs retain their manifest and completed fit/trial records. Use a fresh path to retry;
there is no implicit resume or overwrite. Inputs and results remain local. Run directories
may contain your sensitive identifiers and outcomes and must be handled under your own
data policy. They are excluded from the distribution allowlist.

## Validation and distribution

```bash
ruff check .
mypy src
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/export.py --output /tmp/new-mixture-source-export
```

The exporter reads the explicit [FILES.txt](FILES.txt) allowlist, rejects symlinks, and
creates a reproducible source ZIP with `SHA256SUMS`, plus its archive checksum. Caches,
run results, models, unlisted files and the parent repository are not recursively copied.
Build/install from the exported tree when preparing a distributable. Source and wheel
metadata include only the portable package and two presets. See
[docs/DISTRIBUTION.md](docs/DISTRIBUTION.md) for isolated verification.

The MixNet source is byte-identical to the author-provided implementation. Training differs
by recorded imports, installation-message text, and a float64 rounding margin in the
post-validation mass-sum check; optional source-tree parity checks
are described in the distribution guide. [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json)
pins the original and extracted hashes.

## Scope and licensing

[LICENSE](LICENSE) grants MIT permissions for **this extracted package only**. It does not
change the parent project's license, grant rights to manufacturing data, or relicense
third-party dependencies or encoders. [THIRD_PARTY.md](THIRD_PARTY.md) records the boundary.
The MIT copyright notice identifies the package's contributing authors. No rights to
confidential data or pretrained encoders are granted, and no data-access promise is made.

Covered methods are portable input validation, MixNet, training, grouped CV, matched LCO,
configurable equal-budget HPO, and descriptive aggregation. Historical R01 inference,
proprietary ETL, exact frozen datasets/splits/results, pretrained encoder generation,
prospective manufacturing and inverse optimization are outside this package. It is not a
release of all analysis/statistics code used in the manuscript.
