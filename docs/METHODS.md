# Method mapping and limitations

## Source fidelity

`mixture_lco/mixnet.py` is a byte-for-byte extraction of the authors' MixNet implementation.
It projects molecule vectors, appends mass fractions, uses 0/1/2 residual dense message
layers, fraction-weighted attention, a shared residual trunk and four target heads.
Default feature width is 64 with three message/pooling heads, gate/message hidden width
256, trunk widths 1024/512, and head widths 256/128/64. MVE and extra features are off in
the portable CLI; their original low-level implementation remains in the copied source.

`mixture_lco/train.py` is the original module with the import substitutions,
installation-message changes and mass-sum rounding guard listed in `SOURCE_PROVENANCE.json`.
The CSV validator checks the sum exactly against 1e-6. The downstream check adds
`4 * component_count * float64_epsilon` solely for summation roundoff (for example, three
fractions of 0.333333); accepted fractions are not renormalized and exact sums outside
the CSV tolerance remain rejected. The feature dataclasses
implement its input interface; the timing observer is a portable typing protocol, without
the original service/GPU observer implementation. The original checkpoint codec and
fixed-epoch helper are retained, although the CLI uses early stopping. No research registry,
CAS loader or production integration code is required.

Training fits each target's mean and sample standard deviation (`ddof=1`, floor 1e-8) on
**that arm's training records only**. It minimizes mean absolute error on standardized
targets, averaged over the four targets, with AdamW's single parameter group, betas
0.9/0.999 and epsilon 1e-8. Weight decay applies to all parameters. Each epoch's permutation
uses NumPy PCG64 seeded from SHA-256 of `shuffle:{seed}:{epoch}`. CPU computation uses
one intra-op and one inter-op thread and deterministic PyTorch algorithms.

After each epoch, validation macro-MAE is calculated in Celsius with equal record and
target weights. Strict improvement selects the checkpoint; equal scores do not reset
patience. The best state is restored, evaluated again, and serialized as a canonical JSON
header followed by little-endian float32 buffers. Training and checkpoint parity are
checked against the original source with identical portable features. This validates the
training kernel, not equivalence of newly generated public splits to historical splits.

## Portable orchestration

The three CSV tables replace proprietary ETL and CAS. Formulations are not quantized to
the historical mass grid. Split groups are ordered by a seed/key SHA-256, then assigned
whole to the least-loaded fold by record count (fold index breaks ties). Original frozen
split artifacts and random-search streams are not reproduced. All assignments are saved.

The orchestrator subsets features **and labels** to the union of a fit's training and
validation rows before invoking the copied training function. Evaluation labels never
enter LCO tuning, normalization, early stopping or checkpoint selection. Inference receives
features only; truth is joined afterward for reporting. Subsetting can change padded tensor
shape relative to a full-cohort historical run; floating-point bit patterns are not promised
to match a historical run on another shape/platform. Within the same software/numeric
platform, the smoke tests check byte-deterministic run outputs.

### Formulation CV

1. For each seed, partition complete formulations into CV folds.
2. Train each random-search candidate on each fold's training rows, using that fold's
   validation rows to early-stop; average best record-weighted macro-MAE equally over folds.
3. Select the smallest score, breaking exact ties by trial index. Refit the winning settings
   on the same folds and retain restored best-checkpoint predictions.
4. Aggregate errors within formulations and across seeds.

The final public CV summary weights formulations equally. The manuscript's in-vocabulary
Table 2 instead pools out-of-fold measurement records and reports seed variation. These
are explicitly different aggregations; use the exported predictions to compute any
additional record-weighted table and label that weighting.

This intentionally exposes the manuscript's selection-aware benchmark. Reported validation
folds also selected settings and epochs. It is **not nested CV** and no unbiased error claim
is made. Ingredient sets can cross formulation-CV folds; identical formulations cannot.

### Matched LCO

1. The user explicitly selects X. Every formulation containing X belongs to evaluation.
   Remaining formulations are X-free. No role ranking, ingredient cap or paper eligibility
   filter is inferred from outcomes.
2. Order X-free ingredient sets with `split_seed`; allocate whole sets to early-stopping
   validation until it holds at least `ceil(validation_fraction * X-free records)` records.
   The remainder is the LCO training part. This partition is independent of training seed.
3. Split X-containing evaluation data into ingredient-set cross-fit folds, independently of
   labels and training seed. No evaluation ingredient set crosses folds.
4. For each training seed, tune on **all X-free rows**, including the eventual shared
   early-stopping validation part, using separate ingredient-set HPO folds. This reuse is
   internal to X-free data; no evaluation labels select settings or checkpoints.
5. Fit one LCO model on the X-free training part. For cross-fit fold j, fit ID on that same
   part plus all other evaluation folds, using the **same X-free validation partition**.
   Predict j with both restored best checkpoints. ID may learn from labels in other
   evaluation folds; it never trains or early-stops on j's labels or ingredient sets.

Both arms share architecture, hyperparameters, seed and initial weights, verified in the
receipts by initial-state SHA-256. Each arm estimates its own normalizer. Their data size,
normalizer, optimizer-step count and best epoch can differ. This is an observational
comparison of training conditions, not a causal isolation of exposure to X.

### Equal-budget HPO

The paper default is 40 trials per seed/ingredient/representation, 60 maximum epochs,
patience 8, five HPO folds. Choices are message layers {0,1,2}, log-uniform learning rate
[1e-4,1e-2], weight decay {1e-3,1e-2,1e-1}, batch size {32,64,128}. Each seed has the same
candidate stream across ingredients and vector dimensions. There is no network or model
download. The smoke preset is deliberately smaller and carries a nonscientific label.

Every candidate attempts every fold; numeric failures remain in trial and fit logs. A
failed candidate is ineligible and is not replaced by additional trials. All-failed searches
fail the run. Final-arm failures also fail the run. The fixed trial/fold/max-epoch budget
does not guarantee equal successful epochs or compute time. Trials are sampled with
replacement; repeated candidates still consume a trial. The HPO objective is equal-fold
mean of record-weighted validation macro-MAE, distinct from final formulation weighting.

## Descriptive aggregation

For each ingredient/formulation/arm/seed/target, average **absolute errors over records**.
Average the resulting errors over four targets and seed models. Then average formulations
equally within an ingredient. This order differs from averaging predictions before taking
absolute errors or allowing repeated measurements to multiply a formulation's final weight.
The per-target and formulation values remain in `summary.json`.

Each prediction records X's mass and the summed mass of all compounds absent from its own
arm's training vocabulary. LCO training is seed-independent, so LCO unseen mass defines
stable bands: <0.10, [0.10,0.20), >=0.20. Other co-ingredients can also be unseen. Validation
membership does not make a compound part of the training vocabulary.

All per-ingredient summaries retain all evaluation memberships. Pooled paired summaries
require every ingredient in the test formulation to occur in the corresponding ID training
set for **every seed**. Both ID and LCO errors use exactly that same subset. They weight
ingredient-formulation memberships equally, not ingredients equally. A formulation with
several selected held-out ingredients appears repeatedly; counts of memberships and distinct
formulations are both reported. Do not interpret these as independent observations.

## Coverage boundary

Included: reusable CSV format, one-hot and supplied frozen vectors, original MixNet/training,
grouped CV, matched LCO, equal-budget random search, explicit memberships and descriptive errors.

Excluded: proprietary ingredient identities/ETL/QC, paper data and exact split artifacts,
automatic pretrained encoding/fine-tuning, historical R01 bootstrap/sign-flip inference and
its size diagnostics, significance classifications/encoder selection, OOD distance tiers,
uncertainty calibration, prospective manufacturing, inverse design and optimizer evaluation,
the final full-cohort frozen policy, and all frozen paper numbers/figures. No claim is made
that all manuscript analysis or statistics code is released. The toy validates software
behavior only; it cannot establish thermochromic accuracy or reproduce paper findings.
