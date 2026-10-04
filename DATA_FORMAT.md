# Data format

Provide these three files in one explicitly named directory. UTF-8 without BOM, comma
delimiter, standard CSV quoting, exactly the headers below **in this order**. No extra
columns, missing columns or blank records are accepted. LF and CRLF record separators
are readable. IDs and SMILES are nonempty, trimmed, printable strings. IDs are arbitrary
stable identifiers; their spelling has no chemical or filesystem meaning. Decimal numeric
values may use exponent notation. No unit or ratio basis is inferred.

## compounds.csv

```csv
compound_id,canonical_smiles
mol-01,O
mol-02,C
```

Every compound ID is unique and represents one canonical molecule. Exactly duplicated
SMILES strings under multiple IDs are rejected. This package deliberately has no chemistry
toolkit: it neither validates chemical syntax nor asserts that distinct SMILES strings are
distinct molecules. **Before providing inputs**, validate structures with a chemistry
toolkit, choose and record canonicalization/isomer/salt/charge/tautomer policies, merge
same-molecule aliases and sum their masses within each formulation. Resolve remaining
chemical identity ambiguities before running CV. Merely comparing different SMILES strings
is not chemical validation. Canonicalization and registry generation must not use labels.

Unused registry molecules are allowed and retained in the one-hot vocabulary and supplied
embedding map. No `role` column is implemented. The package does not apply role-specific
eligibility, mixing rules or safety constraints.

## formulation_components.csv

```csv
formulation_id,compound_id,mass_fraction
toy-f-01,mol-01,0.08
toy-f-01,mol-02,0.92
```

One row per unique `(formulation_id, compound_id)`, even when source records mention the
same molecule in multiple roles. Every compound ID must resolve to the registry. Fractions
are **total-mixture mass fractions**, strictly positive, finite, at most one, and at least
the smallest positive normal float32 value. Sum per formulation must differ from 1 by no
more than **1e-6 absolute**, checked with exact decimal/rational arithmetic. A value of 8
means eight, not 8%; it is rejected. Percent, mole fraction, volume fraction, role-local
ratios and unspecified ratio bases must be explicitly converted by the input owner.

Accepted fractions are used as supplied, without renormalizing, rounding to a grid, or
guessing a denominator. The downstream float64 sum check allows a machine-roundoff
margin after this exact CSV check; it does not enlarge the CSV acceptance tolerance. For duplicate detection only, exact rational fractions are divided
by their exact total; decimal spellings and proportional near-unit copies therefore have
the same identity. Duplicate identical compositions under different formulation IDs are
rejected, as are distinct decimals that collapse to identical float32 model inputs. Merge
their measurements under one formulation ID before retrying. Other almost-equal recipes
remain distinct; choose a measurement-resolution policy upstream if these should be grouped.
No silent quantization is used. Reporting sums exact decimal input fractions before final
numeric serialization, preserving the 0.10/0.20 mass-band boundaries.

Ingredient-set identity is the full SHA-256 of the sorted component ID list serialized with
the package's deterministic JSON dialect (indented, ASCII-escaped, trailing newline). It
ignores component order and mass ratios. Grouped formulation CV keeps all records of a
formulation together. LCO cross-fitting and HPO keep all formulations sharing an ingredient
set together. Every declared formulation must have a measurement.

## measurements.csv

```csv
sample_id,formulation_id,DS,DE,CS,CE
toy-s-1,toy-f-01,12,17,10,5
toy-s-2,toy-f-01,12.2,17.2,10.2,5.2
```

Each row is one complete measured record; each sample ID is unique and resolves to an
existing formulation. All four targets are finite numeric values in **degrees Celsius**,
representable as float32. No missing-value imputation, unit conversion, row deduplication
or target correction occurs. The validator reports `DE < DS` or `CS < CE` violations and
keeps the values unchanged. No additional cross-branch ordering assumption is imposed.

Repeated measurements of one formulation are separate rows, whether they originate in
replicate readings or different lots. The input owner must avoid accidental duplicate
exports. **Lot metadata is not supported** in this exact schema, so the package cannot
separate technical replicates, model batch effects, or estimate manufacturing variance.
Keeping all formulation records together prevents their crossing folds; it does not turn
correlated records into independent observations. Training and checkpoint validation use
equal record weights, matching the original training implementation. Descriptive output
metrics first average errors within a formulation and then weight formulations equally.

## Embedding JSON

```json
{"mol-01": [0.1, -0.2, 0.3], "mol-02": [0.4, 0.5, -0.6]}
```

The top-level object must contain **exactly** the registry IDs. JSON keys must be unique;
each value is a nonempty array of equal dimension containing finite float32-representable
numbers. Booleans, NaN, infinities, ragged vectors and missing or extra molecules are
rejected. Vectors are converted to float32; no scaling, centering or normalization is fitted
by this package. Pickle and executable model files are never loaded by this interface.
The raw file checksum, vector dimension and vocabulary are recorded in the run manifest.

For one-hot inputs the sorted registry defines the vocabulary and remains fixed across
all folds and arms. The X column persists when X is excluded from training. It receives no
label gradient through X-free inputs, although AdamW weight decay applies to all parameters.

## Size requirements and outputs

Every training partition needs at least two measurement rows. Each requested grouped split
needs at least as many distinct groups as folds; X-free validation and training must both
be nonempty. The package fails rather than silently lowering fold counts. Choose a clearly
identified configuration appropriate to your data. This minimum is a software requirement,
not the manuscript's ingredient-eligibility threshold or evidence of statistical adequacy.

`predictions.csv` has exact columns determined by `report.Prediction`: run and representation;
sample, formulation, ingredient-set and held-out ingredient identities; arm, seed, fold and
target; `y_true`, `y_pred`, `held_out_mass`, `unseen_mass`, `id_all_ingredients_seen`.
Folds are zero-based; CV has an empty held-out ingredient. Boolean CSV fields are `True` or
`False`. In paired LCO rows, the eligibility flag refers to that fold's **ID** vocabulary in
both arms, while `unseen_mass` refers to the actual vocabulary of the row's own arm.
