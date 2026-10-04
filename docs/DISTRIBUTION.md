# Local distribution and verification

This guide creates local distribution artifacts for [insilicomd/MixNet](https://github.com/insilicomd/MixNet).
Publishing a new version is a separate maintainer action. The MIT notice names the package's
contributing authors. No access to confidential manufacturing data is promised.

## Source export

From the package root:

```bash
python scripts/export.py --output /tmp/new-source-export
```

`FILES.txt` is a sorted, explicit allowlist with no globs. Review additions to this list
alongside their source content. The exporter reads only those regular files, rejects
symlinks and traversal paths, fixes ZIP timestamps/modes and stores bytes without
compression. Equal source inputs produce identical ZIP bytes. The archive contains one
top-level `mixture-lco-0.1.0/` directory and `SHA256SUMS` for every included source file.
The external `ARCHIVE_SHA256SUMS` hashes the complete ZIP. Checksum manifests intentionally
do not hash themselves. The source ZIP is the reproducible distribution artifact;
platform-specific wheel metadata is not claimed to be byte-identical across build tools.

`MANIFEST.in` enumerates the same audited files for setuptools sdist creation. The wheel
contains only the explicitly named `mixture_lco` package, two config presets, its typing
marker, and standard package/license metadata. Build from a fresh export to ensure caches
and any unlisted working-tree files cannot enter either artifact.

Do not add user datasets, checkpoints, results, model weights or parent-repository content
to the inventory. Example CSV and embedding JSON files are the only included data fixtures
and were generated solely for software validation.

## Isolated import and smoke

Extract into a fresh temporary directory, verify all checksums, and use your existing
dependency environment to build/install into a separate target or virtual environment.
For an environment with pip, setuptools, wheel and the runtime dependencies already present:

```bash
unzip /tmp/new-source-export/mixture-lco-0.1.0-source.zip -d /tmp/new-source-unpacked
cd /tmp/new-source-unpacked/mixture-lco-0.1.0
python -m pip install --no-deps --no-build-isolation --target /tmp/new-install-target .
cd /tmp
PYTHONPATH=/tmp/new-install-target python -m mixture_lco validate \
  --data /tmp/new-source-unpacked/mixture-lco-0.1.0/examples/tiny
PYTHONPATH=/tmp/new-install-target python -m mixture_lco lco \
  --data /tmp/new-source-unpacked/mixture-lco-0.1.0/examples/tiny \
  --ingredient mol-01 --preset smoke --output /tmp/new-installed-lco-run
```

Ensure `PYTHONPATH` contains only the installed target for this check, no parent repository
or CAS location. Dependencies can be reused from an existing environment; the source
package itself requires neither an editable parent install nor repository-relative data.
Run both CV and LCO, and repeat LCO with `--embeddings .../examples/tiny/embeddings.json`.
Inspect source ZIP, sdist and wheel member lists before sharing them.

## Developer checks

```bash
ruff check .
mypy src
PYTHONPATH=src python -m unittest discover -s tests -v
```

Tests cover malformed input, duplicate composition identity, repeated records, grouped
splits, X-free tuning, held-out-label perturbations, initialization pairing, weighting,
fresh-output refusal, failure records, one-hot/supplied embeddings, deterministic CPU runs,
source hashes and reproducible allowlist export. Full training benchmarks are not run.

When the original author source is separately available, perform the optional comparison:

```bash
PYTHONPATH=src python scripts/check_parity.py --original-root /path/to/original-checkout
```

This explicitly opt-in script verifies original source hashes, the complete substitution
map, fixed-input model outputs, and identical training histories, restored checkpoint bytes
and predictions in separate processes. It reads the original source only; it uses this
package's synthetic inputs and needs no confidential data or CAS. Normal installation,
tests and CLI commands never search for or import the parent project. Cross-platform
numerical equivalence is limited to the pinned runtime/platform actually checked.
