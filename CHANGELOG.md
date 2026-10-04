# Changelog

## 0.1.0 - 2026-10-04

Initial public source release of MixNet and the `mixture-lco` Python package.

- Original MixNet architecture with portable training interfaces.
- Explicit mixture CSV and supplied-embedding schemas, validation and synthetic fixtures.
- Formulation-grouped CV and matched leave-compound-out evaluation with equal tuning budgets.
- Descriptive summaries, split assignments, provenance and safe checkpoints.
- Source allowlist and reproducible ZIP export.

Validation: 20 behavioral tests; Ruff and strict mypy; fixed-input original-source parity;
isolated installation and four CPU smoke runs. A downstream floating-point mass-sum
boundary discrepancy was fixed before release. Confidential manufacturing data,
learned property-model weights and internal development history are not included.
