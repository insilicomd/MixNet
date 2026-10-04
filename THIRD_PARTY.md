# Code origin and third-party boundary

The extracted MixNet and training modules are author-provided DeepChameleon code released
under MIT **only within this package**, with original and extracted SHA-256 hashes and
the complete import/message substitutions and floating-point validation adaptation in `SOURCE_PROVENANCE.json`. The parent project
retains its own license. Portable input, split, orchestration, reporting, fixtures and
packaging code are included under the package MIT license.

The mixture model follows the conceptual fraction-weighted attention approach associated
with Roost/MixRoost. No Roost/MixRoost implementation is vendored in this distribution.
The model implementation here is the authors' MixNet implementation; the origin metadata
does not claim ownership of third-party methods or references.

NumPy, PyTorch, setuptools, wheel and optional developer tools are external dependencies.
They are not bundled in the source export and retain their respective copyright and
license notices in their own distributions. Installing this package does not relicense
those dependencies.

No SMI-TED, MoLFormer, ChemBERTa, tokenizer, pretrained model weight, chemical toolkit or
third-party encoder source is included. Users supplying encoder vectors must separately
review the encoder's license, model terms, provenance and any redistribution conditions.
The MIT license here does not grant rights to those assets or to user-supplied datasets.

The tiny molecule strings describe arbitrary simple molecules. All fixture IDs, ratios,
target values and numeric embedding vectors were generated expressly for software tests;
none are copied from manufacturing records. The embedding fixture is not a chemical model.
