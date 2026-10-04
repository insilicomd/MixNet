"""Optional author-side parity check against an explicitly supplied original source tree."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def probe(original: Path | None) -> dict[str, Any]:
    import numpy as np
    import torch

    from mixture_lco.config import load_settings
    from mixture_lco.data import TARGETS, load_dataset
    from mixture_lco.features import build_features

    root = Path(__file__).resolve().parents[1]
    data = load_dataset(root / "examples/tiny")
    features, _ = build_features(data)
    if original is not None:
        sys.path.insert(0, str(original / "src"))
        from deepchameleon.features.composition import ComponentFeatureSet, ComponentSample
        from deepchameleon.models.mixnet import MixNet, MixNetConfig
        from deepchameleon.training.train import ensure_torch_runtime, train_run
        features = ComponentFeatureSet(
            features.sample_ids, features.feature_family,
            tuple(ComponentSample(s.embeddings, s.mass_fractions, s.compound_ids)
                  for s in features.samples), features.max_components, features.dimension)
    else:
        from mixture_lco.mixnet import MixNet, MixNetConfig
        from mixture_lco.train import ensure_torch_runtime, train_run
    settings, _ = load_settings(preset="smoke")
    candidate = settings.candidates(17)[0]
    architecture = settings.model(features.dimension, candidate)
    # Reconstruct using the backend's own config type; no source implementation is patched.
    from dataclasses import asdict
    config = MixNetConfig(**asdict(architecture))
    ensure_torch_runtime()
    torch.manual_seed(123)
    model = MixNet(config).eval()
    with torch.no_grad():
        outputs = model(
            torch.from_numpy(np.stack([s.embeddings for s in features.samples[:3]])),
            torch.from_numpy(np.stack([s.mass_fractions for s in features.samples[:3]])),
            torch.ones((3, 2), dtype=torch.bool))
    targets = np.asarray([s.targets for s in data.samples])
    fitted = train_run(
        model_config=config, features=features, targets=targets, target_names=TARGETS,
        train_indices=list(range(40)), val_indices=list(range(40, len(data.samples))), seed=17,
        max_epochs=3, patience=2, batch_size=16, learning_rate=0.001, weight_decay=0.01)
    return {"fixed_input_outputs": [output.tolist() for output in outputs],
            "checkpoint_sha256": sha256(fitted.checkpoint_bytes).hexdigest(),
            "best_epoch": fitted.best_epoch, "history": list(fitted.val_macro_mae_history),
            "predictions": {t: fitted.y_pred[t].tolist() for t in TARGETS}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path)
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(probe(args.original_root), sort_keys=True, allow_nan=False))
        return
    if args.original_root is None:
        parser.error("--original-root is required for the optional author-side comparison")
    root = Path(__file__).resolve().parents[1]
    provenance = json.loads((root / "SOURCE_PROVENANCE.json").read_text())
    for entry in provenance["files"]:
        original = args.original_root / "src" / entry["source_module"]
        payload = original.read_bytes()
        if sha256(payload).hexdigest() != entry["original_sha256"]:
            raise ValueError(f"original source hash differs: {entry['source_module']}")
        adapted = payload.decode()
        for before, after in entry["adaptations"]:
            adapted = adapted.replace(before, after)
        if adapted.encode() != (root / entry["package_file"]).read_bytes():
            raise ValueError("extraction differs beyond the recorded substitutions")
    command = [sys.executable, str(Path(__file__).resolve()), "--probe"]
    portable = subprocess.check_output(command, text=True)
    original_output = subprocess.check_output(
        command + ["--original-root", str(args.original_root.resolve())], text=True)
    if json.loads(portable) != json.loads(original_output):
        raise ValueError("fixed-input model output or training checkpoint/predictions differ")
    print(json.dumps({"status": "passed", "model_source": "byte-identical",
                      "training_source": "only recorded substitutions",
                      "fixed_input_outputs": "identical", "training": "identical",
                      "checkpoint_sha256": json.loads(portable)["checkpoint_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
