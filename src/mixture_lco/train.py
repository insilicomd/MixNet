"""Deterministic MixNet training and weights-only checkpoint serialization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from numbers import Integral, Real
import os
from threading import Lock
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
from numpy.typing import NDArray

from mixture_lco.features import ComponentFeatureSet

if TYPE_CHECKING:
    from mixture_lco.features import InferenceTimingObserver
    from mixture_lco.mixnet import MixNet, MixNetConfig


CHECKPOINT_FORMAT_VERSION: Final[str] = "1.0"
NORMALIZER_METHOD: Final[str] = "gaussian-per-target-train-fold"
MVE_Y_STD_RULE: Final[str] = "exp(log_std) * target_std"
CUBLAS_WORKSPACE_CONFIG: Final[str] = ":4096:8"
_TARGET_COUNT: Final[int] = 4
_MIN_TARGET_STD: Final[float] = 1.0e-8
_CUBLAS_WORKSPACE_CONFIG_ENV: Final[str] = "CUBLAS_WORKSPACE_CONFIG"
_RUNTIME_LOCK = Lock()
_TORCH_RUNTIME_INITIALIZED = False
_TORCH_CUDA_RUNTIME_INITIALIZED = False


class TrainingError(ValueError):
    """Raised when training inputs or numeric state violate the contract."""


class TrainingDependencyError(RuntimeError):
    """Raised when the optional PyTorch runtime is unavailable."""


class CheckpointError(TrainingError):
    """Raised when a MixNet checkpoint violates its byte-level contract."""


@dataclass(frozen=True)
class TrainResult:
    """Best-state metrics, checkpoint, and aligned physical-unit validation output."""

    target_names: tuple[str, ...]
    best_epoch: int
    best_val_macro_mae: float
    val_macro_mae_history: tuple[float, ...]
    val_per_target_mae_history: tuple[Mapping[str, float], ...]
    checkpoint_bytes: bytes
    val_row_indices: NDArray[np.int64]
    y_pred: Mapping[str, NDArray[np.float64]]
    y_std: Mapping[str, NDArray[np.float64]] | None


@dataclass(frozen=True)
class FixedEpochTrainResult:
    """Final-epoch checkpoint of a validation-free fixed-epoch training run."""

    target_names: tuple[str, ...]
    trained_epochs: int
    checkpoint_bytes: bytes


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Strictly reconstructed MixNet plus the preprocessing state it requires."""

    model: MixNet
    model_config: MixNetConfig
    best_epoch: int
    target_names: tuple[str, ...]
    target_means: tuple[float, ...]
    target_stds: tuple[float, ...]


@dataclass(frozen=True)
class InferenceResult:
    """Physical-unit predictions for a fixed, sorted set of feature rows."""

    target_names: tuple[str, ...]
    row_indices: NDArray[np.int64]
    y_pred: Mapping[str, NDArray[np.float64]]
    y_std: Mapping[str, NDArray[np.float64]] | None


@dataclass
class _EarlyStopping:
    """Strict single-scalar patience state used directly by ``train_run``."""

    patience: int
    best_metric: float = math.inf
    best_epoch: int = -1
    epochs_without_improvement: int = 0

    def observe(self, *, epoch: int, metric: float) -> tuple[bool, bool]:
        """Return ``(improved, should_stop)`` for one val macro-MAE observation."""

        if not math.isfinite(metric):
            raise TrainingError("validation macro-MAE is non-finite")
        if metric < self.best_metric:
            self.best_metric = metric
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            return True, False
        self.epochs_without_improvement += 1
        return False, self.epochs_without_improvement >= self.patience


@dataclass(frozen=True)
class _PreparedData:
    embeddings: Any
    mass_fractions: Any
    mask: Any
    targets: Any
    target_values: NDArray[np.float64]
    extra_features: Any | None


@dataclass(frozen=True)
class _ComponentTensors:
    embeddings: Any
    mass_fractions: Any
    mask: Any
    extra_features: Any | None
    row_count: int


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise TrainingDependencyError(
            "PyTorch is required for MixNet training; "
            "install mixture-lco with its declared dependencies"
        ) from exc
    return torch


def _import_mixnet() -> tuple[type[MixNet], type[MixNetConfig]]:
    _import_torch()
    try:
        from mixture_lco.mixnet import MixNet, MixNetConfig
    except ImportError as exc:
        raise TrainingDependencyError(
            "PyTorch is required for MixNet training; "
            "install mixture-lco with its declared dependencies"
        ) from exc
    return MixNet, MixNetConfig


def _assert_torch_runtime(torch: Any, device: Any | None = None) -> None:
    if not _TORCH_RUNTIME_INITIALIZED:
        raise TrainingError(
            "PyTorch deterministic runtime is not initialized; "
            "call ensure_torch_runtime() before train_run()"
        )
    if not bool(torch.are_deterministic_algorithms_enabled()):
        raise TrainingError("PyTorch deterministic algorithms were disabled after initialization")
    if int(torch.get_num_threads()) != 1:
        raise TrainingError("PyTorch intra-op thread count must remain 1")
    if int(torch.get_num_interop_threads()) != 1:
        raise TrainingError("PyTorch inter-op thread count must remain 1")
    if device is None or torch.device(device).type != "cuda":
        return
    if not _TORCH_CUDA_RUNTIME_INITIALIZED:
        raise TrainingError(
            "PyTorch CUDA deterministic runtime is not initialized; "
            "call ensure_torch_runtime() with the CUDA device before use"
        )
    workspace = os.environ.get(_CUBLAS_WORKSPACE_CONFIG_ENV)
    if workspace != CUBLAS_WORKSPACE_CONFIG:
        raise TrainingError(
            f"{_CUBLAS_WORKSPACE_CONFIG_ENV} must equal "
            f"{CUBLAS_WORKSPACE_CONFIG!r} for deterministic CUDA execution"
        )
    if not bool(torch.backends.cudnn.deterministic):
        raise TrainingError("torch.backends.cudnn.deterministic must remain enabled")
    if bool(torch.backends.cudnn.benchmark):
        raise TrainingError("torch.backends.cudnn.benchmark must remain disabled")
    if bool(torch.backends.cuda.matmul.allow_tf32):
        raise TrainingError("torch.backends.cuda.matmul.allow_tf32 must remain disabled")
    if bool(torch.backends.cudnn.allow_tf32):
        raise TrainingError("torch.backends.cudnn.allow_tf32 must remain disabled")


def ensure_torch_runtime(device: str = "cpu") -> None:
    """Initialize deterministic PyTorch settings for the requested device.

    The original CPU process-global settings remain the default. CUDA adds a second
    process-global contract and refuses to initialize unless the exact cuBLAS workspace
    configuration was exported before the first CUDA operation.
    """

    global _TORCH_CUDA_RUNTIME_INITIALIZED, _TORCH_RUNTIME_INITIALIZED
    torch = _import_torch()
    if not isinstance(device, str) or not device.strip() or device != device.strip():
        raise TrainingError("device must be a non-blank trimmed string")
    try:
        resolved_device = torch.device(device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise TrainingError(f"invalid torch device {device!r}: {exc}") from exc
    if (
        resolved_device.type == "cuda"
        and os.environ.get(_CUBLAS_WORKSPACE_CONFIG_ENV) != CUBLAS_WORKSPACE_CONFIG
    ):
        raise TrainingError(
            f"{_CUBLAS_WORKSPACE_CONFIG_ENV} must equal "
            f"{CUBLAS_WORKSPACE_CONFIG!r} for deterministic CUDA execution"
        )
    with _RUNTIME_LOCK:
        if not _TORCH_RUNTIME_INITIALIZED:
            try:
                torch.use_deterministic_algorithms(True)
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except RuntimeError as exc:
                raise TrainingError(
                    "PyTorch runtime initialization must occur before parallel torch work"
                ) from exc
            _TORCH_RUNTIME_INITIALIZED = True
        if resolved_device.type == "cuda" and not _TORCH_CUDA_RUNTIME_INITIALIZED:
            if not bool(torch.cuda.is_available()):
                raise TrainingError("CUDA device requested but CUDA is unavailable")
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            except (AttributeError, RuntimeError) as exc:
                raise TrainingError(
                    "PyTorch CUDA deterministic runtime initialization failed"
                ) from exc
            _TORCH_CUDA_RUNTIME_INITIALIZED = True
        _assert_torch_runtime(torch, resolved_device)


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"checkpoint header is not canonical JSON: {exc}") from exc


def _checked_target_names(target_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(target_names)
    if len(names) != _TARGET_COUNT:
        raise TrainingError(f"target_names must contain exactly {_TARGET_COUNT} names")
    for index, name in enumerate(names):
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise TrainingError(f"target_names[{index}] must be a non-blank trimmed string")
    if len(set(names)) != len(names):
        raise TrainingError("target_names must be unique")
    return names


def _checked_float(value: object, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TrainingError(f"{label} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise TrainingError(f"{label} must be finite")
    if minimum is not None and converted < minimum:
        raise TrainingError(f"{label} must be at least {minimum}")
    return converted


def _checked_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise TrainingError(f"{label} must be a positive integer")
    return int(value)


def _checked_indices(
    values: Sequence[int],
    *,
    label: str,
    row_count: int,
) -> NDArray[np.int64]:
    checked: list[int] = []
    for position, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TrainingError(f"{label}[{position}] must be an integer row index")
        index = int(value)
        if index < 0 or index >= row_count:
            raise TrainingError(f"{label}[{position}] is outside the feature row range")
        checked.append(index)
    if not checked:
        raise TrainingError(f"{label} must not be empty")
    if len(set(checked)) != len(checked):
        raise TrainingError(f"{label} must contain unique row indices")
    return np.asarray(sorted(checked), dtype=np.int64)


def _checked_targets(
    targets: object,
    *,
    target_names: tuple[str, ...],
    row_count: int,
) -> NDArray[np.float64]:
    raw: object = targets
    if isinstance(targets, Mapping):
        if set(targets) != set(target_names):
            raise TrainingError("target mapping keys must exactly match target_names")
    try:
        if isinstance(targets, Mapping):
            raw = np.column_stack([targets[name] for name in target_names])
        matrix = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrainingError("targets must be a finite numeric matrix") from exc
    if matrix.shape != (row_count, len(target_names)):
        raise TrainingError(
            f"targets must have shape ({row_count}, {len(target_names)})"
        )
    if not bool(np.isfinite(matrix).all()):
        raise TrainingError("targets must contain only finite values")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _checked_model_config_payload(payload: object) -> MixNetConfig:
    _MixNet, MixNetConfig = _import_mixnet()
    if not isinstance(payload, dict):
        raise CheckpointError("checkpoint model_config must be an object")
    expected = {
        "input_dimension",
        "n_message_layers",
        "feature_len",
        "attention_heads",
        "gate_hidden",
        "message_hidden",
        "trunk_hidden",
        "head_hidden",
        "mve",
        "extra_features_len",
    }
    if set(payload) != expected:
        raise CheckpointError("checkpoint model_config fields are not exact")
    attention = payload["attention_heads"]
    if not isinstance(attention, dict) or set(attention) != {"message", "pooling"}:
        raise CheckpointError("checkpoint model_config attention_heads fields are not exact")

    def hidden(name: str) -> tuple[int, ...]:
        value = payload[name]
        if not isinstance(value, list) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in value
        ):
            raise CheckpointError(
                f"checkpoint model_config {name} must be an integer array"
            )
        return tuple(cast(list[int], value))

    try:
        return MixNetConfig(
            input_dimension=payload["input_dimension"],
            n_message_layers=payload["n_message_layers"],
            feature_len=payload["feature_len"],
            message_heads=attention["message"],
            pooling_heads=attention["pooling"],
            gate_hidden=hidden("gate_hidden"),
            message_hidden=hidden("message_hidden"),
            trunk_hidden=hidden("trunk_hidden"),
            head_hidden=hidden("head_hidden"),
            mve=payload["mve"],
            extra_features_len=payload["extra_features_len"],
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"invalid checkpoint model_config: {exc}") from exc


def _checked_normalizer_values(
    *,
    target_names: Sequence[str],
    target_means: Sequence[float],
    target_stds: Sequence[float],
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    names = _checked_target_names(target_names)
    means = tuple(target_means)
    stds = tuple(target_stds)
    if len(means) != len(names) or len(stds) != len(names):
        raise TrainingError("normalizer means and stds must align with target_names")
    checked_means = tuple(
        _checked_float(value, f"target_means[{index}]")
        for index, value in enumerate(means)
    )
    checked_stds = tuple(
        _checked_float(value, f"target_stds[{index}]", minimum=_MIN_TARGET_STD)
        for index, value in enumerate(stds)
    )
    return names, checked_means, checked_stds


def _readonly_vector(values: object, *, dtype: np.dtype[Any]) -> NDArray[Any]:
    vector = np.ascontiguousarray(values, dtype=dtype)
    vector.flags.writeable = False
    return vector


def _prepare_component_tensors(
    torch: Any,
    *,
    model_config: MixNetConfig,
    features: ComponentFeatureSet,
    extra_features: object | None,
    device: Any,
) -> _ComponentTensors:
    if not isinstance(features, ComponentFeatureSet):
        raise TrainingError("features must be a ComponentFeatureSet")
    row_count = len(features.sample_ids)
    if row_count == 0 or len(features.samples) != row_count:
        raise TrainingError("component features must contain aligned non-empty samples")
    if features.dimension != model_config.input_dimension:
        raise TrainingError(
            f"feature dimension is {features.dimension}; "
            f"model expects {model_config.input_dimension}"
        )
    actual_max = max(len(sample.compound_ids) for sample in features.samples)
    if features.max_components != actual_max or actual_max < 1:
        raise TrainingError("component feature max_components is inconsistent")

    embeddings = np.zeros(
        (row_count, actual_max, features.dimension),
        dtype=np.float32,
    )
    fractions = np.zeros((row_count, actual_max), dtype=np.float32)
    mask = np.zeros((row_count, actual_max), dtype=np.bool_)
    for row_index, sample in enumerate(features.samples):
        count = len(sample.compound_ids)
        sample_embeddings = np.asarray(sample.embeddings)
        sample_fractions = np.asarray(sample.mass_fractions)
        if count < 1 or sample_embeddings.shape != (count, features.dimension):
            raise TrainingError(f"component feature row {row_index} embedding shape is invalid")
        if sample_fractions.shape != (count,):
            raise TrainingError(f"component feature row {row_index} fraction shape is invalid")
        if not bool(np.isfinite(sample_embeddings).all()) or not bool(
            np.isfinite(sample_fractions).all()
        ):
            raise TrainingError(f"component feature row {row_index} contains non-finite values")
        if bool((sample_fractions < 0.0).any()) or not math.isclose(
            float(np.sum(sample_fractions, dtype=np.float64)),
            1.0,
            rel_tol=0.0,
            # CSV sums were checked exactly; allow only float64 summation roundoff here.
            abs_tol=1.0e-6 + 4.0 * count * np.finfo(np.float64).eps,
        ):
            raise TrainingError(
                f"component feature row {row_index} mass fractions must be non-negative and sum to 1"
            )
        converted_embeddings = np.asarray(sample_embeddings, dtype=np.float32)
        converted_fractions = np.asarray(sample_fractions, dtype=np.float32)
        if not bool(np.isfinite(converted_embeddings).all()) or not bool(
            np.isfinite(converted_fractions).all()
        ):
            raise TrainingError(
                f"component feature row {row_index} values overflow float32"
            )
        embeddings[row_index, :count] = converted_embeddings
        fractions[row_index, :count] = converted_fractions
        mask[row_index, :count] = True

    expected_extra = model_config.extra_features_len
    extra_array: NDArray[np.float32] | None = None
    if expected_extra == 0:
        if extra_features is not None:
            raise TrainingError("extra_features were provided but model expects none")
    else:
        if extra_features is None:
            raise TrainingError("extra_features are required by model_config")
        try:
            raw_extra = np.asarray(extra_features, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TrainingError("extra_features must be a finite numeric matrix") from exc
        if raw_extra.shape != (row_count, expected_extra):
            raise TrainingError(
                f"extra_features must have shape ({row_count}, {expected_extra})"
            )
        if not bool(np.isfinite(raw_extra).all()):
            raise TrainingError("extra_features must contain only finite float32 values")
        extra_array = np.ascontiguousarray(raw_extra, dtype=np.float32)

    try:
        return _ComponentTensors(
            embeddings=torch.as_tensor(embeddings, dtype=torch.float32, device=device),
            mass_fractions=torch.as_tensor(fractions, dtype=torch.float32, device=device),
            mask=torch.as_tensor(mask, dtype=torch.bool, device=device),
            extra_features=(
                None
                if extra_array is None
                else torch.as_tensor(extra_array, dtype=torch.float32, device=device)
            ),
            row_count=row_count,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise TrainingError(f"failed to move component inputs to device {device}: {exc}") from exc


def _prepare_features(
    torch: Any,
    *,
    model_config: MixNetConfig,
    features: ComponentFeatureSet,
    targets: object,
    target_names: tuple[str, ...],
    extra_features: object | None,
    device: Any,
) -> _PreparedData:
    tensors = _prepare_component_tensors(
        torch,
        model_config=model_config,
        features=features,
        extra_features=extra_features,
        device=device,
    )
    target_values = _checked_targets(
        targets,
        target_names=target_names,
        row_count=tensors.row_count,
    )
    try:
        targets_tensor = torch.as_tensor(target_values, dtype=torch.float32, device=device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise TrainingError(f"failed to move training inputs to device {device}: {exc}") from exc
    return _PreparedData(
        embeddings=tensors.embeddings,
        mass_fractions=tensors.mass_fractions,
        mask=tensors.mask,
        targets=targets_tensor,
        target_values=target_values,
        extra_features=tensors.extra_features,
    )


def serialize_checkpoint(
    model: MixNet,
    *,
    model_config: MixNetConfig,
    best_epoch: int,
    target_names: Sequence[str],
    target_means: Sequence[float],
    target_stds: Sequence[float],
) -> bytes:
    """Serialize restored MixNet weights and normalizer as deterministic bytes."""

    MixNet, MixNetConfig = _import_mixnet()
    if not isinstance(model, MixNet):
        raise CheckpointError("model must be a MixNet")
    if not isinstance(model_config, MixNetConfig):
        raise CheckpointError("model_config must be a MixNetConfig")
    if model.config != model_config:
        raise CheckpointError("model.config does not match model_config")
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, Integral) or best_epoch < 0:
        raise CheckpointError("best_epoch must be a non-negative integer")
    try:
        names, means, stds = _checked_normalizer_values(
            target_names=target_names,
            target_means=target_means,
            target_stds=target_stds,
        )
    except TrainingError as exc:
        raise CheckpointError(str(exc)) from exc

    parameters: list[dict[str, object]] = []
    buffers: list[bytes] = []
    offset = 0
    state = model.state_dict()
    torch = _import_torch()
    cuda_devices = {
        str(tensor.device)
        for tensor in state.values()
        if tensor.device.type == "cuda"
    }
    for cuda_device in sorted(cuda_devices):
        ensure_torch_runtime(cuda_device)
    for name in sorted(state):
        tensor = state[name]
        if tensor.dtype is not torch.float32:
            raise CheckpointError(f"checkpoint tensor {name!r} must have float32 dtype")
        array = np.array(
            tensor.detach().cpu().numpy(),
            dtype="<f4",
            copy=True,
            order="C",
        )
        if not bool(np.isfinite(array).all()):
            raise CheckpointError(f"checkpoint tensor {name!r} contains non-finite values")
        payload = array.tobytes(order="C")
        parameters.append(
            {
                "name": name,
                "shape": list(array.shape),
                "dtype": "float32",
                "offset": offset,
                "nbytes": len(payload),
            }
        )
        buffers.append(payload)
        offset += len(payload)

    header = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_config": model_config.to_dict(),
        "best_epoch": int(best_epoch),
        "parameters": parameters,
        "normalizer": {
            "method": NORMALIZER_METHOD,
            "target_names": list(names),
            "means": [repr(value) for value in means],
            "stds": [repr(value) for value in stds],
            "mve_y_std": MVE_Y_STD_RULE,
        },
    }
    return _canonical_json_bytes(header) + b"\n" + b"".join(buffers)


def _exact_dict(value: object, fields: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise CheckpointError(f"{label} fields are not exact")
    return cast(dict[str, object], value)


def _parse_repr_float(value: object, label: str, *, positive: bool = False) -> float:
    if not isinstance(value, str):
        raise CheckpointError(f"{label} must be a repr(float) string")
    try:
        converted = float(value)
    except ValueError as exc:
        raise CheckpointError(f"{label} must be a repr(float) string") from exc
    if not math.isfinite(converted) or repr(converted) != value:
        raise CheckpointError(f"{label} must be a canonical finite repr(float) string")
    if positive and converted < _MIN_TARGET_STD:
        raise CheckpointError(f"{label} must be at least {_MIN_TARGET_STD}")
    return converted


def load_checkpoint(payload: bytes, *, device: str = "cpu") -> LoadedCheckpoint:
    """Strictly load a deterministic MixNet checkpoint without pickle."""

    torch = _import_torch()
    MixNet, _MixNetConfig = _import_mixnet()
    if not isinstance(payload, bytes):
        raise CheckpointError("checkpoint payload must be bytes")
    header_bytes, delimiter, buffer_bytes = payload.partition(b"\n")
    if delimiter != b"\n" or not header_bytes:
        raise CheckpointError("checkpoint must contain a JSON header followed by a newline")
    try:
        decoded = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"checkpoint header is not valid UTF-8 JSON: {exc}") from exc
    header = _exact_dict(
        decoded,
        {"format_version", "model_config", "best_epoch", "parameters", "normalizer"},
        "checkpoint header",
    )
    if _canonical_json_bytes(header) != header_bytes:
        raise CheckpointError("checkpoint JSON header is not canonical")
    if header["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(
            f"unsupported checkpoint format_version: {header['format_version']!r}"
        )
    model_config = _checked_model_config_payload(header["model_config"])
    best_epoch = header["best_epoch"]
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, int) or best_epoch < 0:
        raise CheckpointError("checkpoint best_epoch must be a non-negative integer")

    normalizer = _exact_dict(
        header["normalizer"],
        {"method", "target_names", "means", "stds", "mve_y_std"},
        "checkpoint normalizer",
    )
    if normalizer["method"] != NORMALIZER_METHOD:
        raise CheckpointError("checkpoint normalizer method is unsupported")
    if normalizer["mve_y_std"] != MVE_Y_STD_RULE:
        raise CheckpointError("checkpoint MVE y_std convention is unsupported")
    raw_names = normalizer["target_names"]
    raw_means = normalizer["means"]
    raw_stds = normalizer["stds"]
    if not isinstance(raw_names, list):
        raise CheckpointError("checkpoint normalizer target_names must be an array")
    if not isinstance(raw_means, list) or not isinstance(raw_stds, list):
        raise CheckpointError("checkpoint normalizer means and stds must be arrays")
    try:
        target_names = _checked_target_names(raw_names)
    except TrainingError as exc:
        raise CheckpointError(str(exc)) from exc
    if len(raw_means) != len(target_names) or len(raw_stds) != len(target_names):
        raise CheckpointError("checkpoint normalizer values do not align with target_names")
    target_means = tuple(
        _parse_repr_float(value, f"checkpoint normalizer means[{index}]")
        for index, value in enumerate(raw_means)
    )
    target_stds = tuple(
        _parse_repr_float(
            value,
            f"checkpoint normalizer stds[{index}]",
            positive=True,
        )
        for index, value in enumerate(raw_stds)
    )

    raw_parameters = header["parameters"]
    if not isinstance(raw_parameters, list) or not raw_parameters:
        raise CheckpointError("checkpoint parameters must be a non-empty array")
    table: list[tuple[str, tuple[int, ...], int, int]] = []
    expected_offset = 0
    for index, raw_entry in enumerate(raw_parameters):
        entry = _exact_dict(
            raw_entry,
            {"name", "shape", "dtype", "offset", "nbytes"},
            f"checkpoint parameters[{index}]",
        )
        name = entry["name"]
        shape = entry["shape"]
        offset = entry["offset"]
        nbytes = entry["nbytes"]
        if not isinstance(name, str) or not name:
            raise CheckpointError(f"checkpoint parameters[{index}].name must be non-blank")
        if entry["dtype"] != "float32":
            raise CheckpointError(f"checkpoint parameter {name!r} has unsupported dtype")
        if not isinstance(shape, list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape
        ):
            raise CheckpointError(f"checkpoint parameter {name!r} has invalid shape")
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset != expected_offset
        ):
            raise CheckpointError(f"checkpoint parameter {name!r} has non-contiguous offset")
        element_count = math.prod(shape)
        expected_nbytes = element_count * np.dtype("<f4").itemsize
        if (
            isinstance(nbytes, bool)
            or not isinstance(nbytes, int)
            or nbytes != expected_nbytes
        ):
            raise CheckpointError(f"checkpoint parameter {name!r} has invalid nbytes")
        table.append((name, tuple(shape), offset, nbytes))
        expected_offset += nbytes
    parameter_names = [entry[0] for entry in table]
    if parameter_names != sorted(parameter_names) or len(parameter_names) != len(
        set(parameter_names)
    ):
        raise CheckpointError("checkpoint parameter names must be unique and sorted")
    if expected_offset != len(buffer_bytes):
        raise CheckpointError("checkpoint parameter buffers have missing or trailing bytes")

    try:
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda":
            ensure_torch_runtime(str(resolved_device))
        model = MixNet(model_config).to(resolved_device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise CheckpointError(f"cannot construct checkpoint model on device {device!r}: {exc}") from exc
    expected_state = model.state_dict()
    if set(parameter_names) != set(expected_state):
        missing = sorted(set(expected_state) - set(parameter_names))
        unexpected = sorted(set(parameter_names) - set(expected_state))
        raise CheckpointError(
            f"checkpoint state keys differ from model; missing={missing}, unexpected={unexpected}"
        )
    state: dict[str, Any] = {}
    for name, shape, offset, nbytes in table:
        expected_tensor = expected_state[name]
        if tuple(expected_tensor.shape) != shape or expected_tensor.dtype is not torch.float32:
            raise CheckpointError(f"checkpoint parameter {name!r} shape differs from model")
        array = np.frombuffer(
            buffer_bytes,
            dtype="<f4",
            count=nbytes // 4,
            offset=offset,
        ).copy()
        array = np.asarray(array.reshape(shape), dtype=np.float32)
        if not bool(np.isfinite(array).all()):
            raise CheckpointError(f"checkpoint parameter {name!r} contains non-finite values")
        state[name] = torch.as_tensor(array, dtype=torch.float32, device=resolved_device)
    try:
        load_result = model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise CheckpointError(f"strict checkpoint state load failed: {exc}") from exc
    if load_result.missing_keys or load_result.unexpected_keys:
        raise CheckpointError(
            "strict checkpoint state load returned missing or unexpected keys"
        )
    model.eval()
    return LoadedCheckpoint(
        model=model,
        model_config=model_config,
        best_epoch=best_epoch,
        target_names=target_names,
        target_means=target_means,
        target_stds=target_stds,
    )


def _batch_rows(
    values: NDArray[np.int64],
    batch_size: int,
) -> Sequence[NDArray[np.int64]]:
    return tuple(
        values[start : start + batch_size]
        for start in range(0, len(values), batch_size)
    )


def _forward_batch(
    torch: Any,
    model: MixNet,
    data: _PreparedData,
    rows: NDArray[np.int64],
) -> tuple[Any, Any | None]:
    indices = torch.as_tensor(rows, dtype=torch.long, device=data.embeddings.device)
    extra = None if data.extra_features is None else data.extra_features[indices]
    outputs = model(
        data.embeddings[indices],
        data.mass_fractions[indices],
        data.mask[indices],
        extra,
    )
    expected_width = 2 if model.config.mve else 1
    if len(outputs) != _TARGET_COUNT or any(
        output.shape != (len(rows), expected_width) for output in outputs
    ):
        raise TrainingError("MixNet outputs do not match the four-target contract")
    means = torch.stack([output[:, 0] for output in outputs], dim=1)
    log_stds = (
        torch.stack([output[:, 1] for output in outputs], dim=1)
        if model.config.mve
        else None
    )
    return means, log_stds


def _evaluate(
    torch: Any,
    model: MixNet,
    data: _PreparedData,
    rows: NDArray[np.int64],
    *,
    batch_size: int,
    target_means: NDArray[np.float64],
    target_stds: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    model.eval()
    normalized_parts: list[NDArray[np.float64]] = []
    log_std_parts: list[NDArray[np.float64]] = []
    with torch.no_grad():
        for batch in _batch_rows(rows, batch_size):
            normalized, log_stds = _forward_batch(torch, model, data, batch)
            normalized_array = np.asarray(
                normalized.detach().cpu().numpy(),
                dtype=np.float64,
            )
            if not bool(np.isfinite(normalized_array).all()):
                raise TrainingError("validation predictions contain non-finite values")
            normalized_parts.append(normalized_array)
            if log_stds is not None:
                log_std_array = np.asarray(
                    log_stds.detach().cpu().numpy(),
                    dtype=np.float64,
                )
                if not bool(np.isfinite(log_std_array).all()):
                    raise TrainingError("validation log_std predictions are non-finite")
                log_std_parts.append(log_std_array)

    normalized_predictions = np.concatenate(normalized_parts, axis=0)
    physical_predictions = normalized_predictions * target_stds + target_means
    if not bool(np.isfinite(physical_predictions).all()):
        raise TrainingError("inverse-normalized validation predictions are non-finite")
    expected = data.target_values[rows]
    per_target_mae = np.mean(np.abs(physical_predictions - expected), axis=0)
    macro_mae = float(np.mean(per_target_mae, dtype=np.float64))
    if not math.isfinite(macro_mae) or not bool(np.isfinite(per_target_mae).all()):
        raise TrainingError("validation MAE is non-finite")

    physical_stds: NDArray[np.float64] | None = None
    if model.config.mve:
        if len(log_std_parts) != len(normalized_parts):
            raise TrainingError("MVE validation outputs are incomplete")
        with np.errstate(over="ignore", invalid="ignore"):
            physical_stds = np.exp(np.concatenate(log_std_parts, axis=0)) * target_stds
        if not bool(np.isfinite(physical_stds).all()) or bool((physical_stds <= 0).any()):
            raise TrainingError("MVE validation standard deviations are non-finite")
    return macro_mae, per_target_mae, physical_predictions, physical_stds


def _prediction_mapping(
    names: tuple[str, ...],
    matrix: NDArray[np.float64],
) -> Mapping[str, NDArray[np.float64]]:
    return MappingProxyType(
        {
            name: cast(
                NDArray[np.float64],
                _readonly_vector(matrix[:, index], dtype=np.dtype(np.float64)),
            )
            for index, name in enumerate(names)
        }
    )


def train_run(
    *,
    model_config: MixNetConfig,
    features: ComponentFeatureSet,
    targets: object,
    target_names: Sequence[str],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    seed: int,
    device: str = "cpu",
    optimizer_name: str = "adamw",
    learning_rate: float = 1.0e-3,
    weight_decay: float = 0.0,
    betas: Sequence[float] = (0.9, 0.999),
    eps: float = 1.0e-8,
    max_epochs: int = 100,
    batch_size: int = 32,
    patience: int = 10,
    grad_clip_norm: float | None = None,
    extra_features: object | None = None,
) -> TrainResult:
    """Train one deterministic ``(config, protocol, seed, fold)`` MixNet run."""

    torch = _import_torch()
    MixNet, MixNetConfig = _import_mixnet()
    _assert_torch_runtime(torch)
    if not isinstance(model_config, MixNetConfig):
        raise TrainingError("model_config must be a MixNetConfig")
    names = _checked_target_names(target_names)
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= int(seed) < 2**63:
        raise TrainingError("seed must be an integer between 0 and 2**63 - 1")
    checked_seed = int(seed)
    if not isinstance(device, str) or not device.strip() or device != device.strip():
        raise TrainingError("device must be a non-blank trimmed string")
    if optimizer_name != "adamw":
        raise TrainingError("optimizer_name must be 'adamw'")
    checked_learning_rate = _checked_float(learning_rate, "learning_rate", minimum=0.0)
    if checked_learning_rate == 0.0:
        raise TrainingError("learning_rate must be positive")
    checked_weight_decay = _checked_float(weight_decay, "weight_decay", minimum=0.0)
    checked_eps = _checked_float(eps, "eps", minimum=0.0)
    if checked_eps == 0.0:
        raise TrainingError("eps must be positive")
    checked_betas = tuple(betas)
    if len(checked_betas) != 2:
        raise TrainingError("betas must contain exactly two values")
    beta_values = tuple(
        _checked_float(value, f"betas[{index}]", minimum=0.0)
        for index, value in enumerate(checked_betas)
    )
    if any(value >= 1.0 for value in beta_values):
        raise TrainingError("betas values must be less than 1")
    checked_max_epochs = _checked_positive_int(max_epochs, "max_epochs")
    checked_batch_size = _checked_positive_int(batch_size, "batch_size")
    checked_patience = _checked_positive_int(patience, "patience")
    checked_clip = (
        None
        if grad_clip_norm is None
        else _checked_float(grad_clip_norm, "grad_clip_norm", minimum=0.0)
    )
    if checked_clip == 0.0:
        raise TrainingError("grad_clip_norm must be positive when provided")

    if not isinstance(features, ComponentFeatureSet):
        raise TrainingError("features must be a ComponentFeatureSet")
    row_count = len(features.sample_ids)
    train_rows = _checked_indices(
        train_indices,
        label="train_indices",
        row_count=row_count,
    )
    val_rows = _checked_indices(
        val_indices,
        label="val_indices",
        row_count=row_count,
    )
    if len(train_rows) < 2:
        raise TrainingError("train_indices must contain at least two rows")
    if set(train_rows.tolist()) & set(val_rows.tolist()):
        raise TrainingError("train_indices and val_indices must be disjoint")

    try:
        resolved_device = torch.device(device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise TrainingError(f"invalid torch device {device!r}: {exc}") from exc
    ensure_torch_runtime(str(resolved_device))
    torch.manual_seed(checked_seed)
    try:
        model = MixNet(model_config).to(resolved_device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise TrainingError(f"failed to construct MixNet on device {device!r}: {exc}") from exc
    data = _prepare_features(
        torch,
        model_config=model_config,
        features=features,
        targets=targets,
        target_names=names,
        extra_features=extra_features,
        device=resolved_device,
    )

    train_target_values = data.target_values[train_rows]
    target_means = np.mean(train_target_values, axis=0, dtype=np.float64)
    target_stds = np.std(train_target_values, axis=0, ddof=1, dtype=np.float64)
    target_stds = np.maximum(target_stds, _MIN_TARGET_STD)
    if not bool(np.isfinite(target_means).all()) or not bool(np.isfinite(target_stds).all()):
        raise TrainingError("train-fold target normalizer is non-finite")
    means_tensor = torch.as_tensor(target_means, dtype=torch.float32, device=resolved_device)
    stds_tensor = torch.as_tensor(target_stds, dtype=torch.float32, device=resolved_device)
    normalized_targets = (data.targets - means_tensor) / stds_tensor
    if not bool(torch.isfinite(normalized_targets).all().item()):
        raise TrainingError("normalized training targets are non-finite")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=checked_learning_rate,
        weight_decay=checked_weight_decay,
        betas=beta_values,
        eps=checked_eps,
    )
    early_stopping = _EarlyStopping(checked_patience)
    macro_history: list[float] = []
    target_history: list[Mapping[str, float]] = []
    best_state: dict[str, Any] | None = None

    for epoch in range(checked_max_epochs):
        model.train()
        shuffle_digest = sha256(f"shuffle:{checked_seed}:{epoch}".encode("utf-8")).digest()
        shuffle_seed = int.from_bytes(shuffle_digest, "big")
        shuffled_rows = np.asarray(
            np.random.Generator(np.random.PCG64(shuffle_seed)).permutation(train_rows),
            dtype=np.int64,
        )
        for batch in _batch_rows(shuffled_rows, checked_batch_size):
            optimizer.zero_grad(set_to_none=True)
            predicted_means, predicted_log_stds = _forward_batch(torch, model, data, batch)
            batch_index = torch.as_tensor(
                batch,
                dtype=torch.long,
                device=resolved_device,
            )
            expected = normalized_targets[batch_index]
            absolute_errors = torch.abs(predicted_means - expected)
            if model_config.mve:
                if predicted_log_stds is None:
                    raise TrainingError("MVE model did not return log_std values")
                per_value_loss = (
                    math.sqrt(2.0)
                    * absolute_errors
                    * torch.exp(-predicted_log_stds)
                    + predicted_log_stds
                )
            else:
                per_value_loss = absolute_errors
            per_target_loss = per_value_loss.mean(dim=0)
            loss = per_target_loss.mean()
            if not bool(torch.isfinite(loss).item()):
                raise TrainingError(f"training loss became non-finite at epoch {epoch}")
            loss.backward()
            if checked_clip is not None:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), checked_clip)
                if not bool(torch.isfinite(total_norm).item()):
                    raise TrainingError(f"gradient norm became non-finite at epoch {epoch}")
            optimizer.step()

        macro_mae, per_target_mae, _predictions, _prediction_stds = _evaluate(
            torch,
            model,
            data,
            val_rows,
            batch_size=checked_batch_size,
            target_means=target_means,
            target_stds=target_stds,
        )
        macro_history.append(macro_mae)
        target_history.append(
            MappingProxyType(
                {
                    name: float(per_target_mae[index])
                    for index, name in enumerate(names)
                }
            )
        )
        improved, should_stop = early_stopping.observe(epoch=epoch, metric=macro_mae)
        if improved:
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        if should_stop:
            break

    if best_state is None or early_stopping.best_epoch < 0:
        raise TrainingError("training did not produce a finite best validation state")
    try:
        load_result = model.load_state_dict(best_state, strict=True)
    except RuntimeError as exc:
        raise TrainingError(f"failed to restore best MixNet state: {exc}") from exc
    if load_result.missing_keys or load_result.unexpected_keys:
        raise TrainingError("best-state restore returned missing or unexpected keys")

    best_macro, _best_per_target, best_predictions, best_prediction_stds = _evaluate(
        torch,
        model,
        data,
        val_rows,
        batch_size=checked_batch_size,
        target_means=target_means,
        target_stds=target_stds,
    )
    if best_macro != early_stopping.best_metric:
        raise TrainingError("restored best-state validation metric changed")
    checkpoint = serialize_checkpoint(
        model,
        model_config=model_config,
        best_epoch=early_stopping.best_epoch,
        target_names=names,
        target_means=target_means,
        target_stds=target_stds,
    )
    readonly_val_rows = cast(
        NDArray[np.int64],
        _readonly_vector(val_rows, dtype=np.dtype(np.int64)),
    )
    y_std = (
        None
        if best_prediction_stds is None
        else _prediction_mapping(names, best_prediction_stds)
    )
    return TrainResult(
        target_names=names,
        best_epoch=early_stopping.best_epoch,
        best_val_macro_mae=best_macro,
        val_macro_mae_history=tuple(macro_history),
        val_per_target_mae_history=tuple(target_history),
        checkpoint_bytes=checkpoint,
        val_row_indices=readonly_val_rows,
        y_pred=_prediction_mapping(names, best_predictions),
        y_std=y_std,
    )


def train_fixed_epochs(
    *,
    model_config: MixNetConfig,
    features: ComponentFeatureSet,
    targets: object,
    target_names: Sequence[str],
    train_indices: Sequence[int],
    seed: int,
    device: str = "cpu",
    optimizer_name: str = "adamw",
    learning_rate: float = 1.0e-3,
    weight_decay: float = 0.0,
    betas: Sequence[float] = (0.9, 0.999),
    eps: float = 1.0e-8,
    batch_size: int = 32,
    grad_clip_norm: float | None = None,
    extra_features: object | None = None,
    fixed_epochs: int,
) -> FixedEpochTrainResult:
    """Train a validation-free MixNet run for exactly ``fixed_epochs`` epochs.

    This is the fixed-epoch full-refit mode: no validation subset, no early stopping,
    and the final-epoch state is serialized as-is. The epoch convention is pinned to
    ``train_run``'s 0-based ``best_epoch``: reproducing best epoch ``e`` takes
    ``fixed_epochs = e + 1`` (epochs ``0..e``), and the checkpoint header records
    ``best_epoch = fixed_epochs - 1`` with the existing schema unchanged.
    """

    torch = _import_torch()
    MixNet, MixNetConfig = _import_mixnet()
    _assert_torch_runtime(torch)
    if not isinstance(model_config, MixNetConfig):
        raise TrainingError("model_config must be a MixNetConfig")
    names = _checked_target_names(target_names)
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= int(seed) < 2**63:
        raise TrainingError("seed must be an integer between 0 and 2**63 - 1")
    checked_seed = int(seed)
    if not isinstance(device, str) or not device.strip() or device != device.strip():
        raise TrainingError("device must be a non-blank trimmed string")
    if optimizer_name != "adamw":
        raise TrainingError("optimizer_name must be 'adamw'")
    checked_learning_rate = _checked_float(learning_rate, "learning_rate", minimum=0.0)
    if checked_learning_rate == 0.0:
        raise TrainingError("learning_rate must be positive")
    checked_weight_decay = _checked_float(weight_decay, "weight_decay", minimum=0.0)
    checked_eps = _checked_float(eps, "eps", minimum=0.0)
    if checked_eps == 0.0:
        raise TrainingError("eps must be positive")
    checked_betas = tuple(betas)
    if len(checked_betas) != 2:
        raise TrainingError("betas must contain exactly two values")
    beta_values = tuple(
        _checked_float(value, f"betas[{index}]", minimum=0.0)
        for index, value in enumerate(checked_betas)
    )
    if any(value >= 1.0 for value in beta_values):
        raise TrainingError("betas values must be less than 1")
    checked_batch_size = _checked_positive_int(batch_size, "batch_size")
    checked_fixed_epochs = _checked_positive_int(fixed_epochs, "fixed_epochs")
    checked_clip = (
        None
        if grad_clip_norm is None
        else _checked_float(grad_clip_norm, "grad_clip_norm", minimum=0.0)
    )
    if checked_clip == 0.0:
        raise TrainingError("grad_clip_norm must be positive when provided")

    if not isinstance(features, ComponentFeatureSet):
        raise TrainingError("features must be a ComponentFeatureSet")
    row_count = len(features.sample_ids)
    train_rows = _checked_indices(
        train_indices,
        label="train_indices",
        row_count=row_count,
    )
    if len(train_rows) < 2:
        raise TrainingError("train_indices must contain at least two rows")

    try:
        resolved_device = torch.device(device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise TrainingError(f"invalid torch device {device!r}: {exc}") from exc
    ensure_torch_runtime(str(resolved_device))
    torch.manual_seed(checked_seed)
    try:
        model = MixNet(model_config).to(resolved_device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise TrainingError(f"failed to construct MixNet on device {device!r}: {exc}") from exc
    data = _prepare_features(
        torch,
        model_config=model_config,
        features=features,
        targets=targets,
        target_names=names,
        extra_features=extra_features,
        device=resolved_device,
    )

    train_target_values = data.target_values[train_rows]
    target_means = np.mean(train_target_values, axis=0, dtype=np.float64)
    target_stds = np.std(train_target_values, axis=0, ddof=1, dtype=np.float64)
    target_stds = np.maximum(target_stds, _MIN_TARGET_STD)
    if not bool(np.isfinite(target_means).all()) or not bool(np.isfinite(target_stds).all()):
        raise TrainingError("train-fold target normalizer is non-finite")
    means_tensor = torch.as_tensor(target_means, dtype=torch.float32, device=resolved_device)
    stds_tensor = torch.as_tensor(target_stds, dtype=torch.float32, device=resolved_device)
    normalized_targets = (data.targets - means_tensor) / stds_tensor
    if not bool(torch.isfinite(normalized_targets).all().item()):
        raise TrainingError("normalized training targets are non-finite")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=checked_learning_rate,
        weight_decay=checked_weight_decay,
        betas=beta_values,
        eps=checked_eps,
    )

    for epoch in range(checked_fixed_epochs):
        model.train()
        shuffle_digest = sha256(f"shuffle:{checked_seed}:{epoch}".encode("utf-8")).digest()
        shuffle_seed = int.from_bytes(shuffle_digest, "big")
        shuffled_rows = np.asarray(
            np.random.Generator(np.random.PCG64(shuffle_seed)).permutation(train_rows),
            dtype=np.int64,
        )
        for batch in _batch_rows(shuffled_rows, checked_batch_size):
            optimizer.zero_grad(set_to_none=True)
            predicted_means, predicted_log_stds = _forward_batch(torch, model, data, batch)
            batch_index = torch.as_tensor(
                batch,
                dtype=torch.long,
                device=resolved_device,
            )
            expected = normalized_targets[batch_index]
            absolute_errors = torch.abs(predicted_means - expected)
            if model_config.mve:
                if predicted_log_stds is None:
                    raise TrainingError("MVE model did not return log_std values")
                per_value_loss = (
                    math.sqrt(2.0)
                    * absolute_errors
                    * torch.exp(-predicted_log_stds)
                    + predicted_log_stds
                )
            else:
                per_value_loss = absolute_errors
            per_target_loss = per_value_loss.mean(dim=0)
            loss = per_target_loss.mean()
            if not bool(torch.isfinite(loss).item()):
                raise TrainingError(f"training loss became non-finite at epoch {epoch}")
            loss.backward()
            if checked_clip is not None:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), checked_clip)
                if not bool(torch.isfinite(total_norm).item()):
                    raise TrainingError(f"gradient norm became non-finite at epoch {epoch}")
            optimizer.step()

    checkpoint = serialize_checkpoint(
        model,
        model_config=model_config,
        best_epoch=checked_fixed_epochs - 1,
        target_names=names,
        target_means=target_means,
        target_stds=target_stds,
    )
    return FixedEpochTrainResult(
        target_names=names,
        trained_epochs=checked_fixed_epochs,
        checkpoint_bytes=checkpoint,
    )


def predict_component_targets(
    loaded: LoadedCheckpoint,
    *,
    features: ComponentFeatureSet,
    row_indices: Sequence[int],
    device: str = "cpu",
    batch_size: int = 32,
    extra_features: object | None = None,
    observer: InferenceTimingObserver | None = None,
) -> InferenceResult:
    """Predict physical-unit targets for ``row_indices`` with a restored checkpoint.

    This is the eval-OOD inference seam the LCO probe needs: MS7 training only returns
    validation-fold predictions, so the LCO adapter restores the best checkpoint and runs
    a separate forward pass over the held-out eval-OOD rows. Predictions are inverse-
    normalized with the checkpoint's own per-target normalizer and returned in ascending
    ``row_indices`` order; ``y_std`` is present only when the checkpoint carries an MVE head.
    """

    torch = _import_torch()
    MixNet, _MixNetConfig = _import_mixnet()
    _assert_torch_runtime(torch)
    if not isinstance(loaded, LoadedCheckpoint) or not isinstance(loaded.model, MixNet):
        raise TrainingError("loaded must be a LoadedCheckpoint with a MixNet model")
    if not isinstance(features, ComponentFeatureSet):
        raise TrainingError("features must be a ComponentFeatureSet")
    if not isinstance(device, str) or not device.strip() or device != device.strip():
        raise TrainingError("device must be a non-blank trimmed string")
    checked_batch_size = _checked_positive_int(batch_size, "batch_size")
    names = _checked_target_names(loaded.target_names)

    row_count = len(features.sample_ids)
    rows = _checked_indices(row_indices, label="row_indices", row_count=row_count)

    try:
        resolved_device = torch.device(device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise TrainingError(f"invalid torch device {device!r}: {exc}") from exc
    with observer.span("prepare", device) if observer is not None else nullcontext():
        ensure_torch_runtime(str(resolved_device))
        model = loaded.model.to(resolved_device)
        tensors = _prepare_component_tensors(
            torch,
            model_config=loaded.model_config,
            features=features,
            extra_features=extra_features,
            device=resolved_device,
        )
        data = _PreparedData(
            embeddings=tensors.embeddings,
            mass_fractions=tensors.mass_fractions,
            mask=tensors.mask,
            targets=None,
            target_values=np.zeros((row_count, len(names)), dtype=np.float64),
            extra_features=tensors.extra_features,
        )

        target_means = np.asarray(loaded.target_means, dtype=np.float64)
        target_stds = np.asarray(loaded.target_stds, dtype=np.float64)
        model.eval()
        normalized_parts: list[NDArray[np.float64]] = []
        log_std_parts: list[NDArray[np.float64]] = []
    if observer is not None:
        observer.logical_rows += len(rows)
    with torch.no_grad():
        for batch in _batch_rows(rows, checked_batch_size):
            with observer.span("forward", device) if observer is not None else nullcontext():
                normalized, log_stds = _forward_batch(torch, model, data, batch)
            if observer is not None:
                observer.physical_forwards += 1
            with observer.span("d2h", device) if observer is not None else nullcontext():
                normalized_array = np.asarray(normalized.detach().cpu().numpy(), dtype=np.float64)
            if not bool(np.isfinite(normalized_array).all()):
                raise TrainingError("inference predictions contain non-finite values")
            normalized_parts.append(normalized_array)
            if log_stds is not None:
                with observer.span("d2h", device) if observer is not None else nullcontext():
                    log_std_array = np.asarray(log_stds.detach().cpu().numpy(), dtype=np.float64)
                if not bool(np.isfinite(log_std_array).all()):
                    raise TrainingError("inference log_std predictions are non-finite")
                log_std_parts.append(log_std_array)

    normalized_predictions = np.concatenate(normalized_parts, axis=0)
    physical_predictions = normalized_predictions * target_stds + target_means
    if not bool(np.isfinite(physical_predictions).all()):
        raise TrainingError("inverse-normalized inference predictions are non-finite")

    physical_stds: NDArray[np.float64] | None = None
    if loaded.model_config.mve:
        if len(log_std_parts) != len(normalized_parts):
            raise TrainingError("MVE inference outputs are incomplete")
        with np.errstate(over="ignore", invalid="ignore"):
            physical_stds = np.exp(np.concatenate(log_std_parts, axis=0)) * target_stds
        if not bool(np.isfinite(physical_stds).all()) or bool((physical_stds <= 0).any()):
            raise TrainingError("MVE inference standard deviations are non-finite")

    return InferenceResult(
        target_names=names,
        row_indices=cast(NDArray[np.int64], _readonly_vector(rows, dtype=np.dtype(np.int64))),
        y_pred=_prediction_mapping(names, physical_predictions),
        y_std=None if physical_stds is None else _prediction_mapping(names, physical_stds),
    )
