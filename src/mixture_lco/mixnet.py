"""Ratio-weighted attention model for padded mixture compositions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_TARGET_COUNT: Final[int] = 4
_MESSAGE_LAYER_COUNTS: Final[frozenset[int]] = frozenset({0, 1, 2})


class MixNetError(ValueError):
    """Raised when MixNet configuration or inputs violate the model contract."""


def _check_integer(value: object, label: str, *, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise MixNetError(f"{label} must be a {qualifier} integer")


def _check_hidden_widths(
    value: object,
    label: str,
    *,
    require_non_empty: bool = False,
) -> None:
    if not isinstance(value, tuple):
        raise MixNetError(f"{label} must be a tuple of positive integers")
    if require_non_empty and not value:
        raise MixNetError(f"{label} must not be empty")
    for index, width in enumerate(value):
        _check_integer(width, f"{label}[{index}]", minimum=1)


@dataclass(frozen=True)
class MixNetConfig:
    """Validated, JSON-recordable MixNet architecture configuration."""

    input_dimension: int
    n_message_layers: int = 1
    feature_len: int = 64
    message_heads: int = 3
    pooling_heads: int = 3
    gate_hidden: tuple[int, ...] = (256,)
    message_hidden: tuple[int, ...] = (256,)
    trunk_hidden: tuple[int, ...] = (1024, 512)
    head_hidden: tuple[int, ...] = (256, 128, 64)
    mve: bool = False
    extra_features_len: int = 0

    def __post_init__(self) -> None:
        _check_integer(self.input_dimension, "input_dimension", minimum=1)
        if type(self.n_message_layers) is not int or self.n_message_layers not in (
            _MESSAGE_LAYER_COUNTS
        ):
            raise MixNetError("n_message_layers must be one of: 0, 1, 2")
        _check_integer(self.feature_len, "feature_len", minimum=2)
        _check_integer(self.message_heads, "message_heads", minimum=1)
        _check_integer(self.pooling_heads, "pooling_heads", minimum=1)
        _check_hidden_widths(self.gate_hidden, "gate_hidden")
        _check_hidden_widths(self.message_hidden, "message_hidden")
        _check_hidden_widths(self.trunk_hidden, "trunk_hidden")
        _check_hidden_widths(
            self.head_hidden,
            "head_hidden",
            require_non_empty=True,
        )
        if type(self.mve) is not bool:
            raise MixNetError("mve must be a bool")
        _check_integer(self.extra_features_len, "extra_features_len", minimum=0)

    def to_dict(self) -> dict[str, object]:
        """Return the architecture as JSON-native values for run provenance."""

        return {
            "input_dimension": self.input_dimension,
            "n_message_layers": self.n_message_layers,
            "feature_len": self.feature_len,
            "attention_heads": {
                "message": self.message_heads,
                "pooling": self.pooling_heads,
            },
            "gate_hidden": list(self.gate_hidden),
            "message_hidden": list(self.message_hidden),
            "trunk_hidden": list(self.trunk_hidden),
            "head_hidden": list(self.head_hidden),
            "mve": self.mve,
            "extra_features_len": self.extra_features_len,
        }


class _SimpleNetwork(nn.Module):
    """Small feed-forward network used for attention gates and messages."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        hidden_widths: tuple[int, ...],
    ) -> None:
        super().__init__()
        dimensions = (input_dimension, *hidden_widths)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(dimensions[index], dimensions[index + 1])
            for index in range(len(dimensions) - 1)
        )
        self.output_layer = nn.Linear(dimensions[-1], output_dimension)

    def forward(self, values: Tensor) -> Tensor:
        for layer in self.hidden_layers:
            values = F.leaky_relu(layer(values))
        return cast(Tensor, self.output_layer(values))


class _ResidualNetwork(nn.Module):
    """Feed-forward network with a projection residual at every hidden layer."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        hidden_widths: tuple[int, ...],
    ) -> None:
        super().__init__()
        dimensions = (input_dimension, *hidden_widths)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(dimensions[index], dimensions[index + 1])
            for index in range(len(dimensions) - 1)
        )
        self.residual_layers = nn.ModuleList(
            nn.Linear(dimensions[index], dimensions[index + 1], bias=False)
            if dimensions[index] != dimensions[index + 1]
            else nn.Identity()
            for index in range(len(dimensions) - 1)
        )
        self.output_layer = nn.Linear(dimensions[-1], output_dimension)

    def forward(self, values: Tensor) -> Tensor:
        for layer, residual_layer in zip(
            self.hidden_layers,
            self.residual_layers,
            strict=True,
        ):
            residual = residual_layer(values)
            values = F.relu(layer(values)) + residual
        return cast(Tensor, self.output_layer(values))


class _WeightedAttentionPooling(nn.Module):
    """Masked weighted soft attention over the penultimate tensor axis."""

    def __init__(self, gate: nn.Module, message: nn.Module) -> None:
        super().__init__()
        self.gate = gate
        self.message = message
        self._raw_power = nn.Parameter(torch.zeros(()))

    def forward(self, values: Tensor, mask: Tensor, weights: Tensor) -> Tensor:
        if values.ndim < 2 or values.shape[:-1] != mask.shape:
            raise MixNetError("attention values and mask shapes are not aligned")
        if weights.shape != mask.shape:
            raise MixNetError("attention weights and mask shapes are not aligned")
        if mask.dtype is not torch.bool:
            raise MixNetError("attention mask must have bool dtype")

        gates = self.gate(values).squeeze(-1)
        masked_gates = gates.masked_fill(~mask, -torch.inf)
        has_values = mask.any(dim=-1, keepdim=True)
        maximum = masked_gates.max(dim=-1, keepdim=True).values
        maximum = torch.where(has_values, maximum, torch.zeros_like(maximum))

        stable_weights = weights.clamp(min=1.0e-6, max=1.0e6)
        power = 2.0 * torch.tanh(self._raw_power)
        # Exponentiate the -inf-masked gates: padded slots contribute exp(-inf)=0
        # with a zero (not NaN) gradient, unlike masking after the exp.
        unnormalized = stable_weights.pow(power) * torch.exp(masked_gates - maximum)
        unnormalized = torch.where(mask, unnormalized, torch.zeros_like(unnormalized))
        attention = unnormalized / (unnormalized.sum(dim=-1, keepdim=True) + 1.0e-10)

        messages = self.message(values)
        weighted_messages = attention.unsqueeze(-1) * messages
        weighted_messages = torch.where(
            mask.unsqueeze(-1),
            weighted_messages,
            torch.zeros_like(weighted_messages),
        )
        return weighted_messages.sum(dim=-2)


class _MessageLayer(nn.Module):
    """Dense directed message passing over all non-self component pairs."""

    def __init__(
        self,
        feature_len: int,
        attention_heads: int,
        gate_hidden: tuple[int, ...],
        message_hidden: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.attention_heads = nn.ModuleList(
            _WeightedAttentionPooling(
                _SimpleNetwork(2 * feature_len, 1, gate_hidden),
                _SimpleNetwork(2 * feature_len, feature_len, message_hidden),
            )
            for _ in range(attention_heads)
        )

    def forward(self, node_features: Tensor, mass_fractions: Tensor, mask: Tensor) -> Tensor:
        component_count = node_features.shape[1]
        self_features = node_features.unsqueeze(2).expand(
            -1,
            -1,
            component_count,
            -1,
        )
        neighbor_features = node_features.unsqueeze(1).expand(
            -1,
            component_count,
            -1,
            -1,
        )
        pair_features = torch.cat((self_features, neighbor_features), dim=-1)

        non_self = ~torch.eye(
            component_count,
            dtype=torch.bool,
            device=mask.device,
        ).unsqueeze(0)
        pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1) & non_self
        neighbor_fractions = mass_fractions.unsqueeze(1).expand(
            -1,
            component_count,
            -1,
        )
        messages = torch.stack(
            [
                head(pair_features, pair_mask, neighbor_fractions)
                for head in self.attention_heads
            ],
            dim=0,
        ).mean(dim=0)
        updated = node_features + messages
        return torch.where(mask.unsqueeze(-1), updated, torch.zeros_like(updated))


class MixNet(nn.Module):
    """MixNet with dense masked message passing and ratio-weighted pooling."""

    def __init__(self, config: MixNetConfig) -> None:
        super().__init__()
        if not isinstance(config, MixNetConfig):
            raise MixNetError("config must be a MixNetConfig")
        self.config = config
        self.component_embedding = nn.Linear(
            config.input_dimension,
            config.feature_len - 1,
        )
        self.message_layers = nn.ModuleList(
            _MessageLayer(
                config.feature_len,
                config.message_heads,
                config.gate_hidden,
                config.message_hidden,
            )
            for _ in range(config.n_message_layers)
        )
        self.mixture_pooling = nn.ModuleList(
            _WeightedAttentionPooling(
                _SimpleNetwork(config.feature_len, 1, config.gate_hidden),
                _SimpleNetwork(
                    config.feature_len,
                    config.feature_len,
                    config.message_hidden,
                ),
            )
            for _ in range(config.pooling_heads)
        )

        trunk_output_dimension = config.head_hidden[0]
        self.trunk = _ResidualNetwork(
            config.feature_len,
            trunk_output_dimension,
            config.trunk_hidden,
        )
        head_input_dimension = trunk_output_dimension + config.extra_features_len
        head_output_dimension = 2 if config.mve else 1
        self.target_heads = nn.ModuleList(
            _ResidualNetwork(
                head_input_dimension,
                head_output_dimension,
                config.head_hidden[1:],
            )
            for _ in range(_TARGET_COUNT)
        )

    def _checked_inputs(
        self,
        embeddings: Tensor,
        mass_fractions: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(embeddings, Tensor) or embeddings.ndim != 3:
            raise MixNetError("embeddings must be a rank-3 torch.Tensor")
        batch_size, component_count, input_dimension = embeddings.shape
        if batch_size < 1 or component_count < 1:
            raise MixNetError("embeddings must contain at least one sample and component slot")
        if input_dimension != self.config.input_dimension:
            raise MixNetError(
                f"embeddings dimension is {input_dimension}; "
                f"expected {self.config.input_dimension}"
            )
        if embeddings.dtype is not torch.float32:
            raise MixNetError("embeddings must have float32 dtype")
        if not isinstance(mass_fractions, Tensor) or mass_fractions.shape != (
            batch_size,
            component_count,
        ):
            raise MixNetError("mass_fractions must align with embeddings")
        if not mass_fractions.is_floating_point():
            raise MixNetError("mass_fractions must have a floating-point dtype")
        if not isinstance(mask, Tensor) or mask.shape != (batch_size, component_count):
            raise MixNetError("mask must align with embeddings")
        if mask.dtype is not torch.bool:
            raise MixNetError("mask must have bool dtype")
        if embeddings.device != mass_fractions.device or embeddings.device != mask.device:
            raise MixNetError("embeddings, mass_fractions, and mask must share a device")
        if not bool(mask.any(dim=1).all().item()):
            raise MixNetError("every sample must contain at least one active component")

        active_embeddings = embeddings.masked_select(mask.unsqueeze(-1))
        active_fractions = mass_fractions.masked_select(mask)
        if not bool(torch.isfinite(active_embeddings).all().item()):
            raise MixNetError("active embeddings must contain only finite values")
        if not bool(torch.isfinite(active_fractions).all().item()):
            raise MixNetError("active mass_fractions must contain only finite values")
        if bool((active_fractions < 0).any().item()):
            raise MixNetError("active mass_fractions must be non-negative")

        clean_embeddings = torch.where(
            mask.unsqueeze(-1),
            embeddings,
            torch.zeros_like(embeddings),
        )
        converted_fractions = mass_fractions.to(dtype=embeddings.dtype)
        if not bool(
            torch.isfinite(converted_fractions.masked_select(mask)).all().item()
        ):
            raise MixNetError(
                "active mass_fractions must remain finite when converted to float32"
            )
        clean_fractions = torch.where(
            mask,
            converted_fractions,
            torch.zeros_like(mass_fractions, dtype=embeddings.dtype),
        )
        if bool((clean_fractions.sum(dim=1) <= 0).any().item()):
            raise MixNetError("every sample must have positive total mass_fraction")
        return clean_embeddings, clean_fractions

    def _checked_extra_features(
        self,
        extra_features: Tensor | None,
        *,
        batch_size: int,
        reference: Tensor,
    ) -> Tensor | None:
        expected = self.config.extra_features_len
        if expected == 0:
            if extra_features is not None:
                raise MixNetError("extra_features were provided but the model expects none")
            return None
        if extra_features is None:
            raise MixNetError("extra_features are required by this model configuration")
        if not isinstance(extra_features, Tensor) or extra_features.shape != (
            batch_size,
            expected,
        ):
            raise MixNetError(
                f"extra_features must have shape ({batch_size}, {expected})"
            )
        if not extra_features.is_floating_point():
            raise MixNetError("extra_features must have a floating-point dtype")
        if extra_features.device != reference.device:
            raise MixNetError("extra_features must share the embeddings device")
        if not bool(torch.isfinite(extra_features).all().item()):
            raise MixNetError("extra_features must contain only finite values")
        converted = extra_features.to(dtype=reference.dtype)
        if not bool(torch.isfinite(converted).all().item()):
            raise MixNetError(
                "extra_features must remain finite when converted to float32"
            )
        return converted

    def forward(
        self,
        embeddings: Tensor,
        mass_fractions: Tensor,
        mask: Tensor,
        extra_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Predict four targets for a padded batch of mixture components."""

        clean_embeddings, clean_fractions = self._checked_inputs(
            embeddings,
            mass_fractions,
            mask,
        )
        checked_extra = self._checked_extra_features(
            extra_features,
            batch_size=clean_embeddings.shape[0],
            reference=clean_embeddings,
        )
        return self._compute(clean_embeddings, clean_fractions, mask, checked_extra)

    def forward_validated(
        self,
        embeddings: Tensor,
        mass_fractions: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute from fixed buffers whose owner already validated their inputs.

        Only the resident inference scorer uses this entry point. All value checks
        belong to its host input boundary so this path can be captured by CUDA Graphs.
        """

        clean_embeddings = torch.where(
            mask.unsqueeze(-1), embeddings, torch.zeros_like(embeddings)
        )
        clean_fractions = torch.where(
            mask,
            mass_fractions.to(dtype=embeddings.dtype),
            torch.zeros_like(mass_fractions, dtype=embeddings.dtype),
        )
        return self._compute(clean_embeddings, clean_fractions, mask, None)

    def _compute(
        self,
        clean_embeddings: Tensor,
        clean_fractions: Tensor,
        mask: Tensor,
        extra: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Tensor-only model computation, shared by both validation boundaries."""

        projected = self.component_embedding(clean_embeddings)
        node_features = torch.cat((projected, clean_fractions.unsqueeze(-1)), dim=-1)
        node_features = torch.where(
            mask.unsqueeze(-1),
            node_features,
            torch.zeros_like(node_features),
        )
        for message_layer in self.message_layers:
            node_features = message_layer(node_features, clean_fractions, mask)

        pooled = torch.stack(
            [
                head(node_features, mask, clean_fractions)
                for head in self.mixture_pooling
            ],
            dim=0,
        ).mean(dim=0)
        trunk_features = F.relu(self.trunk(pooled))
        if extra is not None:
            trunk_features = torch.cat((trunk_features, extra), dim=-1)

        outputs = tuple(head(trunk_features) for head in self.target_heads)
        return outputs[0], outputs[1], outputs[2], outputs[3]
