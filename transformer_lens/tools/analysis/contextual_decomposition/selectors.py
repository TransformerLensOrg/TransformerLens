"""Private single-source selectors and target projection for GPT-2 CD."""

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from transformer_lens.tools.analysis.contextual_decomposition.rules import (
    _CDConstituents,
)


@dataclass(frozen=True)
class _CDSource:
    """A residual-space source anchored at one GPT-2 residual merge.

    Head sources exclude shared output bias and anchor after attention. Residual
    locations are block input, after attention (mid), or block output (post).
    Positions select a common set of absolute indices across the input batch.
    """

    layer: int
    kind: Literal["head", "residual"] = "residual"
    location: Literal["pre", "mid", "post"] = "pre"
    head: Optional[int] = None
    positions: Optional[tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if isinstance(self.layer, bool) or not isinstance(self.layer, int) or self.layer < 0:
            raise ValueError("source layer must be a nonnegative integer")
        if self.kind not in ("head", "residual") or self.location not in ("pre", "mid", "post"):
            raise ValueError("unsupported source kind or residual location")
        if self.kind == "head":
            if isinstance(self.head, bool) or not isinstance(self.head, int) or self.head < 0:
                raise ValueError("head source requires a nonnegative head index")
            if self.location != "mid":
                raise ValueError("head sources anchor at the mid residual merge")
        elif self.head is not None:
            raise ValueError("residual source cannot select a head")
        if self.positions is not None:
            if not isinstance(self.positions, tuple) or not self.positions:
                raise ValueError("source positions must be a nonempty tuple")
            if any(isinstance(p, bool) or not isinstance(p, int) or p < 0 for p in self.positions):
                raise ValueError("source positions must be nonnegative integers")
            if len(set(self.positions)) != len(self.positions):
                raise ValueError("source positions must be unique")

    @property
    def anchor(self) -> str:
        suffix = {"pre": "hook_in", "mid": "ln2.hook_in", "post": "hook_out"}[self.location]
        return f"blocks.{self.layer}.{suffix}"


@dataclass(frozen=True)
class _CDTarget:
    """An internal tensor or per-example output logit projection.

    Internal head indices apply only to Q/K/V and attention context coordinates.
    Positions and token IDs accept one integer or a tuple with one per example.
    """

    kind: Literal["internal", "logit", "logit_difference"]
    hook_name: Optional[str] = None
    position: Optional[int | tuple[int, ...]] = None
    token_id: Optional[int | tuple[int, ...]] = None
    negative_token_id: Optional[int | tuple[int, ...]] = None
    head: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kind not in ("internal", "logit", "logit_difference"):
            raise ValueError("unsupported target kind")
        if self.kind == "internal":
            if (
                not self.hook_name
                or self.token_id is not None
                or self.negative_token_id is not None
            ):
                raise ValueError("internal target requires a hook name, without token IDs")
        else:
            if (
                self.hook_name is not None
                or self.head is not None
                or self.token_id is None
                or self.position is None
            ):
                raise ValueError(
                    "logit target requires position and token ID, without hook or head"
                )
            if (self.kind == "logit_difference") != (self.negative_token_id is not None):
                raise ValueError("negative token ID is required only for a logit difference")
        if self.head is not None and (
            isinstance(self.head, bool) or not isinstance(self.head, int) or self.head < 0
        ):
            raise ValueError("target head must be a nonnegative integer")
        for name, value in (
            ("position", self.position),
            ("token_id", self.token_id),
            ("negative_token_id", self.negative_token_id),
        ):
            if value is None:
                continue
            values = value if isinstance(value, tuple) else (value,)
            if not values or any(
                isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values
            ):
                raise ValueError(f"target {name} must contain nonnegative integer selectors")


def _target_stage(name: str, n_layers: int) -> int:
    if name in ("ln_final.hook_out", "unembed.hook_out"):
        return 10 * n_layers + (1 if name.startswith("ln_final") else 2)
    fields = name.split(".")
    if len(fields) < 3 or fields[0] != "blocks" or not fields[1].isdigit():
        raise ValueError("unsupported internal target hook")
    layer = int(fields[1])
    suffix = ".".join(fields[2:])
    stages = {
        "hook_in": 0,
        "ln1.hook_out": 1,
        "attn.q.hook_out": 2,
        "attn.k.hook_out": 2,
        "attn.v.hook_out": 2,
        "attn.hook_pattern": 3,
        "attn.o.hook_in": 4,
        "attn.hook_out": 5,
        "ln2.hook_in": 6,
        "ln2.hook_out": 7,
        "mlp.in.hook_out": 8,
        "mlp.out.hook_in": 8,
        "mlp.hook_out": 8,
        "hook_out": 9,
    }
    if layer >= n_layers or suffix not in stages:
        raise ValueError("unsupported internal target hook or layer")
    return layer * 10 + stages[suffix]


def _validate_source_target(
    source: _CDSource,
    target: Optional[_CDTarget],
    *,
    batch: int,
    sequence: int,
    n_layers: int,
    n_heads: int,
    vocab_size: int,
) -> None:
    if source.layer >= n_layers or (source.head is not None and source.head >= n_heads):
        raise ValueError("source layer or head is out of range")
    if source.positions is not None and max(source.positions) >= sequence:
        raise ValueError("source position is out of range")
    if target is None:
        return
    name = target.hook_name if target.kind == "internal" else "unembed.hook_out"
    assert name is not None
    if _target_stage(name, n_layers) <= _target_stage(source.anchor, n_layers):
        raise ValueError("target must be strictly downstream of the source anchor")
    if target.head is not None:
        if target.head >= n_heads or not name.endswith(
            ("attn.q.hook_out", "attn.k.hook_out", "attn.v.hook_out", "attn.o.hook_in")
        ):
            raise ValueError("target head is invalid for this tensor space")
    if name.endswith("attn.hook_pattern") and target.position is not None:
        raise ValueError("probability target position is ambiguous; select the whole tensor")
    for label, value, bound in (
        ("position", target.position, sequence),
        ("token_id", target.token_id, vocab_size),
        ("negative_token_id", target.negative_token_id, vocab_size),
    ):
        if value is None:
            continue
        values = value if isinstance(value, tuple) else (value,)
        if len(values) not in (1, batch) or (isinstance(value, tuple) and len(values) != batch):
            raise ValueError(f"target {label} tuple must contain one selector per example")
        if max(values) >= bound:
            raise ValueError(f"target {label} is out of range")


@torch.no_grad()
def _select_target(parts: _CDConstituents, target: _CDTarget) -> _CDConstituents:
    def select(value: torch.Tensor) -> torch.Tensor:
        output = value
        rows = torch.arange(value.shape[0], device=value.device)
        if target.position is not None:
            positions = torch.tensor(target.position, device=value.device)
            output = output[rows, positions]
        if target.head is not None:
            output = (
                output[:, target.head] if target.position is not None else output[:, :, target.head]
            )
        if target.kind != "internal":
            tokens = torch.tensor(target.token_id, device=value.device)
            positive = output[rows, tokens]
            if target.negative_token_id is not None:
                negative = torch.tensor(target.negative_token_id, device=value.device)
                positive = positive - output[rows, negative]
            output = positive
        return output.clone()

    return _CDConstituents(select(parts.beta), select(parts.gamma))
