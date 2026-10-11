"""Private validation of immutable natural-language autoencoder artifacts."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from string import Formatter
from typing import Any, Literal, Mapping, Protocol, cast, runtime_checkable

import torch
import yaml
from huggingface_hub import snapshot_download
from safetensors import safe_open

from transformer_lens.utilities.hf_utils import call_hf_with_retry

Role = Literal["av", "ar"]


@runtime_checkable
class Tokenizer(Protocol):
    """Tokenization operations needed to check the artifact contract."""

    unk_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        ...

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> object:
        ...


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping with string keys")
    return cast(Mapping[str, Any], value)


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _ids(value: object, name: str) -> tuple[int, ...]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must contain token ids")
    return tuple(_integer(item, name) for item in value)


def _template(value: object, field: str, name: str) -> str:
    text = _text(value, name)
    try:
        parts = list(Formatter().parse(text))
    except ValueError as exc:
        raise ValueError(f"{name} is not a valid template") from exc
    fields = [key for _, key, _, _ in parts if key is not None]
    if fields != [field] or any(spec or conversion for _, _, spec, conversion in parts):
        raise ValueError(f"{name} must contain exactly one {{{field}}} field")
    return text


@dataclass(frozen=True)
class ArtifactRef:
    """One Hub repository at an immutable commit revision."""

    repo_id: str
    revision: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", self.repo_id):
            raise ValueError("repo_id must be an owner/repository identifier")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("revision must be a lowercase immutable 40-character commit SHA")


@dataclass(frozen=True)
class NLAPairConfig:
    """Pinned target and role identities, not proof of numerical compatibility."""

    target: ArtifactRef
    av: ArtifactRef
    ar: ArtifactRef
    model_type: str
    d_model: int
    layer: int
    target_layers: int

    def __post_init__(self) -> None:
        _text(self.model_type, "model_type")
        _integer(self.d_model, "d_model", minimum=1)
        _integer(self.layer, "layer")
        _integer(self.target_layers, "target_layers", minimum=1)
        if self.layer >= self.target_layers:
            raise ValueError("layer must be below target_layers")
        if len({self.target.repo_id, self.av.repo_id, self.ar.repo_id}) != 3:
            raise ValueError("target, av, and ar must identify distinct repositories")


def parse_registry(value: object) -> dict[str, NLAPairConfig]:
    """Parse supported entries without network access or model loading."""
    registry = _mapping(value, "registry")
    if _integer(registry.get("schema_version"), "registry schema_version") != 1:
        raise ValueError("unsupported registry schema_version")
    entries = _mapping(registry.get("models"), "registry models")
    result = {}
    for name, raw in entries.items():
        _text(name, "registry name")
        entry = _mapping(raw, "registry entry")
        refs = {}
        for role in ("target", "av", "ar"):
            ref = _mapping(entry.get(role), role)
            refs[role] = ArtifactRef(
                _text(ref.get("repo_id"), "repo_id"), _text(ref.get("revision"), "revision")
            )
        result[name] = NLAPairConfig(
            refs["target"],
            refs["av"],
            refs["ar"],
            _text(entry.get("model_type"), "model_type"),
            _integer(entry.get("d_model"), "d_model", minimum=1),
            _integer(entry.get("layer"), "layer"),
            _integer(entry.get("target_layers"), "target_layers", minimum=1),
        )
    return result


def load_registry() -> dict[str, NLAPairConfig]:
    """Read bundled supported pairs through package resources."""
    resource = files("transformer_lens.tools.analysis").joinpath("nla_registry.json")
    return parse_registry(json.loads(resource.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class NLAMetadata:
    """Validated role-specific metadata and canonical tokenization."""

    role: Role
    d_model: int
    layer: int
    injection_char: str
    injection_token_id: int
    actor_template: str
    critic_template: str
    mse_scale: float
    injection_scale: float | None
    actor_token_ids: tuple[int, ...] | None
    injection_position: int | None
    critic_suffix_ids: tuple[int, ...] | None


def validate_metadata(
    value: object,
    model_config: object,
    pair: NLAPairConfig,
    role: Role,
    tokenizer: Tokenizer,
) -> NLAMetadata:
    """Check schema-v2 sidecars against the pinned contract and live tokenizer."""
    if role not in ("av", "ar"):
        raise ValueError("role must be av or ar")
    meta = _mapping(value, "nla_meta")
    config = _mapping(model_config, "model config")
    if meta.get("kind") != "nla_model":
        raise ValueError("kind must be nla_model")
    if _integer(meta.get("schema_version"), "schema_version") != 2:
        raise ValueError("unsupported sidecar schema_version")
    if meta.get("role") != role:
        raise ValueError("sidecar role does not match requested role")
    dimension = _integer(meta.get("d_model"), "d_model", minimum=1)
    layer = _integer(meta.get("extraction_layer_index"), "extraction_layer_index")
    if dimension != pair.d_model or layer != pair.layer:
        raise ValueError("sidecar dimension/layer does not match the pinned pair")
    if config.get("model_type") != pair.model_type:
        raise ValueError("model_type does not match the pinned pair")
    if _integer(config.get("hidden_size"), "hidden_size", minimum=1) != dimension:
        raise ValueError("model hidden_size does not match sidecar")
    depth = _integer(config.get("num_hidden_layers"), "num_hidden_layers", minimum=1)
    if depth != (pair.target_layers if role == "av" else layer + 1):
        raise ValueError("model depth does not match the artifact role")
    critic = meta.get("critic")
    if critic is not None:
        critic_layer = _integer(
            _mapping(critic, "critic").get("extraction_layer_index"), "critic layer"
        )
        if critic_layer != layer:
            raise ValueError("critic layer does not match extraction_layer_index")
    extraction = _mapping(meta.get("extraction"), "extraction")
    if "injection_scale" not in extraction:
        raise ValueError("extraction must record injection_scale, including null for ar")
    mse_scale = _positive(extraction.get("mse_scale"), "mse_scale")
    # Schema-v2 Qwen sidecars serialize sqrt(d) as a float64 value with 16 digits.
    if not math.isclose(mse_scale, math.sqrt(dimension), rel_tol=1e-14, abs_tol=0):
        raise ValueError("unsupported mse_scale; expected sqrt(d_model) normalization")
    tokens = _mapping(meta.get("tokens"), "tokens")
    if "critic_suffix_ids" not in tokens:
        raise ValueError("tokens must record critic_suffix_ids, including null for av")
    char = _text(tokens.get("injection_char"), "injection_char")
    token_id = _integer(tokens.get("injection_token_id"), "injection_token_id")
    left = _integer(tokens.get("injection_left_neighbor_id"), "left neighbor")
    right = _integer(tokens.get("injection_right_neighbor_id"), "right neighbor")
    live_id = _ids(tokenizer.encode(char, add_special_tokens=False), "live injection ids")
    if live_id != (token_id,) or token_id == tokenizer.unk_token_id:
        raise ValueError("injection token encoding drift or UNK")
    templates = _mapping(meta.get("prompt_templates"), "prompt_templates")
    actor_template = _template(templates.get("av"), "injection_char", "av template")
    critic_template = _template(templates.get("ar"), "explanation", "ar template")
    scale = None
    actor_ids = None
    position = None
    suffix = None
    if role == "av":
        scale = _positive(extraction.get("injection_scale"), "injection_scale")
        actor_ids = _ids(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": actor_template.format(injection_char=char)}],
                tokenize=True,
                add_generation_prompt=True,
            ),
            "actor token ids",
        )
        matches = [i for i, item in enumerate(actor_ids) if item == token_id]
        if len(matches) != 1:
            raise ValueError("actor prompt must contain exactly one injection token")
        position = matches[0]
        if not 0 < position < len(actor_ids) - 1:
            raise ValueError("injection token must have both neighbors")
        if actor_ids[position - 1] != left or actor_ids[position + 1] != right:
            raise ValueError("actor prompt neighbor token drift")
        if tokens.get("critic_suffix_ids") is not None:
            raise ValueError("av critic_suffix_ids must be null")
    else:
        if extraction.get("injection_scale") is not None:
            raise ValueError("ar injection_scale must be null")
        suffix = _ids(tokens.get("critic_suffix_ids"), "critic_suffix_ids")
        probe = _ids(
            tokenizer.encode(critic_template.format(explanation="x"), add_special_tokens=True),
            "critic probe ids",
        )
        if len(probe) < len(suffix) or probe[-len(suffix) :] != suffix:
            raise ValueError("critic suffix token drift")
    return NLAMetadata(
        role,
        dimension,
        layer,
        char,
        token_id,
        actor_template,
        critic_template,
        mse_scale,
        scale,
        actor_ids,
        position,
        suffix,
    )


def load_local_metadata(
    directory: Path, pair: NLAPairConfig, role: Role, tokenizer: Tokenizer
) -> NLAMetadata:
    """Validate files from one local snapshot without acquiring any remote files."""
    try:
        meta = yaml.safe_load((directory / "nla_meta.yaml").read_text(encoding="utf-8"))
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ValueError("invalid artifact metadata syntax") from exc
    return validate_metadata(meta, config, pair, role, tokenizer)


def validate_snapshot_path(directory: Path, ref: ArtifactRef) -> Path:
    """Require the cache namespace and revision of the requested repository."""
    expected_namespace = "models--" + ref.repo_id.replace("/", "--")
    if (
        directory.name != ref.revision
        or directory.parent.name != "snapshots"
        or directory.parent.parent.name != expected_namespace
    ):
        raise ValueError("Hub snapshot path does not match the pinned repository and revision")
    return directory


def acquire_snapshot(
    ref: ArtifactRef,
    *,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    include_weights: bool = False,
) -> Path:
    """Acquire one pinned snapshot, excluding large model weights by default."""
    patterns = [
        "nla_meta.yaml",
        "config.json",
        "tokenizer*",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "chat_template*",
        "generation_config.json",
    ]
    if include_weights:
        patterns.extend(["*.safetensors", "*.safetensors.index.json"])
    directory = call_hf_with_retry(
        snapshot_download,
        repo_id=ref.repo_id,
        revision=ref.revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        allow_patterns=patterns,
    )
    return validate_snapshot_path(Path(directory), ref)


def load_snapshot_tokenizer(directory: Path) -> Tokenizer:
    """Load only local tokenizer assets, with remote code disabled."""
    from transformers import AutoTokenizer

    return cast(
        Tokenizer,
        AutoTokenizer.from_pretrained(
            str(directory),
            local_files_only=True,
            trust_remote_code=False,
        ),
    )


def validate_value_head(state: Mapping[str, torch.Tensor], d_model: int) -> None:
    """Require the released bias-free Linear(d,d) state contract."""
    _integer(d_model, "d_model", minimum=1)
    if set(state) != {"weight"}:
        raise ValueError("value head must contain only weight, with no bias")
    weight = state["weight"]
    if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != (d_model, d_model):
        raise ValueError("value head weight must have shape (d_model, d_model)")
    if not weight.is_floating_point() or not torch.isfinite(weight).all().item():
        raise ValueError("value head weight must be finite floating point")


def validate_value_head_file(path: Path, d_model: int) -> None:
    """Inspect safetensors keys and shape without materializing a full backbone."""
    _integer(d_model, "d_model", minimum=1)
    with safe_open(path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {"weight"}:
            raise ValueError("value head must contain only weight, with no bias")
        if handle.get_slice("weight").get_shape() != [d_model, d_model]:
            raise ValueError("value head weight must have shape (d_model, d_model)")
        validate_value_head({"weight": handle.get_tensor("weight")}, d_model)
