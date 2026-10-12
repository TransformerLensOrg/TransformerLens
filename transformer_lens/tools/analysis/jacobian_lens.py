"""Jacobian lens (J-lens), from https://transformer-circuits.pub/2026/workspace/index.html.

``J[l]`` is d(final block output)/d(block ``l`` output), summed over targets ``t' >= t`` and
averaged over source positions and prompts; the lens reads ``W_U norm(J[l] h)`` (``J = I`` is the
logit lens). Fitting and the artifact format follow anthropics/jacobian-lens, so Hub artifacts
load as-is. Only a freshly booted raw ``TransformerBridge`` is supported: compatibility mode and
processed weights change the residual basis and are refused."""

import json
import warnings
from dataclasses import dataclass
from importlib.metadata import version
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
from jaxtyping import Float, Int
from tqdm.auto import tqdm

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.tools.analysis._model_state import require_eval_mode
from transformer_lens.tools.analysis.jacobian_lens_coordinate_patch import (
    CoordinatePatch,
    solve_coordinate_patch,
    solve_coordinate_patch_positions,
)
from transformer_lens.tools.analysis.jacobian_lens_decomposition import (
    DEFAULT_K,
    JSpaceDecomposition,
    JSpaceOccupancy,
    JSpaceVarianceProfile,
    _diagnose_intervention_pair,
    estimate_occupancy,
    get_sparse_decomposition,
)
from transformer_lens.utilities.hf_utils import call_hf_with_retry

TokenInput = Union[str, int]
_LensT = TypeVar("_LensT", bound="JacobianLens")

# (target, sources, cotangent, retain_graph) -> one grad per source, like torch.autograd.grad;
# another estimator reuses the fit loop by swapping only this backward step.
BackwardProvider = Callable[
    [torch.Tensor, List[torch.Tensor], torch.Tensor, bool],
    Tuple[torch.Tensor, ...],
]

_REGISTRY_CACHE: Optional[Dict[str, Any]] = None


def _load_registry() -> Dict[str, Any]:
    """Return the bundled artifact registry, loading it once on first call."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        import pathlib

        registry_path = pathlib.Path(__file__).with_name("jacobian_lens_registry.json")
        with registry_path.open() as fh:
            _REGISTRY_CACHE = json.load(fh)
    return _REGISTRY_CACHE


def _resolve_registry_entry(name_or_path: str) -> Optional[Tuple[str, str]]:
    """Return ``(repo_id, filename)`` for a registry short name or HF-model-id alias, else None."""
    registry = _load_registry()
    if name_or_path in registry:
        entry = registry[name_or_path]
        return entry["repo_id"], entry["filename"]
    for entry in registry.values():
        if isinstance(entry, dict) and name_or_path in entry.get("aliases", []):
            return entry["repo_id"], entry["filename"]
    return None


# Fitting excludes early positions (attention sinks with atypical residual statistics)
# and the final position (no next-token target), matching the reference implementation.
DEFAULT_SKIP_FIRST_POSITIONS = 16
DEFAULT_TOP_K = 10

# Recorded on every fitted lens; merge() refuses shards whose estimators differ.
ESTIMATOR_JACOBIAN = "jacobian_lens"

# Ordinary-Jacobian estimator labels; workspace-lenses J-lens arms record "standard".
_JACOBIAN_ESTIMATOR_NAMES: frozenset = frozenset({ESTIMATOR_JACOBIAN, "standard"})

# Stripped from converted checkpoints so merge() refuses to mix them with TL fits; target_layer
# is deliberately absent so validate_model() can still refuse non-final targets.
_FIT_RESERVED_KEYS: frozenset = frozenset(
    {
        "transformer_lens_fit",
        "transformer_lens_version",
        "model_system",
        "processing",
        "hook_convention",
        "fit_dtype",
        "dim_batch",
        "max_seq_len",
        "skip_first_positions",
    }
)

# Provenance some checkpoint writers store at the top level rather than under "metadata".
_CHECKPOINT_FLAT_PROVENANCE: frozenset = frozenset({"model_name", "model_revision", "corpus"})


@dataclass
class JacobianLensReadout:
    """Result of :meth:`JacobianLens.readout`; tensors are on CPU and ``pos`` axes align with the
    normalized non-negative ``positions``. ``use_jacobian=False`` means a logit-lens readout."""

    lens_topk_values: Dict[int, Float[torch.Tensor, "pos k"]]
    lens_topk_indices: Dict[int, Int[torch.Tensor, "pos k"]]
    model_topk_values: Float[torch.Tensor, "pos k"]
    model_topk_indices: Int[torch.Tensor, "pos k"]
    tokens: Int[torch.Tensor, "seq"]
    positions: List[int]
    use_jacobian: bool = True
    lens_logits: Optional[Dict[int, Float[torch.Tensor, "pos d_vocab"]]] = None
    model_logits: Optional[Float[torch.Tensor, "pos d_vocab"]] = None

    def top_tokens(self, tokenizer: Any, k: int = 5) -> Dict[int, List[List[str]]]:
        """Decode ``{layer: [top-k strings per position]}``."""
        out: Dict[int, List[List[str]]] = {}
        retained = next(iter(self.lens_topk_indices.values())).shape[-1]
        if not 1 <= k <= retained:
            raise ValueError(f"k must be between 1 and the retained top_k={retained}, got {k}")
        for layer, ids in self.lens_topk_indices.items():
            out[layer] = [[tokenizer.decode([t]) for t in row[:k].tolist()] for row in ids]
        return out


class JacobianLens:
    """Fitted lens: ``J[l]`` maps ``blocks.{l}.hook_out`` to the final block's pre-norm output,
    as fp32 CPU matrices; the final layer is the identity and is never fitted."""

    # The bundled registry holds only Jacobian lens artifacts; other estimators opt out.
    _uses_artifact_registry: bool = True

    def __init__(
        self,
        jacobians: Dict[int, Float[torch.Tensor, "d_model d_model"]],
        *,
        n_prompts: int,
        d_model: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not jacobians:
            raise ValueError("jacobians must contain at least one layer")
        for layer, matrix in jacobians.items():
            if matrix.shape != (d_model, d_model):
                raise ValueError(
                    f"jacobians[{layer}] has shape {tuple(matrix.shape)}, "
                    f"expected ({d_model}, {d_model})"
                )
        self.jacobians: Dict[int, torch.Tensor] = {
            int(layer): matrix.detach().float().cpu() for layer, matrix in jacobians.items()
        }
        self.n_prompts = int(n_prompts)
        self.d_model = int(d_model)
        self.metadata: Dict[str, Any] = dict(metadata or {})
        recorded_estimator = self.metadata.get("estimator")
        self.estimator: str = (
            recorded_estimator if isinstance(recorded_estimator, str) else ESTIMATOR_JACOBIAN
        )
        self._device_jacobians: Dict[Tuple[int, torch.device], torch.Tensor] = {}
        self._dictionary_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}
        self._unembedding_snapshots: Dict[torch.device, torch.Tensor] = {}

    @property
    def source_layers(self) -> List[int]:
        """Sorted fitted source layers."""
        return sorted(self.jacobians)

    def __repr__(self) -> str:
        layers = self.source_layers
        return (
            f"JacobianLens(layers={layers[0]}..{layers[-1]} ({len(layers)}), "
            f"d_model={self.d_model}, n_prompts={self.n_prompts})"
        )

    @classmethod
    def _validate_artifact_metadata(cls, path: str, metadata: Any) -> None:
        """Allow legacy artifacts without estimator provenance."""

    @classmethod
    def _validate_checkpoint_payload(cls, path: str, payload: Dict[str, Any]) -> None:
        """Allow conversion of reference Jacobian running-sum checkpoints."""

    @classmethod
    def _validate_merge_inputs(cls, lenses: Sequence["JacobianLens"]) -> None:
        """Allow generic lenses whose recorded provenance agrees."""

    def save(self, path: str, *, dtype: torch.dtype = torch.float16) -> None:
        """Save in the reference artifact format plus an additive ``metadata`` key; fp16 by
        default like the reference, since Jacobian entries are order-one."""
        _validate_metadata(self.metadata)
        payload: Dict[str, Any] = {
            "J": {layer: matrix.to(dtype) for layer, matrix in self.jacobians.items()},
            "n_prompts": self.n_prompts,
            "source_layers": self.source_layers,
            "d_model": self.d_model,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        torch.save(payload, path)

    @classmethod
    def load(cls: Type[_LensT], path: str) -> _LensT:
        """Load a ``J`` artifact or a reference ``jacobian_sum`` fit checkpoint; both schemas are in
        docs/source/content/jacobian_lens_fitting.md."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if "J" in payload:
            metadata = _harvest_artifact_provenance(path, payload)
            cls._validate_artifact_metadata(path, metadata)
            return cls(
                {int(layer): matrix for layer, matrix in payload["J"].items()},
                n_prompts=int(payload.get("n_prompts", 0)),
                d_model=int(payload["d_model"]),
                metadata=metadata,
            )
        if "jacobian_sum" in payload:
            cls._validate_checkpoint_payload(path, payload)
            return cls._from_checkpoint_payload(path, payload)
        raise ValueError(
            f"{path} does not look like a Jacobian lens artifact or fit checkpoint. "
            "Expected a 'J' key (artifact) or 'jacobian_sum' key (fit checkpoint). "
            "See JacobianLens.load() for the supported file schemas."
        )

    @classmethod
    def _from_checkpoint_payload(cls: Type[_LensT], path: str, payload: Dict[str, Any]) -> _LensT:
        """Divide running sums by the prompt count and keep only reload-safe, non-fit-reserved
        provenance, marked ``converted_from`` so :meth:`merge` refuses to mix it with TL fits."""
        n_prompts = int(payload.get("n_done", payload.get("n_prompts", 0)))
        if n_prompts <= 0:
            raise ValueError(
                f"{path} is a fit checkpoint with n_prompts={n_prompts}; "
                "a positive prompt count is required to reconstruct the Jacobian mean."
            )
        if not payload["jacobian_sum"]:
            raise ValueError(
                f"{path} is a fit checkpoint with an empty jacobian_sum; "
                "at least one layer matrix is required to reconstruct d_model."
            )
        first_matrix = next(iter(payload["jacobian_sum"].values()))
        d_model = first_matrix.shape[0]
        jacobians = {
            int(layer): matrix.float() / n_prompts
            for layer, matrix in payload["jacobian_sum"].items()
        }

        raw_meta: Dict[str, Any] = dict(payload.get("metadata") or {})
        for key in _CHECKPOINT_FLAT_PROVENANCE:
            if key in payload and key not in raw_meta:
                raw_meta[key] = payload[key]
        # Harvested so validate_model() can refuse checkpoints fitted against a non-final target.
        if "target_layer" in payload and "target_layer" not in raw_meta:
            raw_meta["target_layer"] = payload["target_layer"]

        # Tensors cannot survive a weights_only=True reload; record what was dropped.
        dropped_fields: List[str] = []
        clean_meta: Dict[str, Any] = {}
        for key, value in raw_meta.items():
            if key in _FIT_RESERVED_KEYS:
                continue
            if isinstance(value, torch.Tensor):
                dropped_fields.append(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
                continue
            try:
                _validate_metadata({key: value})
                clean_meta[key] = value
            except ValueError:
                dropped_fields.append(key)

        clean_meta["converted_from"] = "jacobian_lens_checkpoint"
        if dropped_fields:
            clean_meta["dropped_fields"] = dropped_fields

        return cls(jacobians, n_prompts=n_prompts, d_model=d_model, metadata=clean_meta)

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        *,
        filename: str = "lens.pt",
        revision: Optional[str] = None,
        model: Any = None,
    ) -> "JacobianLens":
        """Load from a ``.pt`` file, ``<dir>/<filename>``, a registry short name or HF model id
        (``filename`` ignored), or a Hub repo; passing ``model`` runs :meth:`validate_model`."""
        import os

        if os.path.isfile(name_or_path):
            lens = cls.load(name_or_path)
        elif os.path.isdir(name_or_path):
            lens = cls.load(os.path.join(name_or_path, filename))
        else:
            from huggingface_hub import hf_hub_download

            if cls._uses_artifact_registry:
                resolved = _resolve_registry_entry(name_or_path)
            else:
                if "/" not in name_or_path:
                    raise ValueError(
                        f"{cls.__name__} does not resolve short names from the "
                        "Jacobian lens registry; pass an explicit path or a Hub "
                        f"repo id (owner/name) instead of {name_or_path!r}."
                    )
                resolved = None
            if resolved is not None:
                repo_id, resolved_filename = resolved
            else:
                repo_id, resolved_filename = name_or_path, filename

            local_path = call_hf_with_retry(
                hf_hub_download,
                repo_id=repo_id,
                filename=resolved_filename,
                revision=revision,
            )
            lens = cls.load(local_path)
        if model is not None:
            lens.validate_model(model)
        return lens

    @classmethod
    def merge(cls, lenses: Sequence["JacobianLens"]) -> "JacobianLens":
        """Average shards fitted on disjoint prompt slices, weighted by ``n_prompts``; all
        provenance other than ``n_prompts`` must match."""
        if not lenses:
            raise ValueError("cannot merge an empty sequence of lenses")
        invalid_counts = [
            (index, lens.n_prompts) for index, lens in enumerate(lenses) if lens.n_prompts <= 0
        ]
        if invalid_counts:
            raise ValueError(
                "every lens passed to merge() must have positive n_prompts; "
                f"invalid shards: {invalid_counts}"
            )
        first = lenses[0]
        for lens in lenses:
            _validate_metadata(lens.metadata)
        cls._validate_merge_inputs(lenses)
        first_provenance = {
            key: value for key, value in first.metadata.items() if key != "n_prompts"
        }
        for other in lenses[1:]:
            if other.source_layers != first.source_layers or other.d_model != first.d_model:
                raise ValueError(
                    "all lenses being merged must share the same source_layers and d_model"
                )
            other_provenance = {
                key: value for key, value in other.metadata.items() if key != "n_prompts"
            }
            if other_provenance != first_provenance:
                raise ValueError(
                    "all lenses being merged must share the same provenance metadata "
                    "apart from n_prompts"
                )
        total = sum(lens.n_prompts for lens in lenses)
        merged = {
            layer: torch.stack([lens.jacobians[layer] * lens.n_prompts for lens in lenses]).sum(
                dim=0
            )
            / total
            for layer in first.source_layers
        }
        metadata = dict(first.metadata)
        if metadata:
            metadata["n_prompts"] = total
        return cls(merged, n_prompts=total, d_model=first.d_model, metadata=metadata)

    def validate_model(self, model: Any) -> "JacobianLens":
        """Raise unless ``model`` is a raw causal Bridge matching this lens's provenance,
        ``d_model``, layer range and final-layer target; returns ``self``."""
        _require_raw_bridge(model, estimator=type(self).__name__)
        artifact_model_name = self.metadata.get("model_name")
        current_model_name = getattr(model.cfg, "model_name", None)
        if artifact_model_name is not None and artifact_model_name != current_model_name:
            raise ValueError(
                f"lens was fitted for model {artifact_model_name!r}, but the supplied "
                f"model is {current_model_name!r}."
            )
        artifact_revision = self.metadata.get("model_revision")
        current_revision = _get_model_revision(model)
        if artifact_revision is not None and artifact_revision != current_revision:
            raise ValueError(
                f"lens was fitted for model revision {artifact_revision!r}, but the "
                f"supplied model revision is {current_revision!r}."
            )
        d_model = model.cfg.d_model
        if d_model != self.d_model:
            raise ValueError(
                f"lens was fitted for d_model={self.d_model}, but the model has "
                f"d_model={d_model} — this lens belongs to a different model."
            )
        n_layers = model.cfg.n_layers
        final_layer = n_layers - 1
        out_of_range = [layer for layer in self.source_layers if not 0 <= layer < final_layer]
        if out_of_range:
            raise ValueError(
                f"lens has source layers {out_of_range} outside the model's "
                f"0..{final_layer - 1} source range — this lens belongs to a different model."
            )
        target_layer = int(self.metadata.get("target_layer", final_layer))
        if target_layer != final_layer:
            raise ValueError(
                f"lens targets layer {target_layer}, but readout supports only the "
                f"published final-layer convention ({final_layer}); refit without "
                "a custom target layer."
            )
        return self

    def clear_device_cache(self) -> None:
        """Release cached Jacobians, dictionaries, and unembedding snapshots on devices."""
        self._device_jacobians.clear()
        self._dictionary_cache.clear()
        self._unembedding_snapshots.clear()

    def _matrix_on(self, layer: int, device: Union[str, torch.device]) -> torch.Tensor:
        """Return one cached fp32 Jacobian copy for a layer/device pair."""
        if layer not in self.jacobians:
            raise ValueError(
                f"layer {layer} is not in this lens's source layers "
                f"({self.source_layers[0]}..{self.source_layers[-1]})"
            )
        resolved_device = torch.device(device)
        key = (layer, resolved_device)
        matrix = self._device_jacobians.get(key)
        if matrix is None:
            matrix = self.jacobians[layer].to(device=resolved_device, dtype=torch.float32)
            self._device_jacobians[key] = matrix
        return matrix

    def transport(
        self,
        residual: Float[torch.Tensor, "... d_model"],
        layer: int,
    ) -> Float[torch.Tensor, "... d_model"]:
        """Map block-``layer`` output to the final block's output basis: fp32 ``J[layer] @ h``."""
        matrix = self._matrix_on(layer, residual.device)
        return residual.float() @ matrix.T

    @torch.no_grad()
    def readout(
        self,
        model: Any,
        input: Union[str, Int[torch.Tensor, "batch seq"]],
        *,
        layers: Optional[Sequence[int]] = None,
        positions: Optional[Sequence[int]] = None,
        use_jacobian: bool = True,
        top_k: int = DEFAULT_TOP_K,
        return_full_logits: bool = False,
    ) -> JacobianLensReadout:
        """Per-layer top-k logits via ``J[layer]`` (identity if ``use_jacobian=False``) and the
        model's final norm and unembed; full logits are opt-in (~1.7 GB for 64 Gemma tokens)."""
        self.validate_model(model)
        tokens = model.to_tokens(input) if isinstance(input, str) else input
        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError(f"readout expects a single prompt; got shape {tuple(tokens.shape)}")
        n_layers = model.cfg.n_layers
        final_layer = n_layers - 1
        if layers is None:
            layers = self.source_layers + [final_layer]
        layers = [_normalize_layer(layer, n_layers) for layer in layers]
        for layer in layers:
            if use_jacobian and layer != final_layer and layer not in self.jacobians:
                raise ValueError(
                    f"layer {layer} is not in this lens's source layers; "
                    f"available: {self.source_layers} (+{final_layer} as identity)"
                )

        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")
        seq_len = tokens.shape[1]
        norm_positions = _normalize_positions(positions, seq_len)

        hook_names = {
            layer: _resid_post_hook_name(layer) for layer in layers if layer != final_layer
        }
        wanted = set(hook_names.values())
        logits, cache = model.run_with_cache(tokens, names_filter=lambda name: name in wanted)
        selected_model_logits = logits[0, norm_positions, :].float()
        if top_k > selected_model_logits.shape[-1]:
            raise ValueError(
                f"top_k={top_k} exceeds the model vocabulary size "
                f"{selected_model_logits.shape[-1]}"
            )
        model_topk = selected_model_logits.topk(top_k, dim=-1)
        full_model_logits = selected_model_logits.cpu() if return_full_logits else None
        lens_topk_values: Dict[int, torch.Tensor] = {}
        lens_topk_indices: Dict[int, torch.Tensor] = {}
        full_lens_logits: Optional[Dict[int, torch.Tensor]] = {} if return_full_logits else None
        for layer in layers:
            if layer == final_layer:
                layer_logits = selected_model_logits
                layer_topk = model_topk
            else:
                activation = cache[hook_names[layer]]
                _validate_residual_activation(
                    activation,
                    d_model=model.cfg.d_model,
                    hook_name=hook_names[layer],
                )
                residual = activation[0, norm_positions, :]
                transported = self.transport(residual, layer) if use_jacobian else residual.float()
                layer_logits = _unembed(model, transported)
                layer_topk = layer_logits.topk(top_k, dim=-1)
            lens_topk_values[layer] = layer_topk.values.cpu()
            lens_topk_indices[layer] = layer_topk.indices.cpu()
            if full_lens_logits is not None:
                if layer == final_layer:
                    assert full_model_logits is not None
                    full_lens_logits[layer] = full_model_logits
                else:
                    full_lens_logits[layer] = layer_logits.cpu()
        return JacobianLensReadout(
            lens_topk_values=lens_topk_values,
            lens_topk_indices=lens_topk_indices,
            model_topk_values=model_topk.values.cpu(),
            model_topk_indices=model_topk.indices.cpu(),
            tokens=tokens[0].cpu(),
            positions=norm_positions,
            use_jacobian=use_jacobian,
            lens_logits=full_lens_logits,
            model_logits=full_model_logits,
        )

    @torch.no_grad()
    def lens_vectors(
        self,
        model: Any,
        tokens: Union[TokenInput, Sequence[TokenInput]],
        layer: int,
    ) -> Float[torch.Tensor, "n d_model"]:
        """J-lens vectors ``v_t = J[layer]^T W_U[:, t]``, fp32 on the model's device."""
        self.validate_model(model)
        layer = _normalize_layer(layer, model.cfg.n_layers)
        token_ids = _to_token_ids(model, tokens)
        unembed_columns = model.W_U[:, token_ids].float()
        matrix = self._matrix_on(layer, unembed_columns.device)
        return (matrix.T @ unembed_columns).T

    @torch.no_grad()
    def lens_vector_dictionary(
        self, model: Any, layer: int
    ) -> Float[torch.Tensor, "d_vocab d_model"]:
        """``[d_vocab, d_model]`` J-lens vectors at ``layer``, cached on device (GB-scale for large
        vocabularies) until ``W_U`` changes; :meth:`clear_device_cache` releases it."""
        self.validate_model(model)
        layer = _normalize_layer(layer, model.cfg.n_layers)
        unembed = model.W_U
        device = torch.device(unembed.device)
        # A device-resident W_U copy detects edits without a host transfer of the weights.
        snapshot = self._unembedding_snapshots.get(device)
        if snapshot is None or not torch.equal(snapshot, unembed):
            self._unembedding_snapshots[device] = unembed.detach().clone()
            stale_keys = [key for key in self._dictionary_cache if key[1] == device]
            for key in stale_keys:
                del self._dictionary_cache[key]

        key = (layer, device)
        dictionary = self._dictionary_cache.get(key)
        if dictionary is None:
            matrix = self._matrix_on(layer, device)
            dictionary = (matrix.T @ unembed.float()).T  # [d_vocab, d_model]
            self._dictionary_cache[key] = dictionary
        return dictionary

    @torch.no_grad()
    def decompose(
        self,
        model: Any,
        activation_or_prompt: Union[torch.Tensor, str],
        layer: int,
        *,
        position: Optional[int] = None,
        k: int = DEFAULT_K,
        algorithm: str = "nonnegative_orthogonal_matching_pursuit",
    ) -> JSpaceDecomposition:
        """Sparse J-space decomposition of a ``[d_model]`` activation or a prompt's at ``position``.
        ``support`` is on CPU; vector-valued outputs stay on the model's device."""
        activation, dictionary = self._resolve_activation(
            model, activation_or_prompt, layer, position
        )
        return get_sparse_decomposition(activation, dictionary, k, algorithm=algorithm)

    @torch.no_grad()
    def coordinate_patch(
        self,
        model: Any,
        activation_or_prompt: Union[torch.Tensor, str],
        layer: int,
        source_token: TokenInput,
        target_token: TokenInput,
        *,
        position: Optional[int] = None,
        decomposition: Optional[JSpaceDecomposition] = None,
        k: int = DEFAULT_K,
        mode: str = "substitute",
        alpha: float = 1.0,
        algorithm: str = "nonnegative_orthogonal_matching_pursuit",
    ) -> CoordinatePatch:
        """Patch an activation's J-space frame: ``substitute`` moves ``source_token``'s coordinate
        onto ``target_token``, ``swap`` exchanges them; the rest and the residual stay fixed."""
        activation, dictionary = self._resolve_activation(
            model, activation_or_prompt, layer, position
        )
        source_id, target_id = _to_token_ids(model, [source_token, target_token])
        return solve_coordinate_patch(
            activation,
            dictionary,
            source_id,
            target_id,
            decomposition=decomposition,
            k=k,
            mode=mode,
            alpha=alpha,
            algorithm=algorithm,
        )

    def _resolve_activation(
        self,
        model: Any,
        activation_or_prompt: Union[torch.Tensor, str],
        layer: int,
        position: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resolve a raw ``[d_model]`` activation or a prompt at ``position`` to
        ``(fp32 activation, dictionary)`` on the dictionary's device."""
        self.validate_model(model)
        resolved_layer = _normalize_layer(layer, model.cfg.n_layers)
        if resolved_layer not in self.jacobians:
            raise ValueError(
                f"layer {layer} is not in this lens's source layers; "
                f"available: {self.source_layers}"
            )

        if position is None:
            if not isinstance(activation_or_prompt, torch.Tensor):
                raise ValueError(
                    "analysis expects a raw activation tensor when position is None; pass a "
                    "prompt together with a position to analyze a model activation"
                )
            activation = activation_or_prompt
            if activation.ndim != 1 or activation.shape[0] != self.d_model:
                raise ValueError(
                    f"activation must be 1-D of length d_model={self.d_model}, "
                    f"got shape {tuple(activation.shape)}"
                )
        else:
            if isinstance(activation_or_prompt, torch.Tensor) and activation_or_prompt.ndim == 1:
                raise ValueError(
                    "position is only valid with a prompt (string or [1, seq] tokens); "
                    "pass a raw 1-D activation with position=None instead"
                )
            tokens = (
                model.to_tokens(activation_or_prompt)
                if isinstance(activation_or_prompt, str)
                else activation_or_prompt
            )
            if tokens.ndim != 2 or tokens.shape[0] != 1:
                raise ValueError(
                    f"analysis expects a single prompt; got shape {tuple(tokens.shape)}"
                )
            hook_name = _resid_post_hook_name(resolved_layer)
            _, cache = model.run_with_cache(tokens, names_filter=lambda name: name == hook_name)
            norm_position = _normalize_positions([position], tokens.shape[1])[0]
            activation = cache[hook_name][0, norm_position, :]
        dictionary = self.lens_vector_dictionary(model, resolved_layer)
        return activation.float().to(dictionary.device), dictionary

    @torch.no_grad()
    def occupancy(
        self,
        model: Any,
        activation_or_prompt: Union[torch.Tensor, str],
        layer: int,
        *,
        position: Optional[int] = None,
        max_atoms: int = DEFAULT_K,
        num_control_dictionaries: int = 32,
        seed: int = 0,
    ) -> JSpaceOccupancy:
        """Estimate how many J-lens vectors are meaningfully active in an activation."""
        activation, dictionary = self._resolve_activation(
            model, activation_or_prompt, layer, position
        )
        return estimate_occupancy(
            activation,
            dictionary,
            max_atoms=max_atoms,
            num_control_dictionaries=num_control_dictionaries,
            seed=seed,
        )

    @torch.no_grad()
    def fraction_of_variance(
        self,
        model: Any,
        prompts: Union[str, torch.Tensor, Sequence[Union[str, torch.Tensor]]],
        layers: Optional[Sequence[int]] = None,
        *,
        k: int = DEFAULT_K,
        skip_first: int = 16,
        positions: Optional[Sequence[int]] = None,
        show_progress: bool = False,
    ) -> JSpaceVarianceProfile:
        """Per-layer median and pooled ``||j_space_component||^2 / ||activation||^2`` over prompts.
        A layer that samples no position reports NaN and an empty ``per_position``."""
        self.validate_model(model)
        if skip_first < 0:
            raise ValueError(f"skip_first must be non-negative, got {skip_first}")
        if layers is None:
            resolved_layers = list(self.source_layers)
        else:
            resolved_layers = [_normalize_layer(layer, model.cfg.n_layers) for layer in layers]
            for layer in resolved_layers:
                if layer not in self.jacobians:
                    raise ValueError(
                        f"layer {layer} is not in this lens's source layers; "
                        f"available: {self.source_layers}"
                    )
        prompt_list: List[Union[str, torch.Tensor]] = (
            [prompts] if isinstance(prompts, (str, torch.Tensor)) else list(prompts)
        )
        if not prompt_list:
            raise ValueError("prompts must be a non-empty prompt or sequence of prompts")

        hook_names = {layer: _resid_post_hook_name(layer) for layer in resolved_layers}
        wanted_hooks = set(hook_names.values())
        dictionaries = {
            layer: self.lens_vector_dictionary(model, layer) for layer in resolved_layers
        }
        fractions: Dict[int, List[float]] = {layer: [] for layer in resolved_layers}
        pooled_j_space: Dict[int, float] = {layer: 0.0 for layer in resolved_layers}
        pooled_total: Dict[int, float] = {layer: 0.0 for layer in resolved_layers}

        for prompt in tqdm(prompt_list, desc="J-space variance", disable=not show_progress):
            tokens = model.to_tokens(prompt) if isinstance(prompt, str) else prompt
            if tokens.ndim != 2 or tokens.shape[0] != 1:
                raise ValueError(
                    "fraction_of_variance expects each tokenized prompt to have shape "
                    f"[1, seq], got {tuple(tokens.shape)}"
                )
            _, cache = model.run_with_cache(tokens, names_filter=lambda name: name in wanted_hooks)
            seq_len = tokens.shape[1]
            sampled = (
                list(range(skip_first, seq_len))
                if positions is None
                else _normalize_positions(positions, seq_len)
            )
            for layer in resolved_layers:
                dictionary = dictionaries[layer]
                activations = cache[hook_names[layer]][0]
                for position in sampled:
                    activation = activations[position].float().to(dictionary.device)
                    total = float(activation @ activation)
                    if total <= 0.0:
                        continue
                    decomposition = get_sparse_decomposition(activation, dictionary, k)
                    # The paper's J-space component is the span projection, not the reconstruction.
                    j_space = float(
                        decomposition.j_space_component @ decomposition.j_space_component
                    )
                    fractions[layer].append(j_space / total)
                    pooled_j_space[layer] += j_space
                    pooled_total[layer] += total

        median = {
            layer: float(torch.tensor(fractions[layer]).median())
            if fractions[layer]
            else float("nan")
            for layer in resolved_layers
        }
        pooled = {
            layer: pooled_j_space[layer] / pooled_total[layer]
            if pooled_total[layer] > 0
            else float("nan")
            for layer in resolved_layers
        }
        per_position = {layer: torch.tensor(fractions[layer]) for layer in resolved_layers}
        return JSpaceVarianceProfile(
            layers=resolved_layers, median=median, pooled=pooled, per_position=per_position
        )

    def steering_hooks(
        self,
        model: Any,
        token: TokenInput,
        layers: Sequence[int],
        *,
        alpha: float = 4.0,
        positions: Optional[Sequence[int]] = None,
    ) -> List[Tuple[str, Any]]:
        """Hooks adding ``alpha * median||h|| * v̂`` (reference norm-matched steering; the median
        ignores attention-sink norms); the paper's raw ``h + alpha * v_t`` needs a custom hook."""
        self.validate_model(model)
        hooks = []
        for layer in [_normalize_layer(layer, model.cfg.n_layers) for layer in layers]:
            direction = self.lens_vectors(model, token, layer)[0]
            unit = _unit_rows(direction.unsqueeze(0), layer=layer)[0]
            device_units: Dict[torch.device, torch.Tensor] = {}

            def transform(
                selected: Float[torch.Tensor, "batch pos d_model"],
                unit: torch.Tensor = unit,
                device_units: Dict[torch.device, torch.Tensor] = device_units,
            ) -> Float[torch.Tensor, "batch pos d_model"]:
                local_unit = _cached_on_device(unit, device_units, selected.device)
                scale = alpha * selected.float().norm(dim=-1).median()
                return selected.float() + scale * local_unit

            hooks.append(
                (
                    _resid_post_hook_name(layer),
                    _make_intervention_hook(transform, positions, model.cfg.d_model),
                )
            )
        return hooks

    def ablation_hooks(
        self,
        model: Any,
        tokens: Union[TokenInput, Sequence[TokenInput]],
        layers: Sequence[int],
        *,
        positions: Optional[Sequence[int]] = None,
    ) -> List[Tuple[str, Any]]:
        """Hooks projecting each token's unit lens vector out of ``h``, sequentially."""
        self.validate_model(model)
        hooks = []
        for layer in [_normalize_layer(layer, model.cfg.n_layers) for layer in layers]:
            vectors = self.lens_vectors(model, tokens, layer)
            units = _unit_rows(vectors, layer=layer)
            device_units: Dict[torch.device, torch.Tensor] = {}

            def transform(
                selected: Float[torch.Tensor, "batch pos d_model"],
                units: torch.Tensor = units,
                device_units: Dict[torch.device, torch.Tensor] = device_units,
            ) -> Float[torch.Tensor, "batch pos d_model"]:
                local_units = _cached_on_device(units, device_units, selected.device)
                result = selected.float()
                for unit in local_units:
                    coeff = result @ unit
                    result = result - coeff.unsqueeze(-1) * unit
                return result

            hooks.append(
                (
                    _resid_post_hook_name(layer),
                    _make_intervention_hook(transform, positions, model.cfg.d_model),
                )
            )
        return hooks

    def swap_hooks(
        self,
        model: Any,
        source_token: TokenInput,
        target_token: TokenInput,
        layers: Sequence[int],
        *,
        alpha: float = 1.0,
        positions: Optional[Sequence[int]] = None,
    ) -> List[Tuple[str, Any]]:
        """Hooks applying ``h + alpha * V (swap(V⁺h) - V⁺h)``, ``V = [v_s, v_t]``; re-read per hook,
        so later layers can undo the swap (the paper's clamp is :meth:`swap_clamp_hooks`)."""
        self.validate_model(model)
        source_id, target_id = _to_token_ids(model, [source_token, target_token])
        if source_id == target_id:
            raise ValueError(
                "source_token and target_token resolve to the same token id; "
                "a coordinate swap would be a silent no-op"
            )

        hooks = []
        for layer in [_normalize_layer(layer, model.cfg.n_layers) for layer in layers]:
            vectors = self.lens_vectors(model, [source_id, target_id], layer)
            units = _unit_rows(vectors, layer=layer)
            _diagnose_intervention_pair(
                units, description=f"swap vectors at layer {layer}", stacklevel=3
            )
            basis = vectors.T  # [d, 2]
            pinv = torch.linalg.pinv(basis)  # [2, d]
            device_basis: Dict[torch.device, torch.Tensor] = {}
            device_pinv: Dict[torch.device, torch.Tensor] = {}

            def transform(
                selected: Float[torch.Tensor, "batch pos d_model"],
                basis: torch.Tensor = basis,
                pinv: torch.Tensor = pinv,
                device_basis: Dict[torch.device, torch.Tensor] = device_basis,
                device_pinv: Dict[torch.device, torch.Tensor] = device_pinv,
            ) -> Float[torch.Tensor, "batch pos d_model"]:
                local_basis = _cached_on_device(basis, device_basis, selected.device)
                local_pinv = _cached_on_device(pinv, device_pinv, selected.device)
                coords = selected.float() @ local_pinv.T  # [..., 2]
                delta = alpha * ((coords[..., [1, 0]] - coords) @ local_basis.T)
                return selected.float() + delta

            hooks.append(
                (
                    _resid_post_hook_name(layer),
                    _make_intervention_hook(transform, positions, model.cfg.d_model),
                )
            )
        return hooks

    def swap_clamp_hooks(
        self,
        model: Any,
        source_token: TokenInput,
        target_token: TokenInput,
        layers: Sequence[int],
        clean_cache: Union[ActivationCache, Mapping[str, torch.Tensor]],
        *,
        positions: Optional[Sequence[int]] = None,
    ) -> List[Tuple[str, Any]]:
        """Hooks clamping ``h + V (c_target - V⁺h)`` to the clean run's exchanged coordinates;
        ``clean_cache`` must hold ``blocks.{layer}.hook_out`` for every layer."""
        self.validate_model(model)
        source_id, target_id = _to_token_ids(model, [source_token, target_token])
        if source_id == target_id:
            raise ValueError(
                "source_token and target_token resolve to the same token id; "
                "a coordinate clamp would be a silent no-op"
            )

        hooks = []
        for layer in [_normalize_layer(layer, model.cfg.n_layers) for layer in layers]:
            hook_name = _resid_post_hook_name(layer)
            if hook_name not in clean_cache:
                raise ValueError(f"clean_cache is missing activation {hook_name!r}")
            clean = clean_cache[hook_name]
            _validate_residual_activation(
                clean, d_model=model.cfg.d_model, hook_name=f"clean_cache[{hook_name!r}]"
            )
            normalized = _normalize_positions(positions, clean.shape[1])
            clean_selected = clean if positions is None else clean[:, normalized, :]

            vectors = self.lens_vectors(model, [source_id, target_id], layer)
            units = _unit_rows(vectors, layer=layer)
            _diagnose_intervention_pair(
                units, description=f"swap vectors at layer {layer}", stacklevel=3
            )
            basis = vectors.T  # [d, 2]
            pinv = torch.linalg.pinv(basis)  # [2, d]
            target_coords = (clean_selected.float() @ pinv.T)[..., [1, 0]]
            device_basis: Dict[torch.device, torch.Tensor] = {}
            device_pinv: Dict[torch.device, torch.Tensor] = {}
            device_targets: Dict[torch.device, torch.Tensor] = {}

            def transform(
                selected: Float[torch.Tensor, "batch pos d_model"],
                basis: torch.Tensor = basis,
                pinv: torch.Tensor = pinv,
                target_coords: torch.Tensor = target_coords,
                device_basis: Dict[torch.device, torch.Tensor] = device_basis,
                device_pinv: Dict[torch.device, torch.Tensor] = device_pinv,
                device_targets: Dict[torch.device, torch.Tensor] = device_targets,
            ) -> Float[torch.Tensor, "batch pos d_model"]:
                local_basis = _cached_on_device(basis, device_basis, selected.device)
                local_pinv = _cached_on_device(pinv, device_pinv, selected.device)
                local_targets = _cached_on_device(target_coords, device_targets, selected.device)
                if local_targets.shape[1] != selected.shape[1] or local_targets.shape[0] not in (
                    1,
                    selected.shape[0],
                ):
                    raise ValueError(
                        "clean_cache activation shape is incompatible with the live activation: "
                        f"target coordinates have shape {tuple(local_targets.shape)}, "
                        f"live activation has shape {tuple(selected.shape)}"
                    )
                coords = selected.float() @ local_pinv.T
                delta = (local_targets - coords) @ local_basis.T
                return selected.float() + delta

            hooks.append(
                (
                    hook_name,
                    _make_intervention_hook(transform, positions, model.cfg.d_model),
                )
            )
        return hooks

    def coordinate_patch_hooks(
        self,
        model: Any,
        source_token: TokenInput,
        target_token: TokenInput,
        layers: Sequence[int],
        *,
        positions: Sequence[int],
        decomposition_cache: Optional[
            MutableMapping[Tuple[int, int, int], JSpaceDecomposition]
        ] = None,
        k: int = DEFAULT_K,
        mode: str = "substitute",
        alpha: float = 1.0,
        algorithm: str = "nonnegative_orthogonal_matching_pursuit",
    ) -> List[Tuple[str, Any]]:
        """Live coordinate-patch hooks; ``decomposition_cache`` keys are ``(layer, batch, pos)``.
        Fails if ``source_token`` is inactive at hook time, e.g. after an earlier layer's patch."""
        self.validate_model(model)
        # No full-sequence default: every patched position costs a vocabulary-scale solve.
        if not positions:
            raise ValueError("positions must contain at least one index")
        resolved_layers = [_normalize_layer(layer, model.cfg.n_layers) for layer in layers]
        source_id, target_id = _to_token_ids(model, [source_token, target_token])
        if source_id == target_id:
            raise ValueError(
                "source_token and target_token resolve to the same token id; "
                "a coordinate patch would be a silent no-op"
            )
        warnings.warn(
            f"coordinate_patch_hooks installs {len(resolved_layers)} layer(s) x "
            f"{len(positions)} position(s) of coordinate-patch hooks; every (batch, position) "
            "pair not already present in decomposition_cache performs a vocabulary-scale sparse "
            "decomposition on every forward pass",
            UserWarning,
            stacklevel=2,
        )

        requested = tuple(positions)
        hooks = []
        for layer in resolved_layers:
            dictionary = self.lens_vector_dictionary(model, layer)

            def hook_fn(
                activation: Float[torch.Tensor, "batch pos d_model"],
                hook: Any,
                layer: int = layer,
                dictionary: torch.Tensor = dictionary,
            ) -> Float[torch.Tensor, "batch pos d_model"]:
                hook_name = getattr(hook, "name", "intervention hook")
                _validate_residual_activation(
                    activation, d_model=model.cfg.d_model, hook_name=hook_name
                )
                # Chunk-local positions: a KV-cached decode step has one, so normalize per firing.
                normalized = _normalize_positions(requested, activation.shape[1])
                selected = activation[:, normalized, :].float().to(dictionary.device)
                patched, _ = solve_coordinate_patch_positions(
                    selected,
                    dictionary,
                    normalized,
                    source_id,
                    target_id,
                    layer=layer,
                    decomposition_cache=decomposition_cache,
                    k=k,
                    mode=mode,
                    alpha=alpha,
                    algorithm=algorithm,
                )
                output = activation.clone()
                output[:, normalized, :] = patched.to(
                    device=activation.device, dtype=activation.dtype
                )
                return output

            hooks.append((_resid_post_hook_name(layer), hook_fn))
        return hooks

    @classmethod
    def fit(
        cls,
        model: Any,
        prompts: Sequence[str],
        *,
        corpus: str,
        source_layers: Optional[Sequence[int]] = None,
        dim_batch: int = 8,
        max_seq_len: int = 128,
        skip_first_positions: int = DEFAULT_SKIP_FIRST_POSITIONS,
        show_progress: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "JacobianLens":
        """Fit with the reference estimator: per prompt, one forward replicated ``dim_batch``
        times, then ``ceil(d_model / dim_batch)`` one-hot backwards; short prompts are skipped."""
        options = _prepare_fit(
            model,
            operation=cls.__name__,
            estimator=ESTIMATOR_JACOBIAN,
            corpus=corpus,
            source_layers=source_layers,
            dim_batch=dim_batch,
            max_seq_len=max_seq_len,
            skip_first_positions=skip_first_positions,
            metadata=metadata,
        )
        transport_matrices, n_done = _fit_transport_matrices(
            model,
            prompts,
            source_layers=list(options.source_layers),
            dim_batch=options.dim_batch,
            max_seq_len=options.max_seq_len,
            skip_first_positions=options.skip_first_positions,
            show_progress=show_progress,
            backward_provider=_ordinary_vjp,
        )
        return cls(
            transport_matrices,
            n_prompts=n_done,
            d_model=options.d_model,
            metadata=_build_fit_metadata(
                model, options, estimator=ESTIMATOR_JACOBIAN, n_prompts=n_done, metadata=metadata
            ),
        )


@dataclass(frozen=True)
class _FitOptions:
    """Resolved options shared by transport-matrix estimators."""

    source_layers: Tuple[int, ...]
    target_layer: int
    d_model: int
    fit_dtype: torch.dtype
    corpus: str
    dim_batch: int
    max_seq_len: int
    skip_first_positions: int


def _prepare_fit(
    model: Any,
    *,
    operation: str,
    estimator: str,
    corpus: str,
    source_layers: Optional[Sequence[int]],
    dim_batch: int,
    max_seq_len: int,
    skip_first_positions: int,
    metadata: Optional[Dict[str, Any]],
    estimator_metadata_keys: Sequence[str] = (),
) -> _FitOptions:
    """Validate the shared fit contract before tokenization or rule installation."""
    _require_raw_bridge(model, estimator=operation)
    require_eval_mode(model, operation=f"{operation}.fit()")
    if not isinstance(corpus, str) or not corpus.strip():
        raise ValueError("corpus must be a non-empty provenance identifier")
    n_layers = model.cfg.n_layers
    target_layer = n_layers - 1
    if source_layers is None:
        resolved_sources = list(range(target_layer))
    else:
        resolved_sources = sorted({_normalize_layer(layer, n_layers) for layer in source_layers})
    if not resolved_sources:
        raise ValueError("source_layers is empty")
    if resolved_sources[-1] >= target_layer:
        raise ValueError(
            f"every source layer must be below target_layer={target_layer}; got {resolved_sources}"
        )
    if dim_batch < 1:
        raise ValueError(f"dim_batch must be >= 1, got {dim_batch}")
    if skip_first_positions < 0:
        raise ValueError(f"skip_first_positions must be >= 0, got {skip_first_positions}")
    options = _FitOptions(
        source_layers=tuple(resolved_sources),
        target_layer=target_layer,
        d_model=model.cfg.d_model,
        fit_dtype=model.W_U.dtype,
        corpus=corpus,
        dim_batch=dim_batch,
        max_seq_len=max_seq_len,
        skip_first_positions=skip_first_positions,
    )
    # Fail on bad provenance before the expensive fit; real counts are filled in afterwards.
    _build_fit_metadata(
        model,
        options,
        estimator=estimator,
        n_prompts=0,
        metadata=metadata,
        estimator_metadata=dict.fromkeys(estimator_metadata_keys),
    )
    if options.fit_dtype in (torch.float16, torch.bfloat16):
        warnings.warn(
            f"fitting in {options.fit_dtype} accumulates transport gradients at reduced "
            "precision; use a float32 TransformerBridge for the highest-fidelity fit",
            UserWarning,
            stacklevel=3,
        )
    return options


def _build_fit_metadata(
    model: Any,
    options: _FitOptions,
    *,
    estimator: str,
    n_prompts: int,
    metadata: Optional[Dict[str, Any]],
    estimator_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Combine validated user provenance with common and estimator-specific fit fields."""
    fit_metadata: Dict[str, Any] = {
        "model_name": getattr(model.cfg, "model_name", None),
        "model_revision": _get_model_revision(model),
        "transformer_lens_version": version("transformer-lens"),
        "model_system": "TransformerBridge",
        "estimator": estimator,
        "processing": {"compatibility_mode": False, "weight_basis": "raw_huggingface"},
        "hook_convention": "blocks.{layer}.hook_out",
        "corpus": options.corpus,
        "n_prompts": n_prompts,
        "fit_dtype": str(options.fit_dtype).removeprefix("torch."),
        "target_layer": options.target_layer,
        "dim_batch": options.dim_batch,
        "max_seq_len": options.max_seq_len,
        "skip_first_positions": options.skip_first_positions,
        "transformer_lens_fit": True,
    }
    extra_metadata = dict(estimator_metadata or {})
    overlapping = sorted(set(fit_metadata).intersection(extra_metadata))
    if overlapping:
        raise ValueError(f"estimator metadata cannot override common fit provenance: {overlapping}")
    fit_metadata.update(extra_metadata)
    reserved = sorted(set(fit_metadata).intersection(metadata or {}))
    if reserved:
        raise ValueError(f"metadata cannot override fit provenance keys: {reserved}")
    full_metadata = dict(metadata or {})
    full_metadata.update(fit_metadata)
    _validate_metadata(full_metadata)
    return full_metadata


def _resid_post_hook_name(layer: int) -> str:
    """Bridge-native hook for the output of block ``layer``."""
    return f"blocks.{layer}.hook_out"


def _get_model_revision(model: Any) -> Optional[str]:
    """Return the resolved Hugging Face commit recorded on a booted model."""
    original_model = getattr(model, "original_model", None)
    hf_config = getattr(original_model, "config", None)
    revision = getattr(hf_config, "_commit_hash", None)
    return revision if isinstance(revision, str) and revision else None


def _require_raw_bridge(model: Any, *, estimator: str) -> None:
    """Require the causal raw-Bridge contract used by fit and readout."""
    from transformer_lens.model_bridge import TransformerBridge

    if not isinstance(model, TransformerBridge):
        raise TypeError(
            f"{estimator} supports TransformerBridge only; load a fresh model with "
            "TransformerBridge.boot_transformers(...)."
        )
    if getattr(model, "compatibility_mode", False):
        raise ValueError(
            "compatibility mode is enabled on this TransformerBridge and changes "
            "the residual basis. Use a freshly booted "
            "TransformerBridge.boot_transformers(...) model with raw weights."
        )
    if getattr(model, "_weights_processed", False):
        raise ValueError(
            "process_weights was called on this TransformerBridge and changed the "
            "raw HuggingFace weight basis. Use a freshly booted "
            "TransformerBridge.boot_transformers(...) model."
        )
    adapter = model.adapter
    if not adapter.supports_generation:
        raise ValueError(
            f"{estimator} requires a causal decoder-only Bridge whose adapter "
            f"supports text generation; {type(adapter).__name__} declares "
            "supports_generation=False."
        )
    adapter.validate_output_logits_transform()
    attention_dir = getattr(model.cfg, "attention_dir", "causal")
    if attention_dir != "causal":
        raise ValueError(
            f"{estimator} requires causal attention because its estimator relies on "
            "causality to exclude target positions before each source position; "
            f"got attention_dir={attention_dir!r}."
        )
    total_ut_steps = int(getattr(model.cfg, "total_ut_steps", 1) or 1)
    if total_ut_steps != 1:
        raise ValueError(
            f"{estimator} requires each physical block hook to fire once per forward; "
            f"this looped-depth Bridge runs total_ut_steps={total_ut_steps}."
        )
    component_mapping = adapter.get_component_mapping()
    required_components = ("blocks", "ln_final", "unembed")
    missing_components = [
        component
        for component in required_components
        if component not in component_mapping or not hasattr(model, component)
    ]
    if missing_components:
        raise ValueError(
            f"{estimator} requires the standard direct ln_final -> unembed output path; "
            f"this Bridge is missing {missing_components}."
        )
    blocks_component = component_mapping["blocks"]
    if not getattr(blocks_component, "hook_out_is_single_residual_stream", False):
        raise ValueError(
            f"{estimator} requires single-stream [batch, position, d_model] block "
            f"outputs; {type(blocks_component).__name__} does not provide that "
            "contract."
        )
    if "project_out" in component_mapping:
        raise ValueError(
            f"{estimator} does not yet support a final output projection between "
            "the residual stream and unembedding."
        )
    unembed_width = model.W_U.shape[0]
    if unembed_width != model.cfg.d_model:
        raise ValueError(
            f"{estimator} requires a direct d_model-width unembedding after ln_final; "
            f"got W_U input width {unembed_width} for d_model={model.cfg.d_model}. "
            "Architectures with a final output projection are not yet supported."
        )


def _harvest_artifact_provenance(path: str, payload: Dict[str, Any]) -> Any:
    """Copy a workspace-lenses provenance ``target_layer``; refuse non-Jacobian estimators."""
    metadata = payload.get("metadata")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        return metadata
    estimator = None
    config_json = provenance.get("config_json")
    if isinstance(config_json, str):
        try:
            estimator = json.loads(config_json).get("estimator")
        except (ValueError, AttributeError):
            pass
    if estimator is not None and not (
        isinstance(estimator, str) and estimator in _JACOBIAN_ESTIMATOR_NAMES
    ):
        raise ValueError(
            f"{path} records estimator {estimator!r} in its provenance; only ordinary "
            "Jacobian lenses load from this format."
        )
    harvested: Dict[str, Any] = dict(metadata) if isinstance(metadata, dict) else {}
    if "target_layer" in provenance:
        harvested.setdefault("target_layer", provenance["target_layer"])
    return harvested


def _validate_metadata(metadata: Dict[str, Any]) -> None:
    """Reject values that ``torch.load(weights_only=True)`` cannot reload."""

    def validate(value: Any, path: str) -> None:
        if value is None or type(value) in (bool, int, float, str):
            return
        if type(value) in (list, tuple):
            for index, item in enumerate(value):
                validate(item, f"{path}[{index}]")
            return
        if type(value) is dict:
            for key, item in value.items():
                if type(key) is not str:
                    raise ValueError(
                        f"{path} has non-string key {key!r}; metadata keys must be strings"
                    )
                validate(item, f"{path}.{key}")
            return
        raise ValueError(
            f"{path} has unsupported type {type(value).__name__}; use only "
            "None, bool, int, float, str, lists, tuples, and string-keyed dicts"
        )

    validate(metadata, "metadata")


def _normalize_positions(positions: Optional[Sequence[int]], seq_len: int) -> List[int]:
    """Normalize negative chunk-local positions and raise before indexing."""
    if positions is None:
        return list(range(seq_len))
    normalized = [position + seq_len if position < 0 else position for position in positions]
    out_of_range = [position for position in normalized if not 0 <= position < seq_len]
    if out_of_range:
        raise ValueError(
            f"positions {out_of_range} out of range for an activation chunk of length {seq_len}"
        )
    return normalized


def _cached_on_device(
    tensor: torch.Tensor,
    cache: Dict[torch.device, torch.Tensor],
    device: Union[str, torch.device],
) -> torch.Tensor:
    """Cache a small fp32 intervention tensor on each activation device."""
    resolved_device = torch.device(device)
    local = cache.get(resolved_device)
    if local is None:
        local = tensor.to(device=resolved_device, dtype=torch.float32)
        cache[resolved_device] = local
    return local


def _unit_rows(vectors: torch.Tensor, *, layer: int) -> torch.Tensor:
    """Normalize intervention vectors and reject zero/non-finite directions."""
    vectors = vectors.float()
    norms = vectors.norm(dim=-1, keepdim=True)
    if (~torch.isfinite(norms) | (norms <= torch.finfo(torch.float32).eps)).any():
        raise ValueError(f"lens vectors at layer {layer} contain a zero or non-finite direction")
    return vectors / norms


def _make_intervention_hook(
    transform: Callable[[torch.Tensor], torch.Tensor],
    positions: Optional[Sequence[int]],
    d_model: int,
) -> Callable[..., torch.Tensor]:
    """Apply a transform with shared position, dtype, and device hardening."""
    requested = None if positions is None else tuple(positions)
    if requested == ():
        raise ValueError("positions must contain at least one index")

    def hook_fn(
        activation: Float[torch.Tensor, "batch pos d_model"], hook: Any
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        hook_name = getattr(hook, "name", "intervention hook")
        _validate_residual_activation(activation, d_model=d_model, hook_name=hook_name)
        normalized = _normalize_positions(requested, activation.shape[1])
        selected = activation if requested is None else activation[:, normalized, :]
        transformed = transform(selected)
        if transformed.shape != selected.shape:
            raise ValueError(
                f"intervention returned shape {tuple(transformed.shape)}, "
                f"expected {tuple(selected.shape)}"
            )
        transformed = transformed.to(device=activation.device, dtype=activation.dtype)
        if requested is None:
            return transformed
        output = activation.clone()
        output[:, normalized, :] = transformed
        return output

    return hook_fn


def _unembed(
    model: Any, residual: Float[torch.Tensor, "pos d_model"]
) -> Float[torch.Tensor, "pos d_vocab"]:
    """Apply the model's final norm, unembedding, logit scale and soft cap on the unembed's
    device; the components contract on ``[batch, pos, d_model]``, hence the singleton batch axis."""
    unembed_weight = model.W_U
    compute_dtype = unembed_weight.dtype
    batched = residual.to(device=unembed_weight.device, dtype=compute_dtype).unsqueeze(0)
    logits = model.unembed(model.ln_final(batched)).squeeze(0)
    return model.adapter.apply_output_logits_transform(logits).float()


def _to_token_ids(model: Any, tokens: Union[TokenInput, Sequence[TokenInput]]) -> List[int]:
    """Convert token strings / ids into a list of single-token ids."""
    if isinstance(tokens, (str, int)):
        tokens = [tokens]
    ids: List[int] = []
    for token in tokens:
        if isinstance(token, str):
            ids.append(model.to_single_token(token))
        elif isinstance(token, bool):
            raise ValueError(f"token {token!r} must be a string or integer token id, not bool")
        else:
            ids.append(int(token))
    if not ids:
        raise ValueError("tokens must contain at least one token")
    d_vocab = model.W_U.shape[1]
    invalid = [token_id for token_id in ids if not 0 <= token_id < d_vocab]
    if invalid:
        raise ValueError(f"token ids {invalid} out of range for vocabulary size {d_vocab}")
    return ids


def _validate_residual_activation(
    activation: torch.Tensor,
    *,
    d_model: int,
    hook_name: str,
) -> None:
    """Fail before interpreting a non-standard block output as a residual stream."""
    if activation.ndim != 3 or activation.shape[-1] != d_model:
        raise ValueError(
            f"{hook_name} must have shape [batch, position, {d_model}], "
            f"got {tuple(activation.shape)}"
        )


def _normalize_layer(layer: int, n_layers: int) -> int:
    """Resolve negative layer indices and bounds-check."""
    resolved = layer + n_layers if layer < 0 else layer
    if not 0 <= resolved < n_layers:
        raise ValueError(f"layer {layer} out of range for a {n_layers}-layer model")
    return resolved


class _frozen_parameters:
    """Freeze parameters (restoring flags on exit) so the fit graph roots at the residual stream."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.saved: List[Tuple[torch.nn.Parameter, bool]] = []

    def __enter__(self) -> None:
        self.saved = [(param, param.requires_grad) for param in self.model.parameters()]
        for param, _ in self.saved:
            param.requires_grad_(False)

    def __exit__(self, *exc: Any) -> None:
        for param, flag in self.saved:
            param.requires_grad_(flag)


def _ordinary_vjp(
    target: torch.Tensor,
    sources: List[torch.Tensor],
    cotangent: torch.Tensor,
    retain_graph: bool,
) -> Tuple[torch.Tensor, ...]:
    """Ordinary vector-Jacobian product: the J-lens :data:`BackwardProvider`."""
    return torch.autograd.grad(
        outputs=target,
        inputs=sources,
        grad_outputs=cotangent,
        retain_graph=retain_graph,
    )


def _jacobian_for_prompt(
    model: Any,
    tokens: Int[torch.Tensor, "one seq"],
    *,
    source_layers: List[int],
    dim_batch: int,
    skip_first_positions: int,
    backward_provider: BackwardProvider,
) -> Dict[int, Float[torch.Tensor, "d_model d_model"]]:
    """Per-prompt Jacobian rows via batched one-hot cotangents; expects frozen parameters so
    the graph roots at the earliest source activation."""
    d_model = model.cfg.d_model
    target_layer = model.cfg.n_layers - 1
    seq_len = tokens.shape[1]
    valid_positions = list(range(skip_first_positions, seq_len - 1))
    replicated = tokens.expand(dim_batch, -1)

    captured: Dict[str, torch.Tensor] = {}
    root_name = _resid_post_hook_name(min(source_layers))
    hook_layers = sorted(set(source_layers) | {target_layer})

    def capture_fn(
        activation: Float[torch.Tensor, "batch pos d_model"], hook: Any
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        _validate_residual_activation(activation, d_model=d_model, hook_name=hook.name)
        if hook.name == root_name and not activation.requires_grad:
            activation.requires_grad_(True)
        captured[hook.name] = activation
        return activation

    fwd_hooks = [(_resid_post_hook_name(layer), capture_fn) for layer in hook_layers]
    with torch.enable_grad(), model.hooks(fwd_hooks=fwd_hooks):
        model(replicated, return_type=None)

    target = captured[_resid_post_hook_name(target_layer)]
    sources = [captured[_resid_post_hook_name(layer)] for layer in source_layers]
    device = target.device
    positions_index = torch.tensor(valid_positions, device=device)
    batch_index = torch.arange(dim_batch, device=device)

    jacobians = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32) for layer in source_layers
    }
    cotangent = torch.zeros_like(target)
    n_passes = -(-d_model // dim_batch)
    for pass_index in range(n_passes):
        dim_start = pass_index * dim_batch
        n_dims = min(dim_batch, d_model - dim_start)
        cotangent.zero_()
        # One-hot at every valid target at once: causality makes each source grad sum over t' >= t.
        cotangent[
            batch_index[:n_dims, None],
            positions_index[None, :],
            dim_start + batch_index[:n_dims, None],
        ] = 1.0
        grads = backward_provider(
            target,
            sources,
            cotangent,
            pass_index < n_passes - 1,
        )
        for layer, grad in zip(source_layers, grads):
            # each gradient lives on its layer's device under sharded/device_map setups
            rows = grad[:n_dims, positions_index.to(grad.device), :].float().mean(dim=1)
            jacobians[layer][dim_start : dim_start + n_dims, :] = rows.cpu()
        del grads
    return jacobians


def _fit_transport_matrices(
    model: Any,
    prompts: Sequence[str],
    *,
    source_layers: List[int],
    dim_batch: int,
    max_seq_len: int,
    skip_first_positions: int,
    show_progress: bool,
    backward_provider: BackwardProvider,
) -> Tuple[Dict[int, Float[torch.Tensor, "d_model d_model"]], int]:
    """Shared fit loop returning ``(per-layer prompt-mean matrices, n_prompts)``; estimators vary
    only ``backward_provider``. Prompts without a valid position are skipped with a warning."""
    d_model = model.cfg.d_model
    jacobian_sum = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32) for layer in source_layers
    }
    n_done = 0
    iterator = tqdm(prompts, desc="fitting J-lens", disable=not show_progress)
    with _frozen_parameters(model):
        for prompt in iterator:
            tokens = model.to_tokens(prompt)[:, :max_seq_len]
            seq_len = tokens.shape[1]
            if seq_len <= skip_first_positions + 1:
                warnings.warn(
                    f"skipping prompt with only {seq_len} tokens "
                    f"(need > {skip_first_positions + 1})",
                    stacklevel=3,
                )
                continue
            per_prompt = _jacobian_for_prompt(
                model,
                tokens,
                source_layers=source_layers,
                dim_batch=dim_batch,
                skip_first_positions=skip_first_positions,
                backward_provider=backward_provider,
            )
            for layer in source_layers:
                jacobian_sum[layer] += per_prompt[layer]
            n_done += 1
    if n_done == 0:
        raise ValueError(
            "every prompt was too short to contribute valid positions; nothing was fitted"
        )
    means = {layer: jacobian_sum[layer] / n_done for layer in source_layers}
    return means, n_done
