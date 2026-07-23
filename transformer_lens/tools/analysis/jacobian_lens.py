"""Jacobian Lens (J-lens).

The Jacobian lens characterizes an intermediate residual-stream activation by its
first-order causal effect on the model's output, averaged over a corpus of contexts.
For each layer :math:`\\ell` it fits a single :math:`d_{model} \\times d_{model}` matrix

.. math::

    J_\\ell = \\mathbb{E}_{\\text{prompt}}\\left[
        \\frac{1}{|V|}\\sum_{t \\in V}\\sum_{t' \\in V,\\,t' \\geq t}
        \\frac{\\partial h_{\\text{final},t'}}{\\partial h_{\\ell,t}}
        \\right]

where ``V`` is the set of valid positions after the configured leading skip
and final-position exclusion. Target-position effects are summed for each
source; only source positions and prompts are averaged.

mapping the output of block :math:`\\ell` to the final block's output (pre final
norm). Reading the lens applies the model's own final norm and unembedding:
:math:`\\text{lens}(h_\\ell) = W_U\\,\\mathrm{norm}(J_\\ell h_\\ell)`.
The logit lens is the special case :math:`J_\\ell = I`. The rows of
:math:`W_U J_\\ell` ("J-lens vectors") are residual-stream directions associated
with single vocabulary tokens, and support causal interventions: steering,
ablation, and exchanging one concept for another via a pseudoinverse coordinate
swap.

Introduced in `Verbalizable Representations Form a Global Workspace in Language
Models <https://transformer-circuits.pub/2026/workspace/index.html>`_ (Gurnee et
al., Transformer Circuits Thread, 2026). The fitting estimator and the artifact
format follow Anthropic's Apache-2.0 reference implementation
(`anthropics/jacobian-lens <https://github.com/anthropics/jacobian-lens>`_), so
lenses fitted here interoperate with artifacts published on the Hugging Face Hub
(e.g. `neuronpedia/jacobian-lens
<https://huggingface.co/neuronpedia/jacobian-lens>`_); the interventions are
implemented from the paper's Methods section.

Warning:
    Published lens artifacts are fitted on **raw** HuggingFace activations.
    Jacobian lens supports only a freshly booted
    ``TransformerBridge.boot_transformers`` model, whose weights are raw by
    default. Compatibility mode and direct ``process_weights`` calls change the
    residual basis and are refused rather than returning silently wrong
    readouts. The model must also be a causal decoder whose adapter supports
    text generation, use single-stream block outputs, and expose the standard
    direct ``ln_final -> d_model-width unembed`` output path.

Example::

    import torch
    from transformer_lens.model_bridge import TransformerBridge
    from transformer_lens.tools.analysis import JacobianLens

    model = TransformerBridge.boot_transformers("gpt2", device="cpu")
    lens = JacobianLens.from_pretrained(
        "neuronpedia/jacobian-lens",
        filename="gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt",
        model=model,
    )
    result = lens.readout(model, "The Eiffel Tower is in the city of")
    print(result.top_tokens(model.tokenizer, k=5)[8][-1])  # layer 8, final position
"""

import math
import warnings
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from jaxtyping import Float, Int
from tqdm.auto import tqdm

from transformer_lens.utilities.hf_utils import call_hf_with_retry

TokenInput = Union[str, int]

# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

_REGISTRY_CACHE: Optional[Dict[str, Any]] = None


def _load_registry() -> Dict[str, Any]:
    """Return the bundled artifact registry, loading it once on first call."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        import json
        import pathlib

        registry_path = pathlib.Path(__file__).with_name("jacobian_lens_registry.json")
        with registry_path.open() as fh:
            _REGISTRY_CACHE = json.load(fh)
    return _REGISTRY_CACHE


def _resolve_registry_entry(name_or_path: str) -> Optional[Tuple[str, str]]:
    """Return ``(repo_id, filename)`` if *name_or_path* matches the registry.

    Matching is tried in two passes:
    1. Direct key match against short model names (e.g. ``"gemma-2-2b"``).
    2. Alias match against Hugging Face model IDs (e.g. ``"google/gemma-2-2b"``).

    Returns ``None`` when there is no match so callers can fall through to the
    generic Hub download path.
    """
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
_SWAP_WARN_COSINE = 0.99
_SWAP_ERROR_COSINE = 0.999


@dataclass
class JacobianLensReadout:
    """Result of a :meth:`JacobianLens.readout` call.

    Attributes:
        lens_topk_values:
            Per-layer retained top-k pre-softmax values, on CPU.
        lens_topk_indices:
            Per-layer retained top-k vocabulary ids, on CPU.
        model_topk_values:
            The model output's retained top-k pre-softmax values, on CPU.
        model_topk_indices:
            The model output's retained top-k vocabulary ids, on CPU.
        lens_logits:
            Optional full per-layer logits, on CPU. Present only when
            ``readout(return_full_logits=True)`` was requested.
        model_logits:
            Optional full model logits, on CPU. Present only when
            ``readout(return_full_logits=True)`` was requested.
        tokens:
            The token ids of the run prompt, ``[seq]``.
        positions:
            The (normalized, non-negative) positions the readout covers, aligned
            with the ``pos`` axis of retained top-k and optional full logits.
        use_jacobian:
            Whether the Jacobian transport was applied (``False`` = logit lens).
    """

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
        """Decode the top-``k`` tokens per layer and position.

        Args:
            tokenizer: The model's tokenizer (``model.tokenizer``).
            k: Number of top tokens to decode per (layer, position).

        Returns:
            ``{layer: [ [top-k strings] per position ]}``, positions aligned with
            :attr:`positions`.
        """
        out: Dict[int, List[List[str]]] = {}
        retained = next(iter(self.lens_topk_indices.values())).shape[-1]
        if not 1 <= k <= retained:
            raise ValueError(f"k must be between 1 and the retained top_k={retained}, got {k}")
        for layer, ids in self.lens_topk_indices.items():
            out[layer] = [[tokenizer.decode([t]) for t in row[:k].tolist()] for row in ids]
        return out


class JacobianLens:
    """A fitted Jacobian lens: one transport matrix per source layer.

    Layer convention (matching the reference implementation and the published
    artifacts): index ``l`` refers to the **output of block** ``l`` at the
    Bridge-native hook ``blocks.{l}.hook_out``. ``J[l]`` maps that activation
    to the final block's output, pre final norm. The final layer itself is never
    fitted (its transport is the identity), so
    ``source_layers == [0, ..., n_layers - 2]`` for a full fit.

    Attributes:
        jacobians: ``{layer: [d_model, d_model]}`` transport matrices, fp32, CPU.
        n_prompts: Number of prompts averaged into the fit.
        d_model: Residual stream width the lens was fitted for.
        metadata: Optional provenance (model name, TransformerLens version, fit
            hyperparameters). Preserved by :meth:`save`/:meth:`load`; artifacts
            from the reference implementation load with empty metadata.
    """

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
        self._device_jacobians: Dict[Tuple[int, torch.device], torch.Tensor] = {}

    @property
    def source_layers(self) -> List[int]:
        """Sorted list of layers this lens has transport matrices for."""
        return sorted(self.jacobians)

    def __repr__(self) -> str:
        layers = self.source_layers
        return (
            f"JacobianLens(layers={layers[0]}..{layers[-1]} ({len(layers)}), "
            f"d_model={self.d_model}, n_prompts={self.n_prompts})"
        )

    # ------------------------------------------------------------------ #
    # persistence                                                        #
    # ------------------------------------------------------------------ #

    def save(self, path: str, *, dtype: torch.dtype = torch.float16) -> None:
        """Save the lens in the reference implementation's artifact format.

        The four official keys (``J``, ``n_prompts``, ``source_layers``,
        ``d_model``) are written unchanged so the file stays loadable by the
        reference package; TransformerLens provenance is stored under an
        additive ``metadata`` key.

        Args:
            path: Destination ``.pt`` path.
            dtype: Storage dtype. Defaults to fp16 like the reference
                implementation — Jacobian entries are order-one, so the smaller
                dtype costs little precision and halves the artifact on disk.
        """
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
    def load(cls, path: str) -> "JacobianLens":
        """Load a lens artifact saved by this class or the reference package.

        Args:
            path: Path to the ``.pt`` artifact.

        Raises:
            ValueError: If the file lacks the ``J`` key (e.g. a fit checkpoint
                rather than a saved lens).
        """
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if "J" not in payload:
            raise ValueError(
                f"{path} does not look like a Jacobian lens artifact (no 'J' key). "
                "Fit checkpoints and other formats are not supported."
            )
        return cls(
            {int(layer): matrix for layer, matrix in payload["J"].items()},
            n_prompts=int(payload.get("n_prompts", 0)),
            d_model=int(payload["d_model"]),
            metadata=payload.get("metadata"),
        )

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        *,
        filename: str = "lens.pt",
        revision: Optional[str] = None,
        model: Any = None,
    ) -> "JacobianLens":
        """Load a lens from a local path, a short model name, or a Hub repo.

        Resolution order
        ----------------
        1. **Local file** — if *name_or_path* is an existing ``.pt`` file,
           load it directly.
        2. **Local directory** — if *name_or_path* is a directory, load
           ``<name_or_path>/<filename>``.
        3. **Registry short name or HF model ID** — if *name_or_path* matches
           a key or alias in the bundled ``jacobian_lens_registry.json`` (e.g.
           ``"gemma-2-2b"`` or ``"google/gemma-2-2b"``), the corresponding
           artifact in ``neuronpedia/jacobian-lens`` is fetched automatically.
           The *filename* argument is ignored in this case because the registry
           already encodes the correct subpath.
        4. **Explicit Hub repo** — otherwise *name_or_path* is treated as a Hub
           repo id and *filename* is used as-is, preserving full backward
           compatibility (e.g. ``from_pretrained("neuronpedia/jacobian-lens",
           filename="gpt2-small/jlens/...")``).

        Args:
            name_or_path: A local ``.pt`` file, a local directory, a short
                model name such as ``"gemma-2-2b"`` or ``"llama3.1-8b"``, a
                Hugging Face model ID such as ``"google/gemma-2-2b"``, or an
                explicit Hub repo id paired with *filename*.
            filename: File (or subpath) inside a local directory or an explicit
                Hub repo.  Ignored when *name_or_path* resolves via the
                registry.
            revision: Optional Hub revision (branch, tag, or commit) to pin.
                When omitted, the Hub repository's mutable default branch is
                followed; pin a commit hash for reproducible analyses.
            model: If given, :meth:`validate_model` is called so dimension or
                weight-processing mismatches fail here rather than at first use.

        Returns:
            The loaded (and, if ``model`` was given, validated) lens.

        Examples::

            # Short model name — no need to remember the HF subpath
            lens = JacobianLens.from_pretrained("gemma-2-2b", model=model)

            # HF model ID also works
            lens = JacobianLens.from_pretrained("google/gemma-2-2b", model=model)

            # Explicit Hub repo + subpath (backward-compatible)
            lens = JacobianLens.from_pretrained(
                "neuronpedia/jacobian-lens",
                filename="gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt",
                model=model,
            )
        """
        import os

        if os.path.isfile(name_or_path):
            lens = cls.load(name_or_path)
        elif os.path.isdir(name_or_path):
            lens = cls.load(os.path.join(name_or_path, filename))
        else:
            from huggingface_hub import hf_hub_download

            resolved = _resolve_registry_entry(name_or_path)
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
        """Combine lenses fitted on disjoint prompt slices.

        The per-layer matrices are averaged weighted by each lens's
        ``n_prompts``, matching the reference implementation, so fitting can be
        parallelized across processes or machines and merged afterwards.
        Provenance must match across shards (apart from ``n_prompts``), so a
        merge cannot silently relabel matrices fitted with different models,
        corpora, dtypes, or estimator settings. The merged count replaces the
        per-shard count.

        Args:
            lenses: Lenses that agree exactly on ``source_layers`` and
                ``d_model``.

        Raises:
            ValueError: On an empty sequence or mismatched lenses.
        """
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

    # ------------------------------------------------------------------ #
    # model validation                                                   #
    # ------------------------------------------------------------------ #

    def validate_model(self, model: Any) -> "JacobianLens":
        """Check that ``model`` matches this lens; raise loudly if not.

        Requires a raw causal ``TransformerBridge`` with the standard direct
        final-norm/unembed path, verifies recorded model provenance, residual
        width and layer range, and enforces the published final-block target
        convention.

        Args:
            model: A raw ``TransformerBridge``.

        Returns:
            ``self``, for chaining.

        Raises:
            TypeError: If model is not a ``TransformerBridge``.
            ValueError: On model provenance or ``d_model`` mismatch,
                out-of-range source layers, compatibility mode, unsupported
                attention/output paths, or a non-final target convention.
        """
        _require_raw_bridge(model)
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

    # ------------------------------------------------------------------ #
    # reading                                                            #
    # ------------------------------------------------------------------ #

    def clear_device_cache(self) -> None:
        """Release lazily cached Jacobian copies on accelerator devices."""
        self._device_jacobians.clear()

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
        """Map layer-``layer`` activations into the final block's output basis.

        Computes ``J[layer] @ h`` per activation vector, in fp32.

        Args:
            residual: Activations from the output of block ``layer``.
            layer: Source layer index.
        """
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
        """Read per-layer vocabulary logits for a prompt.

        Runs the model once with caching, transports the residual stream at each
        requested layer through ``J[layer]`` (or the identity when
        ``use_jacobian=False`` — the logit lens), and applies the model's own
        final norm, unembedding, architecture logit scaling, and logit soft cap.

        Args:
            model: A raw ``TransformerBridge``.
            input: A prompt string, or a ``[1, seq]`` token tensor.
            layers: Layers to read. Defaults to every fitted layer plus the
                final layer. The final layer (``n_layers - 1``) is always read
                with the identity transport — by construction its lens equals
                the model's own output distribution.
            positions: Token positions to read (negative indices allowed).
                Defaults to all positions.
            use_jacobian: Apply the Jacobian transport. ``False`` gives the
                logit-lens baseline through the identical code path.
            top_k: Number of values and vocabulary ids retained per layer and
                position. Defaults to 10.
            return_full_logits: Also retain full vocabulary tensors on CPU.
                This is opt-in because a 64-token Gemma readout across all
                layers is roughly 1.7 GB.

        Returns:
            A :class:`JacobianLensReadout`.

        Raises:
            ValueError: If the model fails :meth:`validate_model`, ``input`` is
                batched, ``top_k`` is invalid, or a requested layer has no
                transport matrix.
        """
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
        """Residual-stream directions for vocabulary tokens at a layer.

        The J-lens vector for token ``t`` is row ``t`` of ``W_U J[layer]``
        expressed in layer-``layer`` residual coordinates:
        ``v_t = J[layer]^T W_U[:, t]``.

        Args:
            model: The model supplying ``W_U``.
            tokens: A token string / id, or a sequence of them. Strings must
                encode to a single token.
            layer: Source layer for the vectors.

        Returns:
            One vector per token, fp32, on the model's device.
        """
        self.validate_model(model)
        layer = _normalize_layer(layer, model.cfg.n_layers)
        token_ids = _to_token_ids(model, tokens)
        unembed_columns = model.W_U[:, token_ids].float()  # [d_model, n]
        matrix = self._matrix_on(layer, unembed_columns.device)
        return (matrix.T @ unembed_columns).T

    # ------------------------------------------------------------------ #
    # interventions                                                      #
    # ------------------------------------------------------------------ #

    def steering_hooks(
        self,
        model: Any,
        token: TokenInput,
        layers: Sequence[int],
        *,
        alpha: float = 4.0,
        positions: Optional[Sequence[int]] = None,
    ) -> List[Tuple[str, Any]]:
        """Hooks that steer the residual stream along a token's J-lens vector.

        At each layer the unit-normalized lens vector is added, scaled by
        ``alpha`` times the activation's **median** per-position residual norm:
        ``h <- h + alpha * median||h|| * v̂``. This norm-matched
        parameterization follows the steering description in the reference
        implementation's experiment protocols; the paper's minimal form is
        the unscaled ``h <- h + alpha * v_t``, recoverable by passing the raw
        :meth:`lens_vectors` output to your own hook. The median (not mean) is
        used so attention-sink positions — whose residual norms run orders of
        magnitude above typical positions — do not inflate the scale.

        Args:
            model: The model the hooks will run on.
            token: The concept token to steer toward.
            layers: Layers to intervene at.
            alpha: Steering strength scalar; ``0`` disables. Because of the
                norm-matched scale, values of order 1 already perturb the
                stream by roughly its own magnitude.
            positions: Chunk-local positions to steer (negative indices allowed
                and normalized on every hook invocation). Defaults to all.

        Returns:
            ``[(hook_name, fn), ...]`` for ``model.hooks(fwd_hooks=...)`` or
            ``model.run_with_hooks(fwd_hooks=...)``.
        """
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
        """Hooks that project token directions out of the residual stream.

        For each token's unit lens vector ``v̂``: ``h <- h - (h·v̂) v̂``,
        applied sequentially when several tokens are given.

        Args:
            model: The model the hooks will run on.
            tokens: Concept token(s) to suppress.
            layers: Layers to intervene at.
            positions: Chunk-local positions to ablate (negative indices allowed
                and normalized on every hook invocation). Defaults to all.

        Returns:
            ``[(hook_name, fn), ...]`` for ``model.hooks(fwd_hooks=...)``.
        """
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
        """Hooks that swap two concepts' coordinates in lens space.

        The paper's patching-in-lens-coordinates intervention: with
        ``V = [v_s, v_t]`` and lens coordinates ``c = V⁺ h`` (pseudoinverse),
        the update is ``h <- h + alpha * V (sigma(c) - c)`` where ``sigma``
        exchanges the two coordinates. The component of ``h`` orthogonal to
        ``span{v_s, v_t}`` is untouched. ``alpha=2`` is the paper's
        "double-strength" swap.

        Args:
            model: The model the hooks will run on.
            source_token: The concept to remove (e.g. ``" France"``).
            target_token: The concept to install (e.g. ``" China"``).
            layers: Layers to intervene at (the paper clamps the swap across an
                intermediate-layer band).
            alpha: Swap strength.
            positions: Chunk-local positions to swap (negative indices allowed
                and normalized on every hook invocation). Defaults to all.

        Returns:
            ``[(hook_name, fn), ...]`` for ``model.hooks(fwd_hooks=...)``.
        """
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
            cosine = abs(float((units[0] @ units[1]).item()))
            if not math.isfinite(cosine) or cosine >= _SWAP_ERROR_COSINE:
                raise ValueError(
                    f"swap vectors at layer {layer} are numerically near-parallel "
                    f"(abs cosine={cosine:.6f}); choose better-separated concepts"
                )
            if cosine >= _SWAP_WARN_COSINE:
                warnings.warn(
                    f"swap vectors at layer {layer} are poorly conditioned "
                    f"(abs cosine={cosine:.6f}); the intervention may be amplified",
                    UserWarning,
                    stacklevel=2,
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

    # ------------------------------------------------------------------ #
    # fitting                                                            #
    # ------------------------------------------------------------------ #

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
        """Fit a Jacobian lens on a hooked model.

        Implements the reference estimator exactly. For each prompt: one forward
        pass (the prompt replicated ``dim_batch`` times along the batch axis),
        then ``ceil(d_model / dim_batch)`` backward passes. Each backward plants
        a one-hot cotangent for one output dimension at *every* valid target
        position simultaneously — causal attention guarantees the gradient at
        source position ``t`` is then the sum over target positions
        ``t' >= t`` with no explicit masking. Rows are averaged over valid
        source positions (the first ``skip_first_positions`` and the final
        position are excluded), and prompts contribute equally to the final
        mean. There is no randomness: the computation is deterministic given
        the prompts.

        The reference implementation reports that fit quality saturates
        quickly — on the order of 100 prompts of 128 tokens is usable; the
        published lenses use up to 1000. Use :meth:`merge` to parallelize
        across prompt slices.

        Args:
            model: A raw ``TransformerBridge``. Model parameters are temporarily
                frozen (``requires_grad=False``) during fitting and restored
                after. Cotangents and activation gradients use the model dtype;
                fit with a float32 model for the highest-fidelity estimator.
            prompts: Prompt strings. Prompts too short to contain a valid
                position (``seq_len <= skip_first_positions + 1``) are skipped
                with a warning and do not count toward ``n_prompts``.
            corpus: Stable identifier for the prompt corpus or slice, recorded
                in artifact provenance.
            source_layers: Layers to fit. Defaults to every layer below
                the final layer. Negative indices count from ``n_layers``.
            dim_batch: Output dimensions per backward pass. Higher is faster
                but replicates the prompt ``dim_batch`` times in memory; total
                backward FLOPs are unchanged.
            max_seq_len: Prompts are truncated to this many tokens.
            skip_first_positions: Leading positions excluded from the source
                average.
            show_progress: Show a tqdm progress bar over prompts.
            metadata: Extra provenance merged into :attr:`metadata`.

        Returns:
            The fitted :class:`JacobianLens`.

        Raises:
            TypeError: If model is not a ``TransformerBridge``.
            ValueError: On compatibility mode, invalid provenance or layer
                indices, or if no prompt was long enough to fit on.
        """
        _require_raw_bridge(model)
        if not isinstance(corpus, str) or not corpus.strip():
            raise ValueError("corpus must be a non-empty provenance identifier")
        n_layers = model.cfg.n_layers
        d_model = model.cfg.d_model
        resolved_target = n_layers - 1
        if source_layers is None:
            resolved_sources = list(range(resolved_target))
        else:
            resolved_sources = sorted(
                {_normalize_layer(layer, n_layers) for layer in source_layers}
            )
        if not resolved_sources:
            raise ValueError("source_layers is empty")
        if resolved_sources[-1] >= resolved_target:
            raise ValueError(
                f"every source layer must be below target_layer={resolved_target}; "
                f"got {resolved_sources}"
            )
        if dim_batch < 1:
            raise ValueError(f"dim_batch must be >= 1, got {dim_batch}")
        if skip_first_positions < 0:
            raise ValueError(f"skip_first_positions must be >= 0, got {skip_first_positions}")
        fit_dtype = model.W_U.dtype
        if fit_dtype in (torch.float16, torch.bfloat16):
            warnings.warn(
                f"fitting in {fit_dtype} accumulates Jacobian gradients at reduced "
                "precision; use a float32 TransformerBridge for the highest-fidelity fit",
                UserWarning,
                stacklevel=2,
            )

        jacobian_sum = {
            layer: torch.zeros(d_model, d_model, dtype=torch.float32) for layer in resolved_sources
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
                        stacklevel=2,
                    )
                    continue
                per_prompt = _jacobian_for_prompt(
                    model,
                    tokens,
                    source_layers=resolved_sources,
                    dim_batch=dim_batch,
                    skip_first_positions=skip_first_positions,
                )
                for layer in resolved_sources:
                    jacobian_sum[layer] += per_prompt[layer]
                n_done += 1
        if n_done == 0:
            raise ValueError(
                "every prompt was too short to contribute valid positions; nothing was fitted"
            )

        fit_metadata: Dict[str, Any] = {
            "model_name": getattr(model.cfg, "model_name", None),
            "model_revision": _get_model_revision(model),
            "transformer_lens_version": version("transformer-lens"),
            "model_system": "TransformerBridge",
            "processing": {
                "compatibility_mode": False,
                "weight_basis": "raw_huggingface",
            },
            "hook_convention": "blocks.{layer}.hook_out",
            "corpus": corpus,
            "n_prompts": n_done,
            "fit_dtype": str(fit_dtype).removeprefix("torch."),
            "target_layer": resolved_target,
            "dim_batch": dim_batch,
            "max_seq_len": max_seq_len,
            "skip_first_positions": skip_first_positions,
            "transformer_lens_fit": True,
        }
        reserved = sorted(set(fit_metadata).intersection(metadata or {}))
        if reserved:
            raise ValueError(f"metadata cannot override fit provenance keys: {reserved}")
        full_metadata = dict(metadata or {})
        full_metadata.update(fit_metadata)
        _validate_metadata(full_metadata)
        return cls(
            {layer: jacobian_sum[layer] / n_done for layer in resolved_sources},
            n_prompts=n_done,
            d_model=d_model,
            metadata=full_metadata,
        )


# ---------------------------------------------------------------------- #
# helpers                                                                #
# ---------------------------------------------------------------------- #


def _resid_post_hook_name(layer: int) -> str:
    """Bridge-native hook for the output of block ``layer``."""
    return f"blocks.{layer}.hook_out"


def _get_model_revision(model: Any) -> Optional[str]:
    """Return the resolved Hugging Face commit recorded on a booted model."""
    original_model = getattr(model, "original_model", None)
    hf_config = getattr(original_model, "config", None)
    revision = getattr(hf_config, "_commit_hash", None)
    return revision if isinstance(revision, str) and revision else None


def _require_raw_bridge(model: Any) -> None:
    """Require the causal raw-Bridge contract used by fit and readout."""
    from transformer_lens.model_bridge import TransformerBridge

    if not isinstance(model, TransformerBridge):
        raise TypeError(
            "JacobianLens supports TransformerBridge only; load a fresh model with "
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
            "JacobianLens requires a causal decoder-only Bridge whose adapter "
            f"supports text generation; {type(adapter).__name__} declares "
            "supports_generation=False."
        )
    adapter.validate_output_logits_transform()
    attention_dir = getattr(model.cfg, "attention_dir", "causal")
    if attention_dir != "causal":
        raise ValueError(
            "JacobianLens requires causal attention because its estimator relies on "
            "causality to exclude target positions before each source position; "
            f"got attention_dir={attention_dir!r}."
        )
    total_ut_steps = int(getattr(model.cfg, "total_ut_steps", 1) or 1)
    if total_ut_steps != 1:
        raise ValueError(
            "JacobianLens requires each physical block hook to fire once per forward; "
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
            "JacobianLens requires the standard direct ln_final -> unembed output path; "
            f"this Bridge is missing {missing_components}."
        )
    blocks_component = component_mapping["blocks"]
    if not getattr(blocks_component, "hook_out_is_single_residual_stream", False):
        raise ValueError(
            "JacobianLens requires single-stream [batch, position, d_model] block "
            f"outputs; {type(blocks_component).__name__} does not provide that contract."
        )
    if "project_out" in component_mapping:
        raise ValueError(
            "JacobianLens does not yet support a final output projection between "
            "the residual stream and unembedding."
        )
    unembed_width = model.W_U.shape[0]
    if unembed_width != model.cfg.d_model:
        raise ValueError(
            "JacobianLens requires a direct d_model-width unembedding after ln_final; "
            f"got W_U input width {unembed_width} for d_model={model.cfg.d_model}. "
            "Architectures with a final output projection are not yet supported."
        )


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
    """Apply the model's own final norm, unembedding, logit scale, and soft cap.

    The norm/unembed components contract on ``[batch, pos, d_model]``, so the
    position rows are passed through with a singleton batch axis. The residual
    is moved to the unembedding's device for sharded/multi-GPU models.
    """
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
    """Context manager: freeze all parameters, restore their flags on exit.

    Freezing keeps the autograd graph rooted at the residual stream rather than
    at the weights, so fitting retains only the blocks between the earliest
    source layer and the target layer.
    """

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


def _jacobian_for_prompt(
    model: Any,
    tokens: Int[torch.Tensor, "one seq"],
    *,
    source_layers: List[int],
    dim_batch: int,
    skip_first_positions: int,
) -> Dict[int, Float[torch.Tensor, "d_model d_model"]]:
    """Exact per-prompt Jacobian rows via batched one-hot cotangents.

    Assumes parameters are already frozen (see :class:`_frozen_parameters`) so
    that marking the earliest source activation ``requires_grad`` roots the
    graph there.
    """
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
    n_passes = -(-d_model // dim_batch)  # ceil division
    for pass_index in range(n_passes):
        dim_start = pass_index * dim_batch
        n_dims = min(dim_batch, d_model - dim_start)
        cotangent.zero_()
        cotangent[
            batch_index[:n_dims, None],
            positions_index[None, :],
            dim_start + batch_index[:n_dims, None],
        ] = 1.0
        grads = torch.autograd.grad(
            outputs=target,
            inputs=sources,
            grad_outputs=cotangent,
            retain_graph=pass_index < n_passes - 1,
        )
        for layer, grad in zip(source_layers, grads):
            # each gradient lives on its layer's device under sharded/device_map setups
            rows = grad[:n_dims, positions_index.to(grad.device), :].float().mean(dim=1)
            jacobians[layer][dim_start : dim_start + n_dims, :] = rows.cpu()
        del grads
    return jacobians
