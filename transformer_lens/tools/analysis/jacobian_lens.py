"""Jacobian Lens (J-lens).

The Jacobian lens characterizes an intermediate residual-stream activation by its
first-order causal effect on the model's output, averaged over a corpus of contexts.
For each layer :math:`\\ell` it fits a single :math:`d_{model} \\times d_{model}` matrix

.. math::

    J_\\ell = \\mathbb{E}_{\\text{prompt},\\,t,\\,t' \\geq t}
        \\left[ \\partial h_{\\text{final},t'} / \\partial h_{\\ell,t} \\right]

mapping the output of block :math:`\\ell` to the final block's output (pre final
norm). Reading the lens applies the model's own final norm and unembedding:
:math:`\\text{lens}(h_\\ell) = \\mathrm{softmax}(W_U\\,\\mathrm{norm}(J_\\ell h_\\ell))`.
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
    Published lens artifacts are fitted on **raw** HuggingFace activations. Use
    ``TransformerBridge.boot_transformers`` (raw weights by default) or
    ``HookedTransformer.from_pretrained_no_processing``. Weight processing
    (``fold_ln``, ``center_writing_weights``, ``center_unembed``) changes the
    residual basis, so :class:`JacobianLens` refuses processed models rather
    than returning silently wrong readouts.

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

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from jaxtyping import Float, Int
from tqdm.auto import tqdm

from transformer_lens.utilities.activation_functions import apply_softcap

TokenInput = Union[str, int]

# Fitting excludes early positions (attention sinks with atypical residual statistics)
# and the final position (no next-token target), matching the reference implementation.
DEFAULT_SKIP_FIRST_POSITIONS = 16

_PROCESSED_NORM_SUFFIX = "Pre"


@dataclass
class JacobianLensReadout:
    """Result of a :meth:`JacobianLens.readout` call.

    Attributes:
        lens_logits:
            Per-layer lens logits, ``{layer: tensor of shape [pos, d_vocab]}``
            (fp32, on CPU). Entries are pre-softmax scores over the vocabulary;
            ranks are what the paper reports, so no softmax is applied.
        model_logits:
            The model's actual output logits at the same positions,
            ``[pos, d_vocab]`` (fp32, on CPU).
        tokens:
            The token ids of the run prompt, ``[seq]``.
        positions:
            The (normalized, non-negative) positions the readout covers, aligned
            with the ``pos`` axis of ``lens_logits`` / ``model_logits``.
        use_jacobian:
            Whether the Jacobian transport was applied (``False`` = logit lens).
    """

    lens_logits: Dict[int, Float[torch.Tensor, "pos d_vocab"]]
    model_logits: Float[torch.Tensor, "pos d_vocab"]
    tokens: Int[torch.Tensor, "seq"]
    positions: List[int]
    use_jacobian: bool = True

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
        for layer, logits in self.lens_logits.items():
            ids = logits.topk(k, dim=-1).indices
            out[layer] = [[tokenizer.decode([t]) for t in row.tolist()] for row in ids]
        return out


class JacobianLens:
    """A fitted Jacobian lens: one transport matrix per source layer.

    Layer convention (matching the reference implementation and the published
    artifacts): index ``l`` refers to the **output of block** ``l`` —
    ``blocks.{l}.hook_resid_post`` on a ``HookedTransformer``,
    ``blocks.{l}.hook_out`` on a ``TransformerBridge``. ``J[l]`` maps that
    activation to the final block's output, pre final norm. The final layer
    itself is never fitted (its transport is the identity), so
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
            dtype: Storage dtype. fp16 halves file size; entries are O(1) so
                range is not a constraint (the reference default).
        """
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
        """Load a lens from a local path or a Hugging Face Hub repository.

        Args:
            name_or_path: A local ``.pt`` file, a local directory containing
                ``filename``, or a Hub repo id such as
                ``"neuronpedia/jacobian-lens"``.
            filename: File (or subpath) inside the directory / Hub repo. Hub
                repos host lenses for many models, e.g.
                ``"gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"``.
            revision: Optional Hub revision (branch, tag, or commit) to pin.
            model: If given, :meth:`validate_model` is called so dimension or
                weight-processing mismatches fail here rather than at first use.

        Returns:
            The loaded (and, if ``model`` was given, validated) lens.
        """
        import os

        if os.path.isfile(name_or_path):
            lens = cls.load(name_or_path)
        elif os.path.isdir(name_or_path):
            lens = cls.load(os.path.join(name_or_path, filename))
        else:
            from huggingface_hub import hf_hub_download

            local_path = hf_hub_download(name_or_path, filename, revision=revision)
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

        Args:
            lenses: Lenses that agree exactly on ``source_layers`` and
                ``d_model``.

        Raises:
            ValueError: On an empty sequence or mismatched lenses.
        """
        if not lenses:
            raise ValueError("merge() needs at least one lens")
        first = lenses[0]
        for other in lenses[1:]:
            if other.source_layers != first.source_layers or other.d_model != first.d_model:
                raise ValueError("lenses disagree on source_layers / d_model")
        total = sum(lens.n_prompts for lens in lenses)
        if total <= 0:
            raise ValueError("merge() needs lenses with positive n_prompts")
        merged = {
            layer: torch.stack([lens.jacobians[layer] * lens.n_prompts for lens in lenses]).sum(
                dim=0
            )
            / total
            for layer in first.source_layers
        }
        metadata = dict(first.metadata)
        return cls(merged, n_prompts=total, d_model=first.d_model, metadata=metadata)

    # ------------------------------------------------------------------ #
    # model validation                                                   #
    # ------------------------------------------------------------------ #

    def validate_model(self, model: Any) -> "JacobianLens":
        """Check that ``model`` matches this lens; raise loudly if not.

        Verifies the residual width and layer range, and refuses models whose
        weights have been processed into a different residual basis (see the
        module warning).

        Args:
            model: A ``HookedTransformer`` or ``TransformerBridge``.

        Returns:
            ``self``, for chaining.

        Raises:
            ValueError: On ``d_model`` mismatch, out-of-range source layers, or
                processed weights.
        """
        d_model = model.cfg.d_model
        if d_model != self.d_model:
            raise ValueError(
                f"lens was fitted for d_model={self.d_model}, but the model has "
                f"d_model={d_model} — this lens belongs to a different model."
            )
        n_layers = model.cfg.n_layers
        out_of_range = [layer for layer in self.source_layers if not 0 <= layer < n_layers]
        if out_of_range:
            raise ValueError(
                f"lens has source layers {out_of_range} outside the model's "
                f"0..{n_layers - 1} range — this lens belongs to a different model."
            )
        _require_unprocessed_weights(model)
        return self

    # ------------------------------------------------------------------ #
    # reading                                                            #
    # ------------------------------------------------------------------ #

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
        if layer not in self.jacobians:
            raise ValueError(
                f"layer {layer} is not in this lens's source layers "
                f"({self.source_layers[0]}..{self.source_layers[-1]})"
            )
        matrix = self.jacobians[layer].to(residual.device)
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
    ) -> JacobianLensReadout:
        """Read per-layer vocabulary logits for a prompt.

        Runs the model once with caching, transports the residual stream at each
        requested layer through ``J[layer]`` (or the identity when
        ``use_jacobian=False`` — the logit lens), and applies the model's own
        final norm, unembedding, and logit soft cap.

        Args:
            model: A ``HookedTransformer`` or ``TransformerBridge`` with raw
                (unprocessed) weights.
            input: A prompt string, or a ``[1, seq]`` token tensor.
            layers: Layers to read. Defaults to every fitted layer plus the
                final layer. The final layer (``n_layers - 1``) is always read
                with the identity transport — by construction its lens equals
                the model's own output distribution.
            positions: Token positions to read (negative indices allowed).
                Defaults to all positions.
            use_jacobian: Apply the Jacobian transport. ``False`` gives the
                logit-lens baseline through the identical code path.

        Returns:
            A :class:`JacobianLensReadout`.

        Raises:
            ValueError: If the model fails :meth:`validate_model`, ``input`` is
                batched, or a requested layer has no transport matrix.
        """
        self.validate_model(model)
        tokens = model.to_tokens(input) if isinstance(input, str) else input
        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError(f"readout expects a single prompt; got shape {tuple(tokens.shape)}")
        n_layers = model.cfg.n_layers
        final_layer = n_layers - 1
        if layers is None:
            layers = self.source_layers + [final_layer]
        layers = [layer + n_layers if layer < 0 else layer for layer in layers]
        for layer in layers:
            if use_jacobian and layer != final_layer and layer not in self.jacobians:
                raise ValueError(
                    f"layer {layer} is not in this lens's source layers; "
                    f"available: {self.source_layers} (+{final_layer} as identity)"
                )

        seq_len = tokens.shape[1]
        if positions is None:
            norm_positions = list(range(seq_len))
        else:
            norm_positions = [pos + seq_len if pos < 0 else pos for pos in positions]

        hook_names = {layer: _resid_post_hook_name(model, layer) for layer in layers}
        wanted = set(hook_names.values())
        logits, cache = model.run_with_cache(tokens, names_filter=lambda name: name in wanted)

        lens_logits: Dict[int, torch.Tensor] = {}
        for layer in layers:
            residual = cache[hook_names[layer]][0, norm_positions, :]
            transported = (
                self.transport(residual, layer)
                if use_jacobian and layer != final_layer
                else residual.float()
            )
            lens_logits[layer] = _unembed(model, transported)
        return JacobianLensReadout(
            lens_logits=lens_logits,
            model_logits=logits[0, norm_positions, :].float().cpu(),
            tokens=tokens[0].cpu(),
            positions=norm_positions,
            use_jacobian=use_jacobian,
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
        token_ids = _to_token_ids(model, tokens)
        matrix = self.jacobians.get(layer)
        if matrix is None:
            raise ValueError(
                f"layer {layer} is not in this lens's source layers "
                f"({self.source_layers[0]}..{self.source_layers[-1]})"
            )
        unembed_columns = model.W_U[:, token_ids].float()  # [d_model, n]
        return (matrix.to(unembed_columns.device).T @ unembed_columns).T

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

        At each layer the unit-normalized lens vector is added, scaled by the
        activation's mean residual norm times ``alpha`` (the paper's directed
        modulation protocol): ``h <- h + alpha * mean||h|| * v̂``.

        Args:
            model: The model the hooks will run on.
            token: The concept token to steer toward.
            layers: Layers to intervene at.
            alpha: Steering strength scalar (0 disables; the paper sweeps
                small integer strengths).
            positions: Positions to steer. Defaults to all positions.

        Returns:
            ``[(hook_name, fn), ...]`` for ``model.hooks(fwd_hooks=...)`` or
            ``model.run_with_hooks(fwd_hooks=...)``.
        """
        hooks = []
        for layer in layers:
            direction = self.lens_vectors(model, token, layer)[0]
            unit = direction / direction.norm()

            def steer_fn(
                activation: Float[torch.Tensor, "batch pos d_model"],
                hook: Any,
                unit: torch.Tensor = unit,
            ) -> Float[torch.Tensor, "batch pos d_model"]:
                scale = alpha * activation.float().norm(dim=-1).mean()
                delta = (scale * unit).to(activation.dtype)
                if positions is None:
                    return activation + delta
                activation[:, list(positions), :] += delta
                return activation

            hooks.append((_resid_post_hook_name(model, layer), steer_fn))
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
            positions: Positions to ablate. Defaults to all positions.

        Returns:
            ``[(hook_name, fn), ...]`` for ``model.hooks(fwd_hooks=...)``.
        """
        hooks = []
        for layer in layers:
            vectors = self.lens_vectors(model, tokens, layer)
            units = vectors / vectors.norm(dim=-1, keepdim=True)

            def ablate_fn(
                activation: Float[torch.Tensor, "batch pos d_model"],
                hook: Any,
                units: torch.Tensor = units,
            ) -> Float[torch.Tensor, "batch pos d_model"]:
                sliced = activation if positions is None else activation[:, list(positions), :]
                result = sliced.float()
                for unit in units:
                    coeff = result @ unit
                    result = result - coeff.unsqueeze(-1) * unit
                result = result.to(activation.dtype)
                if positions is None:
                    return result
                activation[:, list(positions), :] = result
                return activation

            hooks.append((_resid_post_hook_name(model, layer), ablate_fn))
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
            positions: Positions to swap. Defaults to all positions.

        Returns:
            ``[(hook_name, fn), ...]`` for ``model.hooks(fwd_hooks=...)``.
        """
        hooks = []
        for layer in layers:
            basis = self.lens_vectors(model, [source_token, target_token], layer).T  # [d, 2]
            pinv = torch.linalg.pinv(basis)  # [2, d]

            def swap_fn(
                activation: Float[torch.Tensor, "batch pos d_model"],
                hook: Any,
                basis: torch.Tensor = basis,
                pinv: torch.Tensor = pinv,
            ) -> Float[torch.Tensor, "batch pos d_model"]:
                sliced = activation if positions is None else activation[:, list(positions), :]
                coords = sliced.float() @ pinv.T  # [..., 2]
                delta = alpha * ((coords[..., [1, 0]] - coords) @ basis.T)
                result = (sliced.float() + delta).to(activation.dtype)
                if positions is None:
                    return result
                activation[:, list(positions), :] = result
                return activation

            hooks.append((_resid_post_hook_name(model, layer), swap_fn))
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
        source_layers: Optional[Sequence[int]] = None,
        target_layer: Optional[int] = None,
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
            model: A ``HookedTransformer`` or ``TransformerBridge`` with raw
                (unprocessed) weights. Model parameters are temporarily frozen
                (``requires_grad=False``) during fitting and restored after.
            prompts: Prompt strings. Prompts too short to contain a valid
                position (``seq_len <= skip_first_positions + 1``) are skipped
                with a warning and do not count toward ``n_prompts``.
            source_layers: Layers to fit. Defaults to every layer below
                ``target_layer``. Negative indices count from ``n_layers``.
            target_layer: Layer whose output the Jacobians map to. Defaults to
                the final layer (``n_layers - 1``, the convention of the
                published artifacts).
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
            ValueError: On processed weights, invalid layer indices, or if no
                prompt was long enough to fit on.
        """
        _require_unprocessed_weights(model)
        n_layers = model.cfg.n_layers
        d_model = model.cfg.d_model
        resolved_target = _normalize_layer(
            n_layers - 1 if target_layer is None else target_layer, n_layers
        )
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
                    target_layer=resolved_target,
                    dim_batch=dim_batch,
                    skip_first_positions=skip_first_positions,
                )
                for layer in resolved_sources:
                    jacobian_sum[layer] += per_prompt[layer]
                n_done += 1
        if n_done == 0:
            raise ValueError("no prompt was long enough to fit on")

        full_metadata: Dict[str, Any] = {
            "model_name": getattr(model.cfg, "model_name", None),
            "hook_convention": "resid_post",
            "target_layer": resolved_target,
            "dim_batch": dim_batch,
            "max_seq_len": max_seq_len,
            "skip_first_positions": skip_first_positions,
            "transformer_lens_fit": True,
        }
        full_metadata.update(metadata or {})
        return cls(
            {layer: jacobian_sum[layer] / n_done for layer in resolved_sources},
            n_prompts=n_done,
            d_model=d_model,
            metadata=full_metadata,
        )


# ---------------------------------------------------------------------- #
# helpers                                                                #
# ---------------------------------------------------------------------- #


def _is_bridge(model: Any) -> bool:
    """True when ``model`` is a TransformerBridge (lazy import, DLA pattern)."""
    from transformer_lens.model_bridge import TransformerBridge

    return isinstance(model, TransformerBridge)


def _resid_post_hook_name(model: Any, layer: int) -> str:
    """Hook name for the output of block ``layer`` on either model class."""
    if _is_bridge(model):
        return f"blocks.{layer}.hook_out"
    return f"blocks.{layer}.hook_resid_post"


def _require_unprocessed_weights(model: Any) -> None:
    """Refuse models whose weights may have left the raw HF basis.

    ``fold_ln`` converts the norm type to ``LNPre``/``RMSPre`` on a
    ``HookedTransformer`` — a reliable marker. A ``TransformerBridge`` keeps no
    marker: ``enable_compatibility_mode()`` processes weights by default, and a
    ``no_processing=True`` call cannot be reliably told apart afterwards, so any
    compatibility-mode bridge is refused (the mirror image of DLA, which
    *requires* compatibility mode because it reasons in the folded basis).
    """
    norm_type = getattr(model.cfg, "normalization_type", None) or ""
    if norm_type.endswith(_PROCESSED_NORM_SUFFIX):
        raise ValueError(
            "this model was loaded with fold_ln weight processing "
            f"(normalization_type={norm_type!r}), which changes the residual basis "
            "the lens was fitted in. Reload it with "
            "HookedTransformer.from_pretrained_no_processing(...) or use a raw "
            "TransformerBridge.boot_transformers(...) model."
        )
    if _is_bridge(model) and getattr(model, "compatibility_mode", False):
        raise ValueError(
            "compatibility mode is enabled on this TransformerBridge. "
            "enable_compatibility_mode() applies HookedTransformer-style weight "
            "processing by default, which changes the residual basis the lens was "
            "fitted in, and a no_processing=True call cannot be reliably told apart "
            "afterwards. Use a freshly booted TransformerBridge.boot_transformers(...) "
            "model (raw weights) for Jacobian-lens analyses."
        )


def _unembed(
    model: Any, residual: Float[torch.Tensor, "pos d_model"]
) -> Float[torch.Tensor, "pos d_vocab"]:
    """Apply the model's own final norm, unembedding, and logit soft cap.

    The norm/unembed components contract on ``[batch, pos, d_model]``, so the
    position rows are passed through with a singleton batch axis.
    """
    compute_dtype = model.W_U.dtype
    batched = residual.to(compute_dtype).unsqueeze(0)
    logits = model.unembed(model.ln_final(batched)).float().squeeze(0)
    soft_cap = getattr(model.cfg, "output_logits_soft_cap", None)
    return apply_softcap(logits, soft_cap).cpu()


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
    return ids


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
    target_layer: int,
    dim_batch: int,
    skip_first_positions: int,
) -> Dict[int, Float[torch.Tensor, "d_model d_model"]]:
    """Exact per-prompt Jacobian rows via batched one-hot cotangents.

    Assumes parameters are already frozen (see :class:`_frozen_parameters`) so
    that marking the earliest source activation ``requires_grad`` roots the
    graph there.
    """
    d_model = model.cfg.d_model
    seq_len = tokens.shape[1]
    valid_positions = list(range(skip_first_positions, seq_len - 1))
    replicated = tokens.expand(dim_batch, -1)

    captured: Dict[str, torch.Tensor] = {}
    root_name = _resid_post_hook_name(model, min(source_layers))
    hook_layers = sorted(set(source_layers) | {target_layer})

    def capture_fn(
        activation: Float[torch.Tensor, "batch pos d_model"], hook: Any
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        if hook.name == root_name and not activation.requires_grad:
            activation.requires_grad_(True)
        captured[hook.name] = activation
        return activation

    fwd_hooks = [(_resid_post_hook_name(model, layer), capture_fn) for layer in hook_layers]
    with torch.enable_grad(), model.hooks(fwd_hooks=fwd_hooks):
        model(replicated, return_type=None)

    target = captured[_resid_post_hook_name(model, target_layer)]
    sources = [captured[_resid_post_hook_name(model, layer)] for layer in source_layers]
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
            rows = grad[:n_dims, positions_index, :].float().mean(dim=1)
            jacobians[layer][dim_start : dim_start + n_dims, :] = rows.cpu()
    return jacobians
