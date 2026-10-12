"""Logit lens and vocabulary readout.

The `logit lens <https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens>`_
reads every intermediate residual-stream state through the model's own final
norm and unembedding, so each layer can be asked "what would the model predict
if it stopped here?". :func:`logit_lens` does that in one call over a cache's
accumulated residual stream; :func:`logit_readout` is the kernel underneath it
and reads *any* ``d_model`` vector (a head output, a steering direction, a
sparse-coder decoder column) through the same path.

Both go through the real ``ln_final`` module, ``W_U``, ``b_U`` and the
adapter's post-unembedding transform (Gemma-style softcap, Cohere logit scale),
so the final stack entry reproduces the model's own logits on a raw bridge and
the model's log-probabilities in compatibility mode (``center_unembed`` shifts
raw logits by a per-position constant). Logits are produced chunk by chunk, so
``[layers, batch, pos, d_vocab]`` is never materialized unless the full
vocabulary is explicitly requested.

Example::

    from transformer_lens.model_bridge import TransformerBridge
    from transformer_lens.tools.analysis import logit_lens

    model = TransformerBridge.boot_transformers("gpt2", device="cpu")
    result = logit_lens(model, "The Eiffel Tower is in the city of", targets=" Paris")
    for label, rank in zip(result.labels, result.rank_trajectory(" Paris").tolist()):
        print(f"{label:>12}: rank {rank}")

    # Rank of the actual next token at every position and layer: a per-row target.
    tokens = model.to_tokens("The Eiffel Tower is in the city of Paris")
    nxt = logit_lens(model, tokens[:, :-1], positions=None, targets=tokens[:, 1:])
    nxt.rank_trajectory()  # [layers, batch, pos]
"""

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from jaxtyping import Float, Int

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.tools.analysis._model_state import require_eval_mode
from transformer_lens.utilities import Slice, SliceInput
from transformer_lens.utilities.logits_utils import logits_to_df
from transformer_lens.utilities.quantization import require_readable_weight

# Shared targets: one string / id or a sequence of them. A tensor means per-row
# targets (see logit_readout).
TokenInput = Union[str, int, Sequence[Union[str, int]], torch.Tensor]

_RETURN_TYPES = ("logits", "log_probs", "probs")
_INDEX_DTYPES = (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8)
# Value-column names, matching utilities.logits_utils.logits_to_df.
_VALUE_COLUMNS = {"logits": "logit", "log_probs": "log_prob", "probs": "probability"}

# Normalization types for which the model has a final norm the lens must apply.
_FINAL_NORM_TYPES = ("LN", "LNPre", "RMS", "RMSPre")

# Block variants without a ``resid_mid`` hook; mirrors direct_logit_attribution.
_HYBRID_VARIANT_NAMES = ("mamba", "ssm", "mixer", "linear_attn")


@dataclass
class LogitReadout:
    """Result of :func:`logit_readout`.

    Exactly one of the three selectors expands the trailing axis of ``values``:
    ``targets`` gives ``[..., n_targets]`` (``[..., 1]`` for per-row targets),
    ``top_k`` gives ``[..., k]`` with ``top_ids``, ``vocab`` gives
    ``[..., len(vocab)]``; with none of them the trailing axis is the full
    vocabulary. ``entropy`` and ``target_ranks`` are always computed over the
    full vocabulary, whatever was selected.

    Attributes:
        values: The requested quantity (see ``return_type``) with the leading
            dims of the input vectors and the selected trailing axis.
        return_type: ``"logits"``, ``"log_probs"`` or ``"probs"``.
        entropy: Full-vocabulary entropy of the softmax, in nats, ``[...]``.
        top_ids: Token ids aligned with ``values`` when ``top_k`` was used.
        vocab_ids: The requested subset ids when ``vocab`` was used.
        target_ids: Shared targets as ``[n_targets]``; per-row targets as a
            tensor with the leading dims of the vectors (one id per row).
        target_ranks: Zero-based full-vocabulary rank of each target (0 is the
            argmax; ties count only strictly larger logits), aligned with
            ``values``.
        per_row_targets: Whether ``targets`` was given per row (a tensor).
        applied_ln: Whether the model's final norm was applied to the vectors.
        applied_output_transform: Whether the adapter's post-unembedding
            transform was invoked (it is the identity for most architectures).
    """

    values: Float[torch.Tensor, "... selected"]
    return_type: str
    entropy: Float[torch.Tensor, "..."]
    top_ids: Optional[Int[torch.Tensor, "... k"]] = None
    vocab_ids: Optional[Int[torch.Tensor, "v"]] = None
    target_ids: Optional[torch.Tensor] = None
    target_ranks: Optional[Int[torch.Tensor, "... t"]] = None
    per_row_targets: bool = False
    applied_ln: bool = True
    applied_output_transform: bool = True

    @property
    def selected_ids(self) -> Optional[torch.Tensor]:
        """Token ids along the trailing axis of ``values`` when it is not the full vocabulary."""
        if self.target_ids is not None:
            return self.target_ids
        if self.vocab_ids is not None:
            return self.vocab_ids
        return self.top_ids

    def row_ids(self, index: Tuple[int, ...]) -> Optional[torch.Tensor]:
        """Trailing-axis token ids for one row of ``values`` (``None`` = full vocabulary)."""
        if self.top_ids is not None:
            return self.top_ids[index]
        if self.target_ids is not None and self.per_row_targets:
            return self.target_ids[index].reshape(1)
        return self.selected_ids

    def decode(self, tokenizer: Any) -> Any:
        """Decode the trailing-axis token ids to strings, keeping the leading structure.

        ``top_k`` and per-row readouts decode per row; shared ``targets`` /
        ``vocab`` readouts decode the id list once. Full-vocabulary readouts
        have no id list to decode.
        """
        decode = getattr(tokenizer, "decode", None)
        if not callable(decode):
            raise TypeError("tokenizer must provide a callable decode method")
        if self.top_ids is not None:
            return _decode_nested(self.top_ids, decode)
        if self.target_ids is not None and self.per_row_targets:
            return _decode_nested(self.target_ids, decode)
        ids = self.selected_ids
        if ids is None:
            raise ValueError(
                "decode needs a targets, vocab or top_k readout; this one is full-vocabulary"
            )
        return [str(decode([int(i)])) for i in ids.tolist()]


@dataclass
class LogitLensResult:
    """Result of :func:`logit_lens`.

    Attributes:
        readout: The :class:`LogitReadout` over the accumulated stack. Its
            leading axis is the stack entry, aligned with ``labels``; the
            remaining leading dims are ``[batch, pos]`` after the requested
            ``positions`` / ``batch_slice`` (an integer selection drops its axis).
        labels: Stack entry labels from ``accumulated_resid`` (``"0_pre"``,
            ``"3_mid"``, ``"final_post"``).
        has_batch_dim: Whether a batch axis follows the stack axis.
        target_names: For shared targets, the string passed for each target
            (``None`` where an id was passed), so :meth:`rank_trajectory`
            accepts the strings again.
    """

    readout: LogitReadout
    labels: List[str]
    has_batch_dim: bool
    target_names: Optional[List[Optional[str]]] = None

    @property
    def values(self) -> torch.Tensor:
        """Shortcut for ``readout.values``."""
        return self.readout.values

    def rank_trajectory(self, target: Union[str, int, None] = None) -> torch.Tensor:
        """Full-vocabulary rank of one target at every stack entry, ``[layer, ...]``.

        ``target`` is the string passed to ``targets`` or a token id; ``None``
        selects the only target when exactly one was requested, and is the only
        form for per-row targets.
        """
        ranks = self.readout.target_ranks
        ids = self.readout.target_ids
        if ranks is None or ids is None:
            raise ValueError("rank_trajectory needs a lens run with targets=...")
        if self.readout.per_row_targets:
            if target is not None:
                raise ValueError("per-row targets have one target per row; call rank_trajectory()")
            return ranks[..., 0]
        if target is None:
            if ids.numel() != 1:
                raise ValueError("pass target= when more than one target was requested")
            return ranks[..., 0]
        if isinstance(target, str):
            names = self.target_names or []
            if target not in names:
                known = [n for n in names if n is not None]
                raise ValueError(
                    f"unknown target {target!r}; string lookup needs the string as passed "
                    f"(known: {known}); pass the token id otherwise"
                )
            return ranks[..., names.index(target)]
        matches = (ids == int(target)).nonzero(as_tuple=True)[0]
        if matches.numel() == 0:
            raise ValueError(f"token id {target} was not among the requested targets")
        return ranks[..., int(matches[0])]

    def top_tokens(self, tokenizer: Any) -> Dict[str, Any]:
        """Decoded top-k tokens per stack entry, ``{label: nested lists}``."""
        if self.readout.top_ids is None:
            raise ValueError("top_tokens needs a lens run with top_k=...")
        return {
            label: _decode_nested(self.readout.top_ids[i], tokenizer.decode)
            for i, label in enumerate(self.labels)
        }

    def to_dataframe(self, tokenizer: Optional[Any] = None, top_k: Optional[int] = None) -> Any:
        """One row per (stack entry, batch, position, token) as a pandas DataFrame.

        Rows within a (layer, batch, pos) group are sorted by descending value and
        ``top_k`` keeps the leading ones. Full-vocabulary ``"logits"`` readouts
        go through :func:`transformer_lens.utilities.logits_utils.logits_to_df`
        (columns ``logit``, ``log_prob``, ``probability``); every other readout
        carries ``token_index``, optional ``token_string`` and one value column
        named like ``logits_to_df`` names it (``logit`` / ``log_prob`` /
        ``probability``).
        """
        import pandas as pd

        values = self.readout.values
        lead = tuple(values.shape[1:-1])
        axis_names = ("batch", "pos") if self.has_batch_dim else ("pos",)
        column = _VALUE_COLUMNS[self.readout.return_type]
        frames = []
        for i, label in enumerate(self.labels):
            for idx in itertools.product(*[range(n) for n in lead]):
                row = values[(i, *idx)].detach().cpu()
                row_ids = self.readout.row_ids((i, *idx))
                if row_ids is None and self.readout.return_type == "logits":
                    frame = logits_to_df(row, tokenizer, top_k)
                else:
                    ids = torch.arange(row.shape[-1]) if row_ids is None else row_ids.cpu()
                    order = torch.argsort(row, descending=True)
                    if top_k is not None:
                        order = order[:top_k]
                    frame = pd.DataFrame({"token_index": ids[order].tolist()})
                    if tokenizer is not None:
                        frame["token_string"] = [tokenizer.decode([t]) for t in ids[order].tolist()]
                    frame[column] = row[order].tolist()
                frame.insert(0, "layer", label)
                for axis, value in enumerate(idx):
                    name = axis_names[axis] if axis < len(axis_names) else f"dim{axis}"
                    frame.insert(1 + axis, name, value)
                frames.append(frame)
        return pd.concat(frames, ignore_index=True)


def _decode_nested(ids: torch.Tensor, decode: Callable[[List[int]], str]) -> Any:
    if ids.ndim == 0:
        return str(decode([int(ids)]))
    if ids.ndim == 1:
        return [str(decode([int(i)])) for i in ids.tolist()]
    return [_decode_nested(row, decode) for row in ids]


def _to_token_ids(model: Any, tokens: Union[str, int, Sequence[Union[str, int]]]) -> List[int]:
    """Resolve strings / ids into a list of single-token ids."""
    if isinstance(tokens, (str, int)):
        tokens = [tokens]
    ids: List[int] = []
    for token in tokens:
        if isinstance(token, str):
            ids.append(int(model.to_single_token(token)))
        elif isinstance(token, int) and not isinstance(token, bool):
            ids.append(token)
        else:
            raise TypeError(f"tokens must be strings or ints, got {type(token).__name__}")
    return ids


def _target_names(targets: Optional[TokenInput]) -> Optional[List[Optional[str]]]:
    """Positional record of the strings in a shared-targets argument."""
    if targets is None or isinstance(targets, torch.Tensor):
        return None
    if isinstance(targets, (str, int)):
        return [targets if isinstance(targets, str) else None]
    return [t if isinstance(t, str) else None for t in targets]


def _final_norm(model: Any) -> Optional[torch.nn.Module]:
    """The module to apply before unembedding, or ``None`` when the model has no final norm.

    Raises rather than guessing: an unfolded affine norm applied through the
    cached-scale path, or an unknown normalization type passed through
    untouched, would both read the stream wrong silently.
    """
    norm_type = getattr(model.cfg, "normalization_type", None)
    if norm_type is None:
        return None
    if norm_type not in _FINAL_NORM_TYPES:
        raise ValueError(
            f"logit readout does not know how to apply normalization_type={norm_type!r}; "
            "pass apply_ln=False to read the raw stream, which will not reproduce the model."
        )
    ln_final = getattr(model, "ln_final", None)
    if not isinstance(ln_final, torch.nn.Module):
        raise ValueError(
            f"{getattr(model.cfg, 'model_name', 'this model')} declares "
            f"normalization_type={norm_type!r} but exposes no `ln_final` component, so the "
            "final norm cannot be applied exactly. Add the alias to its adapter; "
            "apply_ln=False reads the raw stream instead, which only works when the "
            "unembedding consumes d_model vectors directly and does not reproduce the model."
        )
    return ln_final


def _require_decoder_unembed(model: Any) -> None:
    modules = getattr(model, "_modules", {})
    if "encoder_blocks" in modules or "decoder_blocks" in modules:
        raise NotImplementedError(
            "logit lens does not support encoder-decoder bridges yet; the decoder stack "
            "with cross-attention needs its own accumulated residual stream."
        )
    if getattr(model, "unembed", None) is None:
        raise NotImplementedError(
            "logit readout needs an `unembed` component; encoder-only bridges expose a "
            "task head instead of W_U."
        )


def _validate_selectors(
    model: Any,
    *,
    return_type: str,
    vocab: Optional[Union[Sequence[int], torch.Tensor]],
    targets: Optional[TokenInput],
    top_k: Optional[int],
    chunk_size: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    """Cheap argument checks shared by both entry points; run before any forward pass.

    Returns ``(vocab_ids, target_ids, d_vocab)`` where ``target_ids`` is a flat
    ``[n_targets]`` tensor for shared targets or the caller's tensor (long) for
    per-row targets.
    """
    if return_type not in _RETURN_TYPES:
        raise ValueError(f"return_type must be one of {_RETURN_TYPES}, got {return_type!r}")
    if sum(x is not None for x in (vocab, targets, top_k)) > 1:
        raise ValueError("pass at most one of vocab, targets and top_k")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if vocab is None and targets is None and top_k is None:
        return None, None, -1
    d_vocab = int(model.W_U.shape[1])  # read only when a selector needs the bound
    if top_k is not None and not 1 <= top_k <= d_vocab:
        raise ValueError(f"top_k must be in [1, {d_vocab}], got {top_k}")
    for tensor, name in ((vocab, "vocab"), (targets, "targets")):
        if isinstance(tensor, torch.Tensor) and tensor.dtype not in _INDEX_DTYPES:
            raise TypeError(f"{name} tensors must hold integer token ids, got dtype {tensor.dtype}")
    vocab_ids: Optional[torch.Tensor] = None
    if vocab is not None:
        vocab_ids = torch.as_tensor(vocab, dtype=torch.long).reshape(-1)
    target_ids: Optional[torch.Tensor] = None
    if isinstance(targets, torch.Tensor):
        target_ids = targets.reshape(1) if targets.ndim == 0 else targets
        target_ids = target_ids.to(torch.long)
    elif targets is not None:
        target_ids = torch.tensor(_to_token_ids(model, targets), dtype=torch.long)
    for ids, name in ((vocab_ids, "vocab"), (target_ids, "targets")):
        if ids is not None and (
            ids.numel() == 0 or int(ids.min()) < 0 or int(ids.max()) >= d_vocab
        ):
            raise ValueError(f"{name} ids must be within [0, {d_vocab})")
    return vocab_ids, target_ids, d_vocab


def _apply_output_transform(model: Any, logits: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    adapter = getattr(model, "adapter", None)
    transform = getattr(adapter, "apply_output_logits_transform", None)
    if not callable(transform):
        return logits, False
    return transform(logits), True


def _run_ln_final(
    ln_final: torch.nn.Module, flat: Float[torch.Tensor, "n d_model"]
) -> torch.Tensor:
    # Norm components contract on [batch, pos, d_model]; stats are per vector, so
    # the row axis can be presented as the position axis.
    return ln_final(flat.unsqueeze(0)).squeeze(0)


def _broadcast_targets(targets: torch.Tensor, lead: Tuple[int, ...]) -> torch.Tensor:
    """Right-align a per-row target tensor with the vectors' leading dims."""
    if targets.ndim > len(lead):
        raise ValueError(
            f"per-row targets of shape {tuple(targets.shape)} have more dims than the "
            f"vectors' leading dims {lead}"
        )
    shaped = targets.reshape((1,) * (len(lead) - targets.ndim) + tuple(targets.shape))
    try:
        return shaped.expand(*lead)
    except RuntimeError as err:
        raise ValueError(
            f"per-row targets of shape {tuple(targets.shape)} do not broadcast to the "
            f"vectors' leading dims {lead}; they are right-aligned, so a per-example "
            f"target over several positions must be shaped [batch, 1]"
        ) from err


def logit_readout(
    model: Any,
    vectors: Float[torch.Tensor, "... d_model"],
    *,
    apply_ln: bool = True,
    vocab: Optional[Union[Sequence[int], torch.Tensor]] = None,
    targets: Optional[TokenInput] = None,
    return_type: str = "logits",
    top_k: Optional[int] = None,
    chunk_size: int = 64,
    enable_grad: bool = False,
) -> LogitReadout:
    """Read ``d_model`` vectors through the model's final norm and unembedding.

    Computes ``transform(ln_final(x) @ W_U + b_U)`` for every vector, where
    ``transform`` is the adapter's post-unembedding transform, in chunks of
    ``chunk_size`` rows so at most ``chunk_size × d_vocab`` logits exist at once.
    Everything derived from the full vocabulary (the softmax normalizer,
    ``entropy``, ``target_ranks``) is computed per chunk and discarded; only the
    selected columns are kept.

    Args:
        model: A ``TransformerBridge`` in eval mode.
        vectors: Residual-stream-shaped vectors, any leading dims.
        apply_ln: Apply the model's ``ln_final`` module first (recomputed
            statistics per vector). ``False`` reads the raw vectors and does not
            reproduce the model.
        vocab: Token ids to keep along the trailing axis.
        targets: Tokens whose values and full-vocabulary ranks to report.
            Strings (single tokens) or ints, or a sequence of them, are
            *shared* across every row. A tensor is *per row*: it is
            right-aligned and broadcast to the vectors' leading dims (for
            ``[batch, pos, d_model]`` vectors pass ``[batch, pos]`` next-token
            ids, or ``[batch, 1]`` for one id per example) and the trailing
            axis of ``values`` has length 1. Note the convention differs from
            ``ActivationCache.logit_attrs``, which reads a 1-D tensor as
            per-example first; here alignment is purely right-to-left.
        return_type: ``"logits"``, ``"log_probs"`` or ``"probs"``.
        top_k: Keep the ``k`` largest-logit tokens per vector, with their ids.
        chunk_size: Rows of ``vectors`` processed per unembedding matmul.
        enable_grad: Keep autograd on (for trainable lenses and attribution);
            default runs under ``torch.no_grad()``.

    Returns:
        A :class:`LogitReadout`.

    Raises:
        ValueError: On a bad ``return_type``, more than one selector, a
            ``chunk_size`` below 1, a wrong trailing dimension, empty vectors,
            per-row targets that do not broadcast, or a model whose final norm
            cannot be applied exactly.
        NotImplementedError: For encoder-decoder and encoder-only bridges.
    """
    require_eval_mode(model, operation="logit_readout")
    _require_decoder_unembed(model)
    if not isinstance(vectors, torch.Tensor) or not vectors.is_floating_point():
        raise TypeError("vectors must be a floating-point tensor")
    ln_final = _final_norm(model) if apply_ln else None
    vocab_ids, target_ids, _ = _validate_selectors(
        model,
        return_type=return_type,
        vocab=vocab,
        targets=targets,
        top_k=top_k,
        chunk_size=chunk_size,
    )
    d_model = int(model.cfg.d_model)
    if vectors.shape[-1] != d_model:
        raise ValueError(f"vectors must end in d_model={d_model}, got shape {tuple(vectors.shape)}")
    lead = tuple(vectors.shape[:-1])
    if vectors.numel() == 0:
        raise ValueError(f"vectors has no rows to read (shape {tuple(vectors.shape)})")

    W_U = require_readable_weight(model.W_U, operation="read logits through W_U", owner=model)
    b_U = model.b_U

    per_row = isinstance(targets, torch.Tensor) and targets.ndim >= 1
    row_targets: Optional[torch.Tensor] = None
    if per_row:
        assert target_ids is not None
        target_ids = _broadcast_targets(target_ids, lead)
        row_targets = target_ids.reshape(-1).to(W_U.device)
    select = vocab_ids if vocab_ids is not None else (None if per_row else target_ids)
    select_dev = None if select is None else select.to(W_U.device)

    flat = vectors.reshape(-1, d_model).to(device=W_U.device, dtype=W_U.dtype)
    out_values: List[torch.Tensor] = []
    out_top_ids: List[torch.Tensor] = []
    out_ranks: List[torch.Tensor] = []
    out_entropy: List[torch.Tensor] = []
    applied_transform = False

    grad_ctx = torch.enable_grad() if enable_grad else torch.no_grad()
    with grad_ctx:
        for start in range(0, flat.shape[0], chunk_size):
            chunk = flat[start : start + chunk_size]
            if ln_final is not None:
                chunk = _run_ln_final(ln_final, chunk)
            logits = chunk @ W_U + b_U
            logits, applied_transform = _apply_output_transform(model, logits)
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            out_entropy.append(-(log_probs.exp() * log_probs).sum(dim=-1))
            if return_type == "logits":
                quantity = logits
            elif return_type == "log_probs":
                quantity = log_probs
            else:
                quantity = log_probs.exp()
            if row_targets is not None:
                rows = row_targets[start : start + chunk_size].unsqueeze(-1)  # [n, 1]
                target_logits = torch.gather(logits, -1, rows)
                out_ranks.append((logits > target_logits).sum(dim=-1, keepdim=True))
                out_values.append(torch.gather(quantity, -1, rows))
            elif target_ids is not None:
                assert select_dev is not None
                target_logits = logits[:, select_dev].unsqueeze(-1)  # [n, t, 1]
                out_ranks.append((logits.unsqueeze(1) > target_logits).sum(dim=-1))
                out_values.append(quantity[:, select_dev])
            elif top_k is not None:
                top = torch.topk(logits, k=top_k, dim=-1, sorted=True)
                out_top_ids.append(top.indices)
                out_values.append(torch.gather(quantity, -1, top.indices))
            elif select_dev is not None:
                out_values.append(quantity[:, select_dev])
            else:
                out_values.append(quantity)

    return LogitReadout(
        values=torch.cat(out_values, dim=0).reshape(*lead, -1),
        return_type=return_type,
        entropy=torch.cat(out_entropy, dim=0).reshape(lead),
        top_ids=torch.cat(out_top_ids, dim=0).reshape(*lead, -1) if out_top_ids else None,
        vocab_ids=vocab_ids,
        target_ids=target_ids,
        target_ranks=torch.cat(out_ranks, dim=0).reshape(*lead, -1) if out_ranks else None,
        per_row_targets=per_row,
        applied_ln=ln_final is not None,
        applied_output_transform=applied_transform,
    )


def _residual_hook_names(n_layers: int, incl_mid: bool) -> List[str]:
    names = []
    for layer in range(n_layers):
        names.append(f"blocks.{layer}.hook_resid_pre")
        if incl_mid:
            names.append(f"blocks.{layer}.hook_resid_mid")
    names.append(f"blocks.{n_layers - 1}.hook_resid_post")
    return names


def _resolve_layers(layers: Optional[Sequence[int]], labels: List[str]) -> List[int]:
    if layers is None:
        return list(range(len(labels)))
    resolved = []
    for layer in layers:
        index = layer + len(labels) if layer < 0 else layer
        if not 0 <= index < len(labels):
            raise ValueError(
                f"layer {layer} is out of range for the accumulated stack; valid entries are "
                f"0..{len(labels) - 1} = {labels}"
            )
        resolved.append(index)
    return resolved


def logit_lens(
    model: Union[Any, ActivationCache],
    input: Union[str, List[str], torch.Tensor, None] = None,
    *,
    cache: Optional[ActivationCache] = None,
    layers: Optional[Sequence[int]] = None,
    incl_mid: bool = False,
    positions: SliceInput = -1,
    batch_slice: SliceInput = None,
    lens: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
    apply_ln: bool = True,
    vocab: Optional[Union[Sequence[int], torch.Tensor]] = None,
    targets: Optional[TokenInput] = None,
    return_type: str = "logits",
    top_k: Optional[int] = None,
    chunk_size: int = 64,
    enable_grad: bool = False,
) -> LogitLensResult:
    """Read the accumulated residual stream through the final norm and unembedding.

    Builds the stack with
    :meth:`~transformer_lens.ActivationCache.ActivationCache.accumulated_resid`,
    optionally applies a per-entry ``lens`` translator, applies the final norm
    through :meth:`~transformer_lens.ActivationCache.ActivationCache.apply_ln_to_stack`
    with recomputed statistics, and hands the result to :func:`logit_readout`.
    The final stack entry reproduces the model's logits (raw bridge) or
    log-probabilities (compatibility mode).

    Args:
        model: A ``TransformerBridge``, or an :class:`ActivationCache` (then
            ``input`` must be omitted).
        input: Prompt(s) or token tensor to run when no ``cache`` is given. Only
            the residual-stream hooks are cached. A batch of strings is padded
            by the tokenizer, so with the default ``positions=-1`` the shorter
            prompts are read at a pad position; pass a token tensor (or
            per-prompt positions) when prompts differ in length.
        cache: An existing cache to read instead of running the model.
        layers: Indices into the accumulated stack (negative allowed); ``None``
            keeps every entry.
        incl_mid: Include ``resid_mid`` entries. Raises on hybrid stacks whose
            SSM blocks have no ``resid_mid``.
        positions: Position selection (``Slice`` semantics); default last token.
        batch_slice: Batch selection (``Slice`` semantics).
        lens: Optional ``f(entry, index) -> entry`` applied to each stack entry
            before the norm, e.g. a tuned-lens translator. Combine with
            ``enable_grad=True`` to fit or attribute through it.
        targets: As in :func:`logit_readout`; a per-row tensor is aligned with
            the stack's ``[batch, pos]`` dims *after* ``positions`` and
            ``batch_slice`` are applied (``[batch, pos]`` next-token ids with
            ``positions=None``, ``[batch]`` with the default last position).
        apply_ln, vocab, return_type, top_k, chunk_size, enable_grad: As in
            :func:`logit_readout`.

    Returns:
        A :class:`LogitLensResult`.
    """
    if isinstance(model, ActivationCache):
        if input is not None or cache is not None:
            raise ValueError("pass either a cache or a model with input, not both")
        cache = model
        model = cache.model
    require_eval_mode(model, operation="logit_lens")
    _require_decoder_unembed(model)
    # Everything cheap is checked before the forward pass is paid for.
    ln_final = _final_norm(model) if apply_ln else None
    _validate_selectors(
        model,
        return_type=return_type,
        vocab=vocab,
        targets=targets,
        top_k=top_k,
        chunk_size=chunk_size,
    )
    if incl_mid:
        layer_types = model.layer_types() if hasattr(model, "layer_types") else []
        hybrid = [t for t in layer_types if any(p in _HYBRID_VARIANT_NAMES for p in t.split("+"))]
        if hybrid:
            raise ValueError(
                f"incl_mid=True needs a resid_mid hook on every block, but this stack has "
                f"block types {sorted(set(hybrid))} without one; use incl_mid=False."
            )
    if cache is None:
        if input is None:
            raise ValueError("logit_lens needs an input to run or a cache to read")
        _, cache = model.run_with_cache(
            input, names_filter=_residual_hook_names(int(model.cfg.n_layers), incl_mid)
        )
    assert cache is not None

    pos_slice = positions if isinstance(positions, Slice) else Slice(positions)
    batch = batch_slice if isinstance(batch_slice, Slice) else Slice(batch_slice)
    stack, labels = cache.accumulated_resid(
        layer=None, incl_mid=incl_mid, apply_ln=False, pos_slice=pos_slice, return_labels=True
    )
    keep = _resolve_layers(layers, labels)
    stack = stack[keep]
    labels = [labels[i] for i in keep]
    if cache.has_batch_dim:
        stack = batch.apply(stack, dim=1)
    has_batch_dim = cache.has_batch_dim and batch.mode != "int"
    grad_ctx = torch.enable_grad() if enable_grad else torch.no_grad()
    with grad_ctx:
        if lens is not None:
            stack = torch.stack([lens(stack[i], keep[i]) for i in range(stack.shape[0])], dim=0)
        if ln_final is not None:
            # Recomputed-statistics path through the ln_final module (the #1076 fix).
            stack = cache.apply_ln_to_stack(
                stack,
                layer=None,
                pos_slice=pos_slice,
                has_batch_dim=has_batch_dim,
                recompute_ln=True,
            )
    readout = logit_readout(
        model,
        stack,
        apply_ln=False,
        vocab=vocab,
        targets=targets,
        return_type=return_type,
        top_k=top_k,
        chunk_size=chunk_size,
        enable_grad=enable_grad,
    )
    readout.applied_ln = ln_final is not None
    return LogitLensResult(
        readout=readout,
        labels=labels,
        has_batch_dim=has_batch_dim,
        target_names=_target_names(targets),
    )
