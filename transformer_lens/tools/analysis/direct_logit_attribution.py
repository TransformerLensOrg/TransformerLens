"""Direct Logit Attribution.

Decomposes a model's logit difference into per-component contributions from
the residual stream. See :func:`dla` for usage.
"""

from typing import List, Tuple

import einops
import torch
from jaxtyping import Float, Int

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.utilities import get_act_name

# Block variants that lack the attn_out + mlp_out structure decompose_resid expects.
_HYBRID_VARIANT_NAMES = ("mamba", "ssm", "mixer", "linear_attn")


def dla(
    bridge: TransformerBridge,
    prompts: List[str],
    answer_tokens: Int[torch.Tensor, "batch answers"],
    accumulated: bool = False,
) -> Tuple[Float[torch.Tensor, "component"], List[str]]:
    """Compute Direct Logit Attribution for a TransformerBridge model.

    Decomposes the logit (or logit difference between a correct and wrong token)
    into per-component contributions from the residual stream, averaged across
    the batch of prompts. Two modes:

    * ``accumulated=False`` (default): the contribution of each individual
      component — embedding, per-layer attention output, per-layer MLP output —
      via :meth:`transformer_lens.ActivationCache.decompose_resid`. These are
      additive: their sum reconstructs the (bias-excluded) logit difference.
    * ``accumulated=True``: the cumulative residual stream at each layer boundary
      (logit-lens style) via
      :meth:`transformer_lens.ActivationCache.accumulated_resid`. These are
      cumulative, so the full reconstruction is the *last* entry, not the sum.

    Warning:

    Returned scores sum (decompose mode) or end (accumulated mode) at
    ``actual_logit_diff - (b_U[correct] - b_U[wrong])``, not ``actual_logit_diff``
    directly. The unembedding bias :math:`b_U` is a constant offset added after
    unembedding rather than a residual-stream contribution, so it is excluded by
    convention — matching :meth:`transformer_lens.ActivationCache.logit_attrs`.

    Warning:

    This function requires the bridge to be in compatibility mode so the final
    LayerNorm weights are folded into :math:`W_U`. Without folding, the
    projection direction is wrong and the scores do not reflect actual logit
    contributions. Call ``bridge.enable_compatibility_mode()`` after loading.

    Warning:

    Hybrid architectures (Mamba, SSM, Mixer, LinearAttention) are not yet
    supported and raise :class:`NotImplementedError`; support requires extending
    :meth:`transformer_lens.ActivationCache.decompose_resid` to handle
    non-attention blocks.

    Args:
        bridge:
            A :class:`transformer_lens.model_bridge.TransformerBridge` with
            compatibility mode enabled.
        prompts:
            Prompt strings to evaluate. Length must match ``answer_tokens.shape[0]``.
        answer_tokens:
            Tensor of shape ``(batch, answers)``. ``answers == 1`` decomposes the
            single target token's logit; ``answers == 2`` treats each row as a
            ``(correct, wrong)`` pair and decomposes the logit difference.
        accumulated:
            If ``True``, return cumulative per-layer contributions (logit lens).
            If ``False`` (default), return per-component contributions.

    Returns:
        ``(scores, labels)``: ``scores`` is a 1D tensor of per-component (or
        per-layer) contributions averaged across the batch, and ``labels`` is the
        matching list of human-readable component names.

    Raises:
        ValueError: If ``len(prompts)`` does not match ``answer_tokens.shape[0]``,
            if ``answer_tokens.shape[1]`` is not ``1`` or ``2``, or if the bridge
            is not in compatibility mode.
        NotImplementedError: If the bridge contains hybrid (non attention + MLP)
            blocks.
    """

    # input validation
    if len(prompts) != answer_tokens.shape[0]:
        raise ValueError(
            "Each prompt needs a matching row of answer tokens: got "
            f"{len(prompts)} prompts but {answer_tokens.shape[0]} answer-token rows."
        )
    if answer_tokens.shape[1] not in (1, 2):
        raise ValueError(
            "answer_tokens must have 1 (single token) or 2 (correct, wrong) columns, "
            f"got {answer_tokens.shape[1]}."
        )

    # safeguard: DLA needs LayerNorm folded into W_U, which compatibility mode does
    if not getattr(bridge, "compatibility_mode", False):
        raise ValueError(
            "DLA requires the bridge to be in compatibility mode so that LayerNorm "
            "weights are folded into W_U. Call `bridge.enable_compatibility_mode()` "
            "after loading the bridge, then re-run DLA."
        )

    # safeguard: hybrid blocks (Mamba/SSM/...) have no attn_out + mlp_out to decompose
    hybrid_blocks = [
        layer_type
        for layer_type in bridge.layer_types()
        if any(part in _HYBRID_VARIANT_NAMES for part in layer_type.split("+"))
    ]
    if hybrid_blocks:
        raise NotImplementedError(
            f"DLA does not yet support hybrid architectures (found block types "
            f"{hybrid_blocks}). Only standard attention + MLP transformers (e.g. "
            f"GPT-2, LLaMA, Pythia) are supported; hybrid support requires extending "
            f"ActivationCache.decompose_resid."
        )

    # unembedding direction per prompt; single-token tensors collapse to [d_model]
    answer_residual_directions = bridge.tokens_to_residual_directions(answer_tokens).reshape(
        answer_tokens.shape[0], answer_tokens.shape[1], -1
    )
    if answer_tokens.shape[1] == 1:
        logit_diff_directions: Float[torch.Tensor, "batch d_model"] = answer_residual_directions[
            :, 0, :
        ]
    else:  # (correct, wrong) pair -> direction of the logit difference
        correct_direction, wrong_direction = answer_residual_directions.unbind(dim=1)
        logit_diff_directions = correct_direction - wrong_direction

    def residual_stack_to_logit_diff(
        residual_stack: Float[torch.Tensor, "... batch d_model"],
        cache: ActivationCache,
    ) -> Float[torch.Tensor, "..."]:
        # apply the final LayerNorm once, project onto the answer direction, average over prompts
        batch_size = residual_stack.size(-2)
        scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
        return (
            einops.einsum(
                scaled_residual_stack,
                logit_diff_directions,
                "... batch d_model, batch d_model -> ...",
            )
            / batch_size
        )

    if accumulated:
        n_layers = bridge.cfg.n_layers
        _, cache = bridge.run_with_cache(
            prompts,
            names_filter=lambda name: name == get_act_name("resid_post", n_layers - 1)
            or name == "ln_final.hook_scale"
            or name.endswith("resid_pre")
            or name.endswith("resid_mid"),
        )
        accumulated_residual, labels = cache.accumulated_resid(
            layer=-1, pos_slice=-1, incl_mid=True, return_labels=True
        )
        return residual_stack_to_logit_diff(accumulated_residual, cache), labels

    _, cache = bridge.run_with_cache(
        prompts,
        names_filter=lambda name: name == "ln_final.hook_scale"
        or name in ("hook_embed", "hook_pos_embed")
        or name.endswith("attn_out")
        or name.endswith("mlp_out"),
    )
    per_component_residual, labels = cache.decompose_resid(
        layer=-1, pos_slice=-1, return_labels=True
    )
    return residual_stack_to_logit_diff(per_component_residual, cache), labels
