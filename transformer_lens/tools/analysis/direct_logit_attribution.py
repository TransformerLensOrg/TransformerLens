"""Direct Logit Attribution.

Decomposes a model's logit difference into per-component contributions from
the residual stream. See :func:`DLA` for usage.
"""

from typing import List, Tuple

import einops
import torch
from jaxtyping import Float, Int

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.utils import get_act_name

# Variant submodule names that indicate a hybrid block. Mamba, SSM, Mixer, and LinearAttention layers don't have the usual attn_out / mlp_out decomposition that ActivationCache.decompose_resid expects.
_HYBRID_VARIANT_NAMES = ("mamba", "ssm", "mixer", "linear_attn")
dont 

def DLA(
    bridge: TransformerBridge,
    prompts: List[str],
    answer_tokens: Int[torch.Tensor, "batch answers"],
    accumulated: bool = False,
) -> Tuple[Float[torch.Tensor, "component"], List[str]]:
    """Compute Direct Logit Attribution For A TransformerBridge Model.

    Decomposes the logit difference between a correct and wrong answer token
    into per-component contributions from the residual stream. Two modes:

    * ``accumulated=False`` (default): returns the contribution of each
      individual component (embedding, per-layer attention output, per-layer
      MLP output) — see :meth:`transformer_lens.ActivationCache.decompose_resid`.
    * ``accumulated=True``: returns the cumulative residual stream contribution
      at each layer boundary (logit-lens style) — see
      :meth:`transformer_lens.ActivationCache.accumulated_resid`.

    Warning:

    Returned scores sum to ``actual_logit_diff - (b_U[correct] - b_U[wrong])``,
    not to ``actual_logit_diff`` directly. The unembedding bias :math:`b_U` is
    a constant offset added after unembedding, not a contribution from the
    residual stream, so it is excluded by convention (matching the behavior of
    :meth:`transformer_lens.ActivationCache.decompose_resid`).

    Warning:

    This function requires the bridge to be in compatibility mode so that the
    final LayerNorm weights are folded into :math:`W_U`. Without folding, the
    projection direction is incorrect and per-component scores will not reflect
    actual logit contributions. Call ``bridge.enable_compatibility_mode()``
    after loading the bridge.

    Warning:

    Hybrid architectures (Mamba, SSM, Mixer, LinearAttention) are not yet
    supported and will raise :class:`NotImplementedError`. Support requires
    extending :meth:`transformer_lens.ActivationCache.decompose_resid` to handle
    non-attention block variants, which is tracked separately.

    Args:
        bridge:
            A :class:`transformer_lens.model_bridge.TransformerBridge` with
            compatibility mode enabled.
        prompts:
            List of prompt strings to evaluate. Length must match
            ``answer_tokens.shape[0]``.
        answer_tokens:
            Tensor of shape ``(batch, answers)``. When ``answers == 1``, treated
            as a single target token per prompt. When ``answers == 2``, treated
            as a ``(correct, wrong)`` pair per prompt and the logit difference
            ``correct - wrong`` is decomposed.
        accumulated:
            If ``True``, return cumulative residual stream contributions at each
            layer boundary (logit lens). If ``False``, return per-component
            contributions. Defaults to ``False``.

    Returns:
        A tuple ``(scores, labels)`` where ``scores`` is a 1D tensor of per-
        component or per-layer contributions (averaged across the batch), and
        ``labels`` is the matching list of human-readable component names.

    Raises:
        ValueError: If the bridge is not in compatibility mode.
        NotImplementedError: If the bridge contains hybrid (non attention + MLP)
            block variants.
        AssertionError: If ``len(prompts)`` does not match
            ``answer_tokens.shape[0]``, or if ``answer_tokens.shape[1]`` is not
            ``1`` or ``2``.
    """

    assert len(prompts) == answer_tokens.shape[0]
    assert answer_tokens.shape[1] == 1 or answer_tokens.shape[1] == 2

    # DLA requires the LayerNorm gamma to be folded into W_U. Without folding,
    # projections onto W_U use the wrong direction (some features under-weighted,
    # others over-weighted) and the per-component scores will not sum to the
    # actual logit difference. Compatibility mode applies fold_ln by default
    if not getattr(bridge, "compatibility_mode", False):
        raise ValueError(
            "DLA requires bridge to be in compatibility mode so that "
            "LayerNorm weights are folded into W_U. Call "
            "`bridge.enable_compatibility_mode()` after loading the bridge, "
            "then re-run DLA."
        )

    # Strict mode: hybrid architectures (Mamba, SSM, etc.) aren't supported yet
    # because ActivationCache.decompose_resid only knows how to decompose into
    # attn_out + mlp_out per layer. A model with Mamba blocks would silently
    # under-attribute. Raise here so callers fail loudly instead of getting
    # wrong numbers.
    for module_name, _ in bridge.named_modules():
        lowered = module_name.lower()
        if any(variant in lowered for variant in _HYBRID_VARIANT_NAMES):
            raise NotImplementedError(
                f"DLA does not yet support hybrid architectures "
                f"(found component {module_name!r}). Currently only standard "
                f"attention + MLP transformers (e.g. GPT-2, LLaMA, Pythia) are "
                f"supported. Hybrid support requires extending "
                f"ActivationCache.decompose_resid — tracked separately."
            )

    #grab residiual directions from bridge (essentially unembedding matrix transposed)
    answer_residual_directions: Float[torch.Tensor, "batch answers d_model"] = bridge.tokens_to_residual_directions(answer_tokens)

    #ensure all residual directions are of shape [batch, d_model]
    if answer_tokens.numel() == 1:
        logit_diff_directions: Float[torch.Tensor, "batch d_model"] = torch.unsqueeze(answer_residual_directions, dim=0)
    elif answer_residual_directions.shape[1] == 1:
        #strip middle dimension
        logit_diff_directions: Float[torch.Tensor, "batch d_model"] = answer_residual_directions[:, 0, :]
    else: #case where we have correct tokens and incorrect tokens
        correct_token_direction, incorrect_token_direction = answer_residual_directions.unbind(dim=1)
        logit_diff_directions: Float[torch.Tensor, "batch d_model"] = (
            correct_token_direction - incorrect_token_direction
        )

#turns residual stream contributions into logit-difference scores accounting for layerNorm
    def residual_stack_to_logit_diff(
        residual_stack: Float[torch.Tensor, "... batch d_model"],
        cache: ActivationCache,
        logit_diff_directions: Float[torch.Tensor, "batch d_model"],
    ) -> Float[torch.Tensor, "..."]:
        batch_size = residual_stack.size(-2)
        scaled_residual_stack = cache.apply_ln_to_stack(
            residual_stack, layer=-1, pos_slice=-1
        )
        return (
            #average logit-difference contribution across prompts
            einops.einsum(
                scaled_residual_stack,
                logit_diff_directions,
                "... batch d_model, batch d_model -> ...",
            )
            / batch_size
        )

    
    if accumulated:
        n_layers = bridge.cfg.n_layers

        #filtered residual stream cache
        _, cache = bridge.run_with_cache(prompts,
            return_type=None,
            names_filter=lambda x: x == get_act_name("resid_post", n_layers - 1)
            or x == get_act_name("ln_final.hook_scale")
            or x.endswith("resid_pre")
            or x.endswith("resid_mid"),
        )
        #stack stream into tensor
        accumulated_residual, labels = cache.accumulated_resid(
            layer=-1, pos_slice=-1, incl_mid=True, return_labels=True
        )
    
        logit_lens_logit_diffs: Float[
            torch.Tensor, "component"
        ] = residual_stack_to_logit_diff(
            accumulated_residual, cache, logit_diff_directions
        )

        return logit_lens_logit_diffs, labels

    else:
        _, cache = bridge.run_with_cache(
            prompts,
            return_type=None,
            names_filter=lambda x: x == get_act_name("ln_final.hook_scale")
            or x.endswith("embed")
            or x.endswith("attn_out")
            or x.endswith("mlp_out"),
        )

        per_layer_residual, labels = cache.decompose_resid(
            layer=-1, pos_slice=-1, return_labels=True
        )
        per_layer_logit_diffs: Float[
            torch.Tensor, "component"
        ] = residual_stack_to_logit_diff(
            per_layer_residual, cache, logit_diff_directions
        )

        return per_layer_logit_diffs, labels
