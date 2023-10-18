"""Direct Logit Attribution (DLA)."""
from typing import Union

from einops import einsum
from jaxtyping import Float, Int
from torch import Tensor, tensor

from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.components import LayerNorm, LayerNormPre, RMSNorm, RMSNormPre


def direct_logit_attribution(
    cache: ActivationCache,
    model: HookedTransformer,
    residual_decomposition: Float[Tensor, "*other d_model"],
    tokens: Int[Tensor, "tokens"],
) -> Float[Tensor, "component *batch_and_pos  token"]:
    """Direct Logit Attribution (DLA).

    Examples:
        >>> from torch import Tensor
        >>> from transformer_lens import HookedTransformer
        >>> from jaxtyping import Float
        >>>
        >>> # Run a forward pass & get the residual decomposition
        >>> model = HookedTransformer.from_pretrained("tiny-stories-instruct-1M")
        Loaded pretrained model tiny-stories-instruct-1M into HookedTransformer
        >>> _logits, cache = model.run_with_cache("Why did the elephant cross the")
        >>> answer = model.to_tokens(" road", prepend_bos=False)[0]
        >>> residual_decomposition, labels = cache.decompose_resid(
        ...     return_labels=True, incl_embeds=False
        ... )
        >>>
        >>> # Get the DLA to the answer token
        >>> attribution: Float[Tensor, "component batch pos token"] = direct_logit_attribution(
        ...    cache, model, residual_decomposition, answer
        ... )
        >>> last_token_attribution = attribution[:,0,-1,0] #
        >>> top_component = last_token_attribution.argmax()
        >>> print(f"Top Component: {labels[top_component]}")
        Top Component: 7_mlp_out

    Args:
        cache: Activation cache.
        model: Transformer model.
        residual_decomposition: Residual decomposition of the model. For example this can be the
            output to the residual stream of every MLP layer in the model, for a specific prompt.
        tokens: Tokens to compute attribution for (these can be e.g. a correct and incorrect answer
            to the prompt). Tokens are given as their indices in the vocabulary.

    Returns:
        Attribution of each component in the residual decomposition, to each token.

    References:

    """
    # Map output tokens to residual directions, using the unembedding matrix.
    tokens_residual_directions: Float[
        Tensor, "token d_model"
    ] = model.tokens_to_residual_directions(tokens)

    # Note if there is only one token, `tokens_to_residual_directions` skips the token dimension. To
    # keep this, we manually fix here.
    if tokens_residual_directions.ndim == 1:
        tokens_residual_directions = tokens_residual_directions.unsqueeze(0)

    # Apply Final Layer Norm
    # Note that whilst most parts of LayerNormPre are linear transformations, the scale part
    # (dividing by the norm) is not. In practice however, we can mostly get aware with ignoring this
    # and treating the scaling factor as a constant, since it does apply across the entire residual
    # stream for each token - this makes it a "global" property of the model's calculation, so for
    # any specific question it hopefully doesn't matter that much. But when you're considering a
    # sufficiently important circuit that it's a good fraction of the norm of the residual stream,
    # it's probably worth thinking about it more.
    def patch_with_cache_hook(_activations, hook):
        """Patch an Activation With it's Value in the Cache."""
        return cache[hook.name]

    ln_final: Union[RMSNorm, RMSNormPre, LayerNorm, LayerNormPre] = model.ln_final
    ln_final_hook = ln_final.hook_scale
    ln_final_hook.add_hook(patch_with_cache_hook)
    result_post_ln_final = ln_final.forward(residual_decomposition)
    ln_final_hook.remove_hooks()

    # Project the residual stack onto the token residual directions.
    return einsum(
        result_post_ln_final,
        tokens_residual_directions,
        "component ... d_model, token d_model -> component ... token",
    )
