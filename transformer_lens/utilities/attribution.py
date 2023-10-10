"""Attribution."""
from typing import Optional

from einops import einsum
from jaxtyping import Float
from torch import Tensor


def direct_attribution(
    residual_stack: Float[Tensor, "component *batch *pos d_model"],
    tokens_residual_directions: Float[Tensor, "token d_model"],
    layer_norm_scale: Optional[Float[Tensor, "*batch *pos d_model"]] = None,
) -> Float[Tensor, "component *batch *pos token"]:
    """Attribution to Logits (DLA) or Other Neurons
    
    Can be used for direct logit attribution (DLA), or attribution to any component's neurons.
    
    Gotya: Don't set `apply_ln` as True when getting the residual stack from the cache, if you're
    doing DLA.
    
    TODO: clean this: The logits of a model are logits=Unembed(LayerNorm(final_residual_stream)).
    The Unembed is a linear map, and LayerNorm is approximately a linear map, so we can decompose
    the logits into the sum of the contributions of each component, and look at which components
    contribute the most to the logit of the correct token! This is called direct logit attribution.
    Here we look at the direct attribution to the logit difference!

    TODO: explain logit differences
    
    TODO: Explain neuron attribution for a different component
    
    TODO: Add example for neuron attribution for a different component
    
    Example for DLA:
    
    >>> from transformer_lens import HookedTransformer
    >>> from transformer_lens.utilities.logit_attribution import logit_attribution
    >>>
    >>> model = HookedTransformer.from_pretrained("attn-only-1l", device="cpu")
    >>> logits, cache = model.run_with_cache("My name is")
    >>> residual_stack, labels = cache.get_full_resid_decomposition(return_labels=True)
    >>>
    >>> answers = [" George", " John"] # Note these must be single token strings
    >>> answer_tokens = model.to_tokens(answers, prepend_bos=False)
    >>> answer_directions = model.tokens_to_residual_directions(answer_tokens)
    >>>
    >>> # Final layer norm is used for DLA
    >>> final_ln = cache["ln_final.hook_scale"][:, -1, :]  # -1 for last position
    >>>
    >>> attribution = logit_attribution(residual_stack, answer_directions, final_ln)
    >>> print(attribution[0].round().to_list()) # Print DLA for each answer, to 2dp
    [12, 14]
    
    Args:
        residual_stack: Stack of components of residual stream to get logit attributions for. This
            is usually obtained from `cache.get_full_resid_decomposition()`. Note that as the
            components are typically post layer norm, you should not set the argument `apply_ln` to
            `True`. Note also we assume the model has been run with folded layer norms.
        tokens_residual_directions: The residual directions for each token. This is usually obtained
            from `model.tokens_to_residual_directions()`.
        layer_norm_scale: The layer norm scale to apply. For DLA this should be
            `ln_final`. For example if you are conducting DLA on the final position, this could be
            `cache["ln_final.hook_scale"][:, -1, :]`.
    
    Returns:
        Attribution of each component to each token.
    """
    # Apply the layer norm
    if layer_norm_scale:
        centered_residual_stack = residual_stack - residual_stack.mean(dim=-1, keepdim=True)
        scaled_stack = centered_residual_stack / layer_norm_scale
    else:
        scaled_stack = residual_stack

    # Project the residual stack onto the residual directions
    return einsum(
        scaled_stack,
        tokens_residual_directions,
        "component ... d_model, token d_model -> component ... token",
