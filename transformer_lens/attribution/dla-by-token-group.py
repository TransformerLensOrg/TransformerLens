"""DLA by token group"""


from typing import List

import einops
import torch
from einops import einsum
from jaxtyping import Float
from transformer_lens import ActivationCache, HookedTransformer

from attention_head_superposition.utils.device import get_default_device

device = get_default_device()


def dla_by_token_group(
    cache: ActivationCache,
    model: HookedTransformer,
    tokenized_attributes,
    token_group_indices: List[List[int]],
    # bos_token_indices,
    # subject_token_indices,
    # relation_token_indices,
) -> Float[torch.Tensor, "attribute model_component token_group"]:
    """DLA by attention head, by token group that is attended to

    The token groups are (bos, subject, relation).
    """

    layers = []

    logit_directions: Float[
        torch.Tensor, "attribute d_model"
    ] = model.tokens_to_residual_directions(torch.tensor(tokenized_attributes))

    if logit_directions.ndim == 1:
        logit_directions = logit_directions.unsqueeze(0)

    for layer in range(model.cfg.n_layers):
        value: Float[torch.Tensor, "pos head_index d_head"] = cache[
            f"blocks.{layer}.attn.hook_v"
        ][0]

        pattern_post_softmax: Float[
            torch.Tensor, "head_index query_pos key_pos"
        ] = cache[f"blocks.{layer}.attn.hook_pattern"][0]

        # Z is usually calculated as values * attention, summed across keys
        z = einsum(
            value,
            pattern_post_softmax,
            "key_pos head_index d_head, \
                head_index query_pos key_pos -> \
                query_pos key_pos head_index d_head",
        )

        weights_output: Float[torch.Tensor, "head_index d_head d_model"] = model.blocks[
            layer
        ].attn.W_O

        # bias_output: Float[torch.Tensor, "d_model"] = model.blocks[layer].attn.b_O

        result = einsum(
            z,
            weights_output,
            "query_pos key_pos head_index d_head, \
                    head_index d_head d_model -> \
                    query_pos key_pos head_index d_model",
        )

        # Stacked head result (pos slice on last token)
        result_on_final_token: Float[
            torch.Tensor, "key_pos head_index d_model"
        ] = result[-1]

        # Scale
        if model.cfg.normalization_type not in ["LN", "LNPre"]:
            scaled_result = result_on_final_token
        else:
            center_stack = result_on_final_token - result_on_final_token.mean(
                dim=-1, keepdim=True
            )
            scale = cache["ln_final.hook_scale"][0, -1, :]  # first batch, last token
            scaled_result = center_stack / scale

        logit_attrs = einsum(
            scaled_result,
            logit_directions,
            "key_pos head_index d_model, attribute d_model -> key_pos head_index attribute",
        )

        logit_attrs = einops.rearrange(
            logit_attrs, "key_pos head_index attribute -> attribute head_index key_pos"
        )

        # Sum by category (bos, subject, relation)
        token_group_attribution = []

        for token_group in token_group_indices:
            attribution: Float[torch.Tensor, "attribute head_index"] = logit_attrs[
                :, :, token_group
            ].sum(dim=-1)
            token_group_attribution.append(attribution)

        combined: Float[torch.Tensor, "attribute head_index token_group"] = torch.stack(
            token_group_attribution,
            dim=-1,
        )
        layers.append(combined)

    layers_tensor: Float[
        torch.Tensor, "layer attribute head_index token_group"
    ] = torch.stack(layers, dim=0)

    rearranged = einops.rearrange(
        layers_tensor,
        "layer attribute head_index token_group -> attribute (layer head_index) token_group",
    )

    return rearranged
