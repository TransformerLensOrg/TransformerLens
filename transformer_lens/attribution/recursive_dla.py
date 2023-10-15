"""DLA by source token"""


import torch
from einops import einsum, rearrange
from jaxtyping import Float, Int
from torch import Tensor, cat, stack

from transformer_lens import ActivationCache, HookedTransformer


def dla_mlp_breakdown_source_component(
    cache: ActivationCache, model: HookedTransformer, tokens: Int[Tensor, "token"]
) -> Float[Tensor, "batch token dest_l src_comp"]:
    """Direct Logit Attribution (DLA) Breakdown of MLP Layers by Source Component."""
    pass


def dla_attn_head_breakdown_source_component(
    cache: ActivationCache, model: HookedTransformer, tokens
) -> Float[Tensor, "batch token dest_l dest_h src_pos src_comp"]:
    """Direct Logit Attribution (DLA) Breakdown of Attention Heads by Source Component.

    Gets the DLA breakdown backwards through the last layer's attention heads -> source tokens ->
    source components. Note this is only an OV breakdown and not a QK breakdown.

    Args:
        cache: Activation Cache
        model: Hooked Transformer
        tokens: Single next tokens (answers) to get the DLA breakdown for.

    Returns:
        DLA breakdown by source components. Note you can sum over the different dimensions to get a
        less granular breakdown. For example if you sum over the last source component dimension you
        get dla breakdown by source token instead.
    """

    layers = []

    logit_directions: Float[
        Tensor, "token d_model"
    ] = model.tokens_to_residual_directions(Tensor(tokens))

    if logit_directions.ndim == 1:
        logit_directions = logit_directions.unsqueeze(0)

    for dest_l in range(model.cfg.n_layers):
        # Note we need to keep the dimensions the same for all destination layers (even though the
        # first destination layer only looks at the source embed + pos encoding, whereas the last
        # one also looks at n-1 layers). To solve this we just add some zeros to fill to the largest
        # dimension size.
        source_residuals: Float[
            Tensor, "src_comp batch src_pos d_model"
        ] = cache.decompose_resid(layer=dest_l)
        extra_empty_needed = model.cfg.n_layers - dest_l
        if extra_empty_needed > 0:
            empty_residuals = torch.zeros(
                (
                    extra_empty_needed,
                    source_residuals.shape[-3],
                    source_residuals.shape[-2],
                    source_residuals.shape[-1],
                ),
                device=source_residuals.device,
            )
            source_residuals = cat((source_residuals, empty_residuals), dim=0)

        norm_scale: Float[Tensor, "batch src_pos 1"] = (
            1 / cache[f"blocks.{dest_l}.ln1.hook_scale"]
        )
        centered: Float[
            Tensor, "src_comp batch src_pos d_model"
        ] = source_residuals - source_residuals.mean(dim=-1, keepdim=True)
        value_input = einsum(
            centered,
            norm_scale.squeeze(-1),
            "src_comp batch src_pos d_model, \
                batch src_pos -> \
                batch src_comp src_pos d_model",
        )

        W_V: Float[Tensor, "dest_h d_model d_head"] = model.state_dict()[
            f"blocks.{dest_l}.attn.W_V"
        ]

        # b_V = model.state_dict()[f"blocks.{dest_l}.attn.b_V"]

        value: Float[Tensor, "src_pos src_comp dest_h d_head"] = (
            einsum(
                value_input,
                W_V,
                "batch src_comp src_pos d_model, \
                    dest_h d_model d_head -> \
                    batch src_comp src_pos dest_h d_head",
            )
            # + b_V
        )

        pattern_post_softmax: Float[Tensor, "batch dest_h dest_pos src_pos"] = cache[
            f"blocks.{dest_l}.attn.hook_pattern"
        ]

        # Z is usually calculated as values * attention, summed across keys
        z = einsum(
            value,
            pattern_post_softmax,
            "batch src_comp src_pos dest_h d_head, \
                batch dest_h dest_pos src_pos -> \
                batch dest_pos src_comp src_pos dest_h d_head",
        )

        weights_output: Float[Tensor, "dest_h d_head d_model"] = model.blocks[
            dest_l
        ].attn.W_O

        # bias_output: Float[Tensor, "d_model"] = model.blocks[dest_l].attn.b_O

        result = einsum(
            z,
            weights_output,
            "batch dest_pos src_comp src_pos dest_h d_head, \
                    dest_h d_head d_model -> \
                    batch dest_pos src_comp src_pos dest_h d_model",
        )

        # Stacked head result (pos slice on last token)
        result_on_final_token: Float[
            Tensor, "batch src_comp src_pos dest_h d_model"
        ] = result[:, -1]

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
            "batch src_comp src_pos dest_h d_model, \
                token d_model -> \
                batch src_comp src_pos dest_h token",
        )

        layers.append(logit_attrs)

    layers_tensor: Float[Tensor, "dest_l atch src_comp src_pos dest_h token"] = stack(
        layers, dim=0
    )

    rearranged = rearrange(
        layers_tensor,
        "dest_l batch src_comp src_pos dest_h token -> \
            token batch (dest_l dest_h) src_pos src_comp",
    )

    return rearranged
