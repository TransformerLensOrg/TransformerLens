"""DLA by source token"""
import torch
from einops import einsum, rearrange
from jaxtyping import Float, Int
from torch import Tensor

from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.components import Attention, LayerNormPre


def expand_tensor_dimension(
    input: Tensor, expand_dimension: int, expand_to_size: int, expand_value: float = 0
) -> Tensor:
    """Expand a Tensor Along a Specific Dimension.

    Args:
        input: The input tensor to be expanded.
        expand_dimension: The dimension to expand (can be negative for indexing from the end of the
            input tensor).
        expand_to_size: The size that the input tensors expansion dimension should be resized to.
        expand_value: The value to assign to the expansion space.

    Returns:
        The input tensor expanded along a specified dimension.
    """
    # Calculate the expansion space size (along the expansion dimension)
    current_size = input.shape[expand_dimension]
    expand_by = expand_to_size - current_size

    # Check the amount to expand by is positive
    if expand_by < 0:
        raise AttributeError(
            f"Expansion to size {expand_to_size} not possible. "
            + f"Dimension {expand_dimension} is already of size {current_size}."
        )

    # Just return he tensor if it's the correct size
    if expand_by == 0:
        return input

    # Otherwise expand
    expand_space: Tensor = torch.empty(
        # Keep the other dimensions the same, and set the expand dimension size to be the amount
        # needed to expand by
        [
            *input.shape[:expand_dimension],
            expand_by,
            *input.shape[expand_dimension + 1 :],
        ],
        device=input.device,
    )

    expand_space = torch.fill(expand_space, expand_value)
    return torch.cat((input, expand_space), dim=expand_dimension)


def dla_mlp_breakdown_source_component(
    cache: ActivationCache, model: HookedTransformer, tokens: Int[Tensor, "token"]
) -> Float[Tensor, "token batch dest_l src_comp"]:
    """Direct Logit Attribution (DLA) Breakdown of MLP Layers by Source Component.

    Gets the DLA breakdown backwards through a token's MLP layers, to the source components
    (earlier components and the embedding/pos encoding).

    TODO: Fix zeroing (doesn't work as with attn heads due to 2xmlp so needs to be done post-layer.

    Args:
        cache: Activation Cache
        model: Hooked Transformer
        tokens: Single next tokens (answers) to get the DLA breakdown for.

    Returns:
        DLA breakdown by source components.
    """

    layers = []

    # Get the logit directions
    logit_directions: Float[
        Tensor, "token d_model"
    ] = model.tokens_to_residual_directions(Tensor(tokens))

    if logit_directions.ndim == 1:
        logit_directions = logit_directions.unsqueeze(0)

    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()

    # For each destination layer, get the sum of all source levels
    for dest_l in range(model.cfg.n_layers):
        # Note we need to keep the dimensions the same for all destination layers (even though the
        # first destination layer only looks at the source embed + pos encoding, whereas the last
        # one also looks at n-1 layers). To solve this we just add some zeros to fill to the largest
        # dimension size.
        source_residuals: Float[
            Tensor, "src_comp batch d_model"
        ] = cache.decompose_resid(layer=dest_l, pos_slice=-1)

        max_source_components = model.cfg.n_layers - dest_l
        if not model.cfg.attn_only:
            max_source_components *= 2

        # Expand across the source components dimension to the max number of components, for easy
        # concatenation of all destination components later.
        source_residuals = expand_tensor_dimension(
            source_residuals, 0, max_source_components, 0.0
        )

        # Apply LN to the stack
        ln_2: LayerNormPre = model.blocks[dest_l].ln2
        ln_2_hook = ln_2.hook_scale
        ln_2_hook_handle = ln_2_hook.add_hook(forward_cache_hook)
        mlp_input: Float[Tensor, "src_comp batch d_model"] = ln_2.forward(
            source_residuals
        )
        ln_2_hook_handle.remove()

        # Approximate MLP non-linearity with the gradient at this point.
        # To do this we can run the MLP forward and backward, and get the gradient of the input

        # Apply MLP
        mlp_module = model.blocks[dest_l].mlp
        mlp_out: Float[Tensor, "batch src_comp d_model"] = mlp_module.forward(mlp_input)

        # Apply non-linearity

        # Apply final LN
        if model.cfg.normalization_type not in ["LN", "LNPre"]:
            scaled_result = mlp_out
        else:
            center_stack = mlp_out - mlp_out.mean(dim=-1, keepdim=True)
            scale = cache["ln_final.hook_scale"][0, -1, :]  # first batch, last token
            scaled_result = center_stack / scale

        # Get the DLA
        logit_attrs = einsum(
            scaled_result,
            logit_directions,
            "batch src_comp d_model, \
                token d_model -> \
                batch src_comp token",
        )

        layers.append(logit_attrs)

    layers_tensor: Float[Tensor, "dest_l batch src_comp token"] = torch.stack(
        layers, dim=0
    )

    rearranged = rearrange(
        layers_tensor,
        "dest_l batch src_comp token -> \
            token batch dest_l src_comp",
    )

    return rearranged


def dla_attn_head_breakdown_source_component(
    cache: ActivationCache, model: HookedTransformer, tokens
) -> Float[Tensor, "token batch dest_l dest_h src_pos src_comp"]:
    """Direct Logit Attribution (DLA) Breakdown of Attention Heads by Source Component.

    Gets the DLA breakdown backwards through the last token's attention heads -> source tokens ->
    source components. Note this is only an OV breakdown (freezes QK).

    TODO: Enable batching along dest_l, dest_h, src_pos and src_comp dimensions. This will allow
    running this function recursively to get the breakdown going back many components.

    TODO: Support all architectures, possibly by having a "keep_dimensions" flag to the forward
    pass of the underlying components.

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

    def patch_with_cache_hook(_activations, hook):
        print(hook.name)
        return cache[hook.name]

    # Get the max number of source components (i.e. number for the last destination component)
    source_residuals_all: Float[
        Tensor, "src_comp batch src_pos d_model"
    ] = cache.decompose_resid()
    max_src_components: int = source_residuals_all.shape[0]

    # Check the model config is setup so that we get the breakdown of attention result by head
    if not model.cfg.use_attn_result:
        raise AttributeError(
            "Model config parameter `use_attn_result` must be set to `True`."
        )

    model.remove_all_hook_fns()

    for dest_l in range(model.cfg.n_layers):
        source_residuals: Float[
            Tensor, "src_comp batch src_pos d_model"
        ] = cache.decompose_resid(layer=dest_l)

        source_residuals = expand_tensor_dimension(
            source_residuals, 0, max_src_components, 0.0
        )

        # Apply LN to the stack
        ln_1: LayerNormPre = model.blocks[dest_l].ln2
        ln_1_hook = ln_1.hook_scale
        ln_1_hook.add_hook(patch_with_cache_hook)
        value_input: Float[Tensor, "batch src_comp src_pos d_model"] = ln_1.forward(
            source_residuals
        )
        ln_1_hook.remove_hooks()

        # Apply attention layer
        attn_module: Attention = model.blocks[dest_l].attn
        attn_pattern_hook = attn_module.hook_pattern
        attn_pattern_hook.add_hook(patch_with_cache_hook)
        query_key_input = cache[f"blocks.{dest_l}.hook_attn_in"]
        attn_out: Float[
            Tensor, "batch src_comp src_pos dest_h d_model"
        ] = attn_module.forward(query_key_input, query_key_input, value_input)

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

    layers_tensor: Float[
        Tensor, "dest_l batch src_comp src_pos dest_h token"
    ] = torch.stack(layers, dim=0)

    rearranged = rearrange(
        layers_tensor,
        "dest_l batch src_comp src_pos dest_h token -> \
            token batch dest_l dest_h src_pos src_comp",
    )

    return rearranged
