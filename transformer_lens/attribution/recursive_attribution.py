"""Recursive Logit Attribution.

Recursively break down the logit attribution of each component.

## Components

### Available weights and activations

```
mlp_output = cache[("mlp_out", layer)] # [batch, pos, d_model]
attention_value = cache[f"blocks.{layer}.attn.hook_v"] # [batch, pos, head, d_head]
attention_pattern = cache[f"blocks.{layer}.attn.hook_pattern"] # [batch, head, query_pos, key_pos]
W_O = model.blocks[layer].attn.W_O # [head, d_head, d_model]
```

### Residual decomposition

#### final_token_layer

final_token_layer = layer_activations[-1] # [layer, d_model]

#### final_token_attention_head

for layer in layers:
    attention_value, attention_pattern[:,:, -1] => z [batch head d_head]
    z, W_O => final_token_layer_attention_head [batch head d_model]
=> final_token_attention_head [batch layer head d_model]

#### final_token_attention_source_token

for layer in layers:
    attention_value, attention_pattern => z [batch, q, k, head, d_head]
    z, W0 => pos  [q, k, head, d_model]
    

### Attention

The attention heads must be calculated:

```
attn_out = cache[f"blocks.{layer}.attn.hook_result"]
W_O = model.blocks[layer].attn.W_O
```

$$ \\text{attn_out} @ W_O $$

### Attention by source



"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from jaxtyping import Float
from torch import Tensor

from transformer_lens import ActivationCache, HookedTransformerConfig
from transformer_lens.attribution.logit_attribution import logit_attribution


class ComponentType(Enum):
    """Component Type.

    Type of a component, that we have calculated the direct logit attribution for.
    """

    ATTENTION = "attention"
    EMBED = "embed"
    MLP = "mlp"
    MLP_NEURON = "mlp_neuron"
    MODEL = "model"
    POSITIONAL_EMBED = "positional_embed"
    SOURCE_TOKEN = "source_token"


@dataclass
class ComponentLabel:
    """Component Label.

    Label for a specific component, that we have calculated the direct logit attribution for.

    Note that where the tree depth is greater than 1, there will be multiple labels for the same
    component (as the logit attribution will have been calculated multiple times, once for each
    parent component).
    """

    depth: int
    head: Optional[int] = None
    layer: Optional[int] = None
    neuron: Optional[int] = None
    parent_component: Optional[ComponentLabel] = None
    position: Optional[int] = None
    type: ComponentType


def get_component_labels_recursively(
    cache: ActivationCache,
    model_config: HookedTransformerConfig,
    number_tokens: int,
    depth_remaining: int,
    current_depth: int = 0,
    current_component: ComponentLabel = ComponentLabel(
        type=ComponentType.MODEL, depth=0
    ),
) -> List[ComponentLabel]:
    """Get the component labels recursively.

    We get just the labels recursively, as the logit attribution can then be calculated in batches
    of components (more computationally efficient).
    """
    # Store of all component labels
    recursive_sub_component_labels: List[ComponentLabel] = []

    # Get the depth of sub-components
    sub_component_depth = current_depth + 1

    # Get any sub-components of the current component
    match current_component.type:
        case ComponentType.MODEL | ComponentType.SOURCE_TOKEN:
            if cache.has_embed:
                recursive_sub_component_labels.append(
                    ComponentLabel(
                        depth=sub_component_depth,
                        parent_component=current_component,
                        position=number_tokens - 1,
                        type=ComponentType.EMBED,
                    )
                )
            if cache.has_pos_embed:
                recursive_sub_component_labels.append(
                    ComponentLabel(
                        depth=sub_component_depth,
                        parent_component=current_component,
                        position=number_tokens - 1,
                        type=ComponentType.POSITIONAL_EMBED,
                    )
                )

            layers = (
                current_component.layer
                if current_component.type == ComponentType.SOURCE_TOKEN
                else model_config.n_layers
            )

            for layer in range(layers):
                for head in range(model_config.n_heads):
                    recursive_sub_component_labels.append(
                        ComponentLabel(
                            depth=sub_component_depth,
                            head=head,
                            layer=layer,
                            parent_component=current_component,
                            position=number_tokens - 1,
                            type=ComponentType.ATTENTION,
                        )
                    )
                if not model_config.attn_only:
                    recursive_sub_component_labels.append(
                        ComponentLabel(
                            depth=sub_component_depth,
                            layer=layer,
                            parent_component=current_component,
                            position=number_tokens - 1,
                            type=ComponentType.MLP,
                        )
                    )

        case ComponentType.ATTENTION:
            # Break down by source token
            for pos in range(number_tokens):
                recursive_sub_component_labels.append(
                    ComponentLabel(
                        depth=sub_component_depth,
                        layer=current_component.layer,
                        parent_component=current_component,
                        position=pos,
                        type=ComponentType.SOURCE_TOKEN,
                    )
                )

        case ComponentType.MLP:
            # Break down by neuron
            for neuron in range(model_config.d_mlp):
                recursive_sub_component_labels.append(
                    ComponentLabel(
                        depth=sub_component_depth,
                        layer=current_component.layer,
                        neuron=neuron,
                        parent_component=current_component,
                        position=current_component.position,
                        type=ComponentType.MLP_NEURON,
                    )
                )

        case ComponentType.EMBED | ComponentType.POSITIONAL_EMBED:
            # No sub-components (as they are the input)
            pass

        case ComponentType.MLP_NEURON:
            # Don't break down, because MLP neurons are non-linear
            pass

        case _:
            raise ValueError(f"Unknown component type: {current_component.type}")

    if depth_remaining > 0:
        # Recursively get the sub-components of the current component
        for component_label in recursive_sub_component_labels:
            recursive_labels = get_component_labels_recursively(
                model_config=model_config,
                cache=cache,
                number_tokens=number_tokens,
                depth_remaining=depth_remaining - 1,
                current_depth=current_depth + 1,
                current_component=component_label,
            )
            recursive_sub_component_labels.extend(recursive_labels)

    return recursive_sub_component_labels


def recursive_logit_attribution(
    cache: ActivationCache,
    model_config: HookedTransformerConfig,
    token_residual_directions: Float[Tensor, "token d_model"],
    number_tokens: int,
    search_depth: int = 3,
):
    # Get the component labels recursively
    component_labels = get_component_labels_recursively(
        cache=cache,
        model_config=model_config,
        number_tokens=number_tokens,
        depth_remaining=search_depth,
    )

    # For each component label, get the residual decomposition

    # Multiply these in batches by the token_residual_directions

