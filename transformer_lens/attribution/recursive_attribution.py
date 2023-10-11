"""Recursive Logit Attribution.

Recursively break down the logit attribution of each component.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from transformer_lens import ActivationCache, HookedTransformerConfig


class ComponentType(Enum):
    """Component Type.

    Type of a component, that we have calculated the direct logit attribution for.
    """

    MODEL = "model"
    EMBED = "embed"
    POSITIONAL_EMBED = "positional_embed"
    ATTENTION = "attention"
    MLP = "mlp"
    SOURCE_TOKEN = "source_token"
    MLP_NEURON = "mlp_neuron"


@dataclass
class ComponentLabel:
    """Component Label.

    Label for a specific component, that we have calculated the direct logit attribution for.

    Note that where the tree depth is greater than 1, there will be multiple labels for the same
    component (as the logit attribution will have been calculated multiple times, once for each
    parent component).
    """

    type: ComponentType
    layer: Optional[int] = None
    head: Optional[int] = None
    position: Optional[int] = None
    neuron: Optional[int] = None
    depth: int = 0
    parent_component: Optional[ComponentLabel] = None


def get_component_labels_recursively(
    model_config: HookedTransformerConfig,
    cache: ActivationCache,
    number_tokens: int,
    current_depth: int,
    depth_remaining: int = 5,
    current_component: ComponentLabel = ComponentLabel(type=ComponentType.MODEL),
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
                        type=ComponentType.EMBED,
                        position=number_tokens - 1,
                        depth=sub_component_depth,
                        parent_component=current_component,
                    )
                )
            if cache.has_pos_embed:
                recursive_sub_component_labels.append(
                    ComponentLabel(
                        type=ComponentType.POSITIONAL_EMBED,
                        position=number_tokens - 1,
                        depth=sub_component_depth,
                        parent_component=current_component,
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
                            type=ComponentType.ATTENTION,
                            position=number_tokens - 1,
                            layer=layer,
                            head=head,
                            depth=sub_component_depth,
                            parent_component=current_component,
                        )
                    )
                if not model_config.attn_only:
                    recursive_sub_component_labels.append(
                        ComponentLabel(
                            type=ComponentType.MLP,
                            position=number_tokens - 1,
                            layer=layer,
                            depth=sub_component_depth,
                            parent_component=current_component,
                        )
                    )

        case ComponentType.ATTENTION:
            # Break down by source token
            for pos in range(number_tokens):
                recursive_sub_component_labels.append(
                    ComponentLabel(
                        type=ComponentType.SOURCE_TOKEN,
                        position=pos,
                        layer=current_component.layer,
                        depth=sub_component_depth,
                        parent_component=current_component,
                    )
                )

        case ComponentType.MLP:
            # Break down by neuron
            for neuron in range(model_config.d_mlp):
                recursive_sub_component_labels.append(
                    ComponentLabel(
                        type=ComponentType.MLP_NEURON,
                        position=current_component.position,
                        layer=current_component.layer,
                        neuron=neuron,
                        depth=sub_component_depth,
                        parent_component=current_component,
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
