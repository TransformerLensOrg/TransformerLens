"""Weight conversion utilities."""

import torch


def get_weight_conversion_field_set(weights: dict) -> str:
    """Creates a formatted string showing how weights are mapped between frameworks.

    Args:
        weights: Dictionary containing weight mappings where:
            - keys are TransformerLens weight names
            - values can be:
                * tuple[str, "BaseTensorConversion"]
                * torch.Tensor
                * strings

    Returns:
        A formatted multi-line string showing each weight's mapping details.
    """
    conversion_string = ""
    for transformer_lens_weight in weights:
        hugging_face_weight = weights[transformer_lens_weight]

        # Case 1: Nested conversion, call __repr__ for the nested conversion
        if isinstance(hugging_face_weight, tuple):
            weight_name, conversion = hugging_face_weight
            conversion_string += (
                f'"{transformer_lens_weight}" -> "{weight_name}", {conversion.__repr__()}\n'
            )

        # Case 2: Tensor, display shape and content
        elif isinstance(hugging_face_weight, torch.Tensor):
            if torch.all(hugging_face_weight == 0):
                conversion_string += f'"{transformer_lens_weight}" -> "Tensor filled with zeros of shape {hugging_face_weight.shape}",\n'
            elif torch.all(hugging_face_weight == 1):
                conversion_string += f'"{transformer_lens_weight}" -> "Tensor filled with ones of shape {hugging_face_weight.shape}",\n'
            else:
                conversion_string += f'"{transformer_lens_weight}" -> "Tensor of shape {hugging_face_weight.shape}",\n'

        # Case 3: String, just display string (name of weight in HuggingFace)
        else:
            conversion_string += f'"{transformer_lens_weight}" -> "{hugging_face_weight}",\n'
    return conversion_string


def model_info_cfg(cfg):
    """
    Displays the weight conversion from HuggingFace to TransformerLens for a given model configuration.

    Args:
        cfg: Model configuration object containing architecture information
    """

    # TODO: WeightConversionFactory import needs to be updated or removed
    # from transformer_lens.factories.weight_conversion_factory import (
    #     WeightConversionFactory,
    # )

    # weight_conversion = WeightConversionFactory.select_weight_conversion_config(cfg)
    print(f"Hook conversion details for architecture {cfg.original_architecture}:")
    # print(weight_conversion.__repr__())
    print("Hook conversion factory not yet implemented")


def model_info(model_name):
    """
    Displays the weight conversion from HuggingFace to TransformerLens for a given model name.

    Args:
        model_name (str): Name of the pretrained model to analyze
                            (e.g., 'gpt2', 'bert-base-uncased', etc.)
    """
    from transformer_lens.loading_from_pretrained import get_pretrained_model_config

    cfg = get_pretrained_model_config(model_name)
    model_info_cfg(cfg)
