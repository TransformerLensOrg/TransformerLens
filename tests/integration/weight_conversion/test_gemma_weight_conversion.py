import pytest

from transformer_lens.loading_from_pretrained import (
    convert_hf_model_config,
    load_hugging_face_model,
)
from transformer_lens.weight_conversion.gemma import GemmaWeightConversion

small_models = [
    "google/gemma-2b",
    "google/gemma-2b-it",
    "google/gemma-2-2b",
    "google/gemma-2-2b-it",
]

@pytest.mark.parametrize("model_id", small_models)
def test_small_model_weight_conversion(model_id):
    model_config = convert_hf_model_config(model_id)
    hf_model = load_hugging_face_model(
        model_id,
        cfg=model_config,
    )
    weight_convertor = GemmaWeightConversion(model_config)
    tl_model = weight_convertor.convert(hf_model)
    
    
