import pytest
from transformers import AutoConfig

MODEL_NAMES = [
    "roneneldan/TinyStories-Instruct-1M",
    "roneneldan/TinyStories-Instruct-3M",
    "roneneldan/TinyStories-Instruct-8M",
    "roneneldan/TinyStories-Instruct-28M",
    "roneneldan/TinyStories-Instruct-33M",
    "roneneldan/TinyStories-1Layer-21M",
    "roneneldan/TinyStories-2Layers-33M",
    "roneneldan/TinyStories-Instuct-1Layer-21M",
    "roneneldan/TinyStories-Instruct-2Layers-33M",
]


def test_all_models_exist():
    for model in MODEL_NAMES:
        try:
            AutoConfig.from_pretrained(model)
        except OSError:
            pytest.fail(
                f"Could not download model '{model}' from Huggingface."
                " Maybe the name was changed or the model has been removed."
            )
