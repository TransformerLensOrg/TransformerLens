import math

import pytest
import torch
from transformers import AutoModelForCausalLM

from transformer_lens import HookedTransformer


class TestMatchHuggingFace:
    # fixtures
    @pytest.fixture(scope="class", params=["gpt2"])
    def model_name(self, request):
        return request.param

    # tests
    def test_compare_huggingface_logits_match_local_implementation(self, model_name):
        tl_model = HookedTransformer.from_pretrained_no_processing(model_name, device="cpu")
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
        tensor_shape = (3, 5, tl_model.cfg.d_model)
        test_tensor = torch.randn(tensor_shape)

        for layer_n in range(len(tl_model.blocks)):
            tl_out = tl_model.blocks[layer_n].mlp(test_tensor)
            hf_out = hf_model.transformer.h[layer_n].mlp(test_tensor)

            assert torch.sum(tl_out == hf_out) == math.prod(tensor_shape)
