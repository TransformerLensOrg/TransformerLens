import torch
from transformers import AutoModelForCausalLM

from transformer_lens import HookedTransformer


class TestMatchHuggingFace:
    def test_test_mlp_outputs_match(self):
        tl_model = HookedTransformer.from_pretrained_no_processing("gpt2", device="cpu")
        hf_model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="cpu")
        test_tensor = torch.randn((1, 1, tl_model.cfg.d_model))

        for layer_n in range(len(tl_model.blocks)):
            tl_out = tl_model.blocks[layer_n].mlp(test_tensor)
            hf_out = hf_model.transformer.h[layer_n].mlp(test_tensor)

            assert torch.sum(tl_out == hf_out) == tl_model.cfg.d_model
