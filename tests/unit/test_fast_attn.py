import pytest
import torch

from transformer_lens.HookedTransformer import HookedTransformer


class TestFastAttn:
    prompt = """
    This is a library for doing mechanistic interpretability of GPT-2 Style language models. 
    The goal of mechanistic interpretability is to take a trained model and reverse engineer 
    the algorithms the model learned during training from its weights.
    """

    # fixtures
    @pytest.fixture(scope="class", params=["gpt2", "facebook/opt-125m"])
    def model_name(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def model(self, model_name):
        return HookedTransformer.from_pretrained(model_name)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA device")
    def test_logits_and_cache(self, model_name):
        model = HookedTransformer.from_pretrained(model_name).to("cuda")
        model.cfg.use_fast_attn = True
        fast_logits, fast_cache = model.run_with_cache(self.prompt)
        model.cfg.use_fast_attn = False
        slow_logits, slow_cache = model.run_with_cache(self.prompt)

        assert torch.allclose(
            fast_logits, slow_logits, rtol=5e-1, atol=5e-1
        ), "Logits mismatch"

        # Fast cache should be missing Attn Scores and Pattern Keys
        assert len(fast_cache) < len(slow_cache)

        for k, v in fast_cache.items():
            assert torch.allclose(
                v, slow_cache[k], rtol=5e-1, atol=5e-1
            ), f"Cache mismatch for {k}"
