"""Offline reconstruction parity, not a live vLLM engine/compiled-graph test."""

from types import SimpleNamespace

import pytest
import torch
from transformers import (
    CohereConfig,
    CohereForCausalLM,
    GraniteConfig,
    GraniteForCausalLM,
)


@pytest.mark.parametrize(
    "family,scale", [("cohere", 0.0625), ("cohere", 1.0), ("granite", 2.0), ("granite", 1.0)]
)
def test_reconstruction_matches_hf_logits_probabilities_and_loss(family, scale):
    from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
    from transformer_lens.model_bridge.sources._bridge_builder import (
        build_bridge_config_from_hf,
    )
    from transformer_lens.model_bridge.sources.vllm.driver import VLLMDriver
    from transformer_lens.model_bridge.sources.vllm.worker_extension import (
        TLWorkerExtension,
    )
    from transformer_lens.utilities.lm_utils import lm_cross_entropy_loss

    common = dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attn_implementation="eager",
    )
    torch.manual_seed(7)
    if family == "cohere":
        config = CohereConfig(
            **common, logit_scale=scale, use_qk_norm=False, tie_word_embeddings=True
        )
        model = CohereForCausalLM(config).eval()
        processor_scale = scale
    else:
        config = GraniteConfig(
            **common, logits_scaling=scale, embedding_multiplier=1.0, residual_multiplier=1.0
        )
        model = GraniteForCausalLM(config).eval()
        processor_scale = 1.0 / scale
    # The processor's plain attributes mirror vLLM's runtime wire contract.
    model.logits_processor = SimpleNamespace(
        scale=processor_scale, soft_cap=None, logits_as_input=False
    )
    worker = TLWorkerExtension()
    worker.model_runner = SimpleNamespace(model=model)
    engine = SimpleNamespace(
        collective_rpc=lambda method, args=(): [getattr(worker, method)(*args)]
    )
    cfg = build_bridge_config_from_hf(config, type(model).__name__, "tiny-offline", torch.float32)
    adapter = ArchitectureAdapter(cfg)
    overlay = SimpleNamespace(
        capture_specs=lambda _: {"ln_final.hook_normalized": ("model.norm", 16)},
        nonfiring_hooks=lambda: [],
    )
    driver = VLLMDriver(engine, adapter, None, overlay, config, max_num_batched_tokens=16)
    captures = {}
    handle = model.model.norm.register_forward_hook(
        lambda module, args, output: captures.__setitem__("norm", output.detach())
    )
    tokens = torch.tensor([[3, 4, 5]])
    try:
        with torch.no_grad():
            expected = model(tokens).logits
        actual = driver._reconstruct_logits(captures["norm"])
        assert actual is not None
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual.softmax(-1), expected.softmax(-1))
        torch.testing.assert_close(
            lm_cross_entropy_loss(actual, tokens), lm_cross_entropy_loss(expected, tokens)
        )
    finally:
        handle.remove()
        driver.close()
