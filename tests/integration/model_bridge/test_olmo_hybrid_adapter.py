"""Integration tests for the OLMo Hybrid architecture adapter.

No tiny OlmoHybrid exists on the hub (only the 7B trio), so the fixture
builds a seeded tiny-random checkpoint from OlmoHybridConfig with the
OLMo-2 tokenizer.
"""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge


@pytest.fixture(scope="module")
def snapshot_path(tmp_path_factory):
    from transformers import AutoTokenizer
    from transformers.models.olmo_hybrid import OlmoHybridConfig, OlmoHybridForCausalLM

    tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
    torch.manual_seed(0)
    cfg = OlmoHybridConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=len(tok),
        max_position_embeddings=512,
        pad_token_id=tok.pad_token_id or 0,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
    )
    path = tmp_path_factory.mktemp("olmo_hybrid") / "tiny-olmo-hybrid"
    model = OlmoHybridForCausalLM(cfg).to(torch.float32)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def olmo_bridge(snapshot_path):
    return TransformerBridge.boot_transformers(snapshot_path, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens():
    torch.manual_seed(0)
    return torch.randint(0, 2000, (1, 12))


class TestOlmoHybridBridgeCreation:
    def test_adapter_selected(self, olmo_bridge):
        from transformer_lens.model_bridge.supported_architectures.olmo_hybrid import (
            OlmoHybridArchitectureAdapter,
        )

        assert isinstance(olmo_bridge.adapter, OlmoHybridArchitectureAdapter)


class TestOlmoHybridForwardEquivalence:
    def test_forward_matches_fresh_hf(self, olmo_bridge, snapshot_path, sample_tokens):
        from transformers import AutoModelForCausalLM

        fresh = AutoModelForCausalLM.from_pretrained(
            snapshot_path, dtype=torch.float32, attn_implementation="eager"
        )
        fresh.eval()
        with torch.no_grad():
            bridge_out = olmo_bridge(sample_tokens)
            hf_out = fresh(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs fresh HF max diff = {max_diff}"


class TestOlmoHybridHooks:
    def test_hooks_fire_per_layer_type(self, olmo_bridge, sample_tokens):
        """Layer 0 is linear attention, layer 1 is full attention."""
        d_model = olmo_bridge.cfg.d_model
        seq = sample_tokens.shape[1]
        expected = {
            "blocks.0.linear_attn.hook_out": (1, seq, d_model),
            "blocks.1.attn.hook_out": (1, seq, d_model),
            "blocks.1.ln2_post.hook_out": (1, seq, d_model),
            "blocks.0.mlp.hook_out": (1, seq, d_model),
        }
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        with torch.no_grad():
            olmo_bridge.run_with_hooks(sample_tokens, fwd_hooks=[(name, grab) for name in expected])
        for name, shape in expected.items():
            assert captured.get(name) == shape, f"{name}: {captured.get(name)}"


class TestOlmoHybridGeneration:
    def test_generate_with_stateful_cache(self, olmo_bridge):
        text = olmo_bridge.generate("Hello", max_new_tokens=5, do_sample=False, verbose=False)
        assert isinstance(text, str)
        assert text.startswith("Hello")
