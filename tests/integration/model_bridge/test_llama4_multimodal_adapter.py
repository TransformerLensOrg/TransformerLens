"""Integration tests for the Llama 4 multimodal architecture adapter.

The only public tiny composite (yujiepan/llama-4-tiny-random) declares
``attn_temperature_tuning: 4``, which transformers 5.x strict config
validation rejects (bool expected); the fixture patches it to a bool in a
local snapshot.
"""

import json
import shutil

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "yujiepan/llama-4-tiny-random"


@pytest.fixture(scope="module")
def snapshot_path(tmp_path_factory):
    from huggingface_hub import snapshot_download

    src = snapshot_download(MODEL)
    path = tmp_path_factory.mktemp("llama4mm") / "tiny-llama4-mm-patched"
    shutil.copytree(src, path)
    cfg_path = path / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["text_config"]["attn_temperature_tuning"] = bool(
        cfg["text_config"]["attn_temperature_tuning"]
    )
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return str(path)


@pytest.fixture(scope="module")
def llama4mm_bridge(snapshot_path):
    return TransformerBridge.boot_transformers(snapshot_path, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(llama4mm_bridge):
    torch.manual_seed(0)
    return torch.randint(0, llama4mm_bridge.cfg.d_vocab - 10, (1, 12))


class TestLlama4MultimodalBridgeCreation:
    def test_adapter_selected(self, llama4mm_bridge):
        from transformer_lens.model_bridge.supported_architectures.llama4_multimodal import (
            Llama4MultimodalArchitectureAdapter,
        )

        assert isinstance(llama4mm_bridge.adapter, Llama4MultimodalArchitectureAdapter)


class TestLlama4MultimodalForwardEquivalence:
    def test_text_forward_matches_fresh_hf(self, llama4mm_bridge, snapshot_path, sample_tokens):
        from transformers import AutoModelForImageTextToText

        fresh = AutoModelForImageTextToText.from_pretrained(
            snapshot_path, dtype=torch.float32, attn_implementation="eager"
        )
        fresh.eval()
        with torch.no_grad():
            bridge_out = llama4mm_bridge(sample_tokens)
            hf_out = fresh(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs fresh HF max diff = {max_diff}"


class TestLlama4MultimodalHooks:
    def test_hooks_fire(self, llama4mm_bridge, sample_tokens):
        d_model = llama4mm_bridge.cfg.d_model
        seq = sample_tokens.shape[1]
        expected = {
            "blocks.0.attn.hook_out": (1, seq, d_model),
            "blocks.0.mlp.hook_out": (1, seq, d_model),
        }
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        with torch.no_grad():
            llama4mm_bridge.run_with_hooks(
                sample_tokens, fwd_hooks=[(name, grab) for name in expected]
            )
        for name, shape in expected.items():
            assert captured.get(name) == shape, f"{name}: {captured.get(name)}"


class TestLlama4MultimodalGeneration:
    def test_generate(self, llama4mm_bridge):
        text = llama4mm_bridge.generate("Hello", max_new_tokens=4, do_sample=False, verbose=False)
        assert isinstance(text, str)
        assert text.startswith("Hello")
