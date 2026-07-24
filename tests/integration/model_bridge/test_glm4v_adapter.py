"""Integration tests for the GLM-4V architecture adapter.

Every public GLM-4V tiny (tiny-random/glm-4v, yujiepan variants) ships
without lm_head.weight despite tie_word_embeddings=false, so HF randomly
re-initializes the head each load; the fixture initializes it
deterministically in a local snapshot.
"""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "tiny-random/glm-4v"


@pytest.fixture(scope="module")
def snapshot_path(tmp_path_factory):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    path = tmp_path_factory.mktemp("glm4v") / "tiny-glm4v-sanitized"
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32)
    torch.manual_seed(42)
    with torch.no_grad():
        model.lm_head.weight.copy_(torch.randn_like(model.lm_head.weight) * 0.02)
    model.save_pretrained(path)
    AutoProcessor.from_pretrained(MODEL).save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def glm4v_bridge(snapshot_path):
    return TransformerBridge.boot_transformers(snapshot_path, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(glm4v_bridge):
    torch.manual_seed(0)
    return torch.randint(0, glm4v_bridge.cfg.d_vocab - 10, (1, 12))


class TestGlm4vBridgeCreation:
    def test_adapter_selected(self, glm4v_bridge):
        from transformer_lens.model_bridge.supported_architectures.glm4v import (
            Glm4vArchitectureAdapter,
        )

        assert isinstance(glm4v_bridge.adapter, Glm4vArchitectureAdapter)


class TestGlm4vForwardEquivalence:
    def test_text_forward_matches_fresh_hf(self, glm4v_bridge, snapshot_path, sample_tokens):
        from transformers import AutoModelForImageTextToText

        fresh = AutoModelForImageTextToText.from_pretrained(
            snapshot_path, dtype=torch.float32, attn_implementation="eager"
        )
        fresh.eval()
        with torch.no_grad():
            bridge_out = glm4v_bridge(sample_tokens)
            hf_out = fresh(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs fresh HF max diff = {max_diff}"

    def test_multimodal_forward_matches_fresh_hf(self, glm4v_bridge, snapshot_path):
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor

        fresh = AutoModelForImageTextToText.from_pretrained(
            snapshot_path, dtype=torch.float32, attn_implementation="eager"
        )
        fresh.eval()
        proc = AutoProcessor.from_pretrained(snapshot_path)
        img = Image.new("RGB", (112, 112), "red")
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": "Describe"}],
            }
        ]
        text = proc.apply_chat_template(messages, add_generation_prompt=True)
        inputs = dict(proc(text=[text], images=[img], return_tensors="pt"))
        # Drive the bridge's own forward (input_ids positional, pixel_values/attention_mask
        # as kwargs) so the multimodal path — not just the wrapped HF model — is exercised.
        bridge_inputs = {k: v for k, v in inputs.items() if k != "input_ids"}
        with torch.no_grad():
            bridge_out = glm4v_bridge(inputs["input_ids"], **bridge_inputs)
            hf_out = fresh(**inputs).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs fresh HF max diff = {max_diff}"


class TestGlm4vHooks:
    def test_hooks_fire_including_sandwich_norms(self, glm4v_bridge, sample_tokens):
        d_model = glm4v_bridge.cfg.d_model
        seq = sample_tokens.shape[1]
        expected = {
            "blocks.0.attn.hook_out": (1, seq, d_model),
            "blocks.0.mlp.hook_out": (1, seq, d_model),
            "blocks.0.ln1_post.hook_out": (1, seq, d_model),
            "blocks.0.ln2_post.hook_out": (1, seq, d_model),
        }
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        with torch.no_grad():
            glm4v_bridge.run_with_hooks(
                sample_tokens, fwd_hooks=[(name, grab) for name in expected]
            )
        for name, shape in expected.items():
            assert captured.get(name) == shape, f"{name}: {captured.get(name)}"


class TestGlm4vGeneration:
    def test_generate(self, glm4v_bridge):
        text = glm4v_bridge.generate("Hello", max_new_tokens=5, do_sample=False, verbose=False)
        assert isinstance(text, str)
        assert text.startswith("Hello")
