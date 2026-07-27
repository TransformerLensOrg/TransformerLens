"""Integration tests for the Mistral 3 (Mistral-Small VLM) architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "tiny-random/mistral-3"


@pytest.fixture(scope="module")
def mistral3_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(mistral3_bridge):
    torch.manual_seed(0)
    return torch.randint(0, mistral3_bridge.cfg.d_vocab - 10, (1, 12))


class TestMistral3BridgeCreation:
    def test_adapter_selected(self, mistral3_bridge):
        from transformer_lens.model_bridge.supported_architectures.mistral3 import (
            Mistral3ArchitectureAdapter,
        )

        assert isinstance(mistral3_bridge.adapter, Mistral3ArchitectureAdapter)


class TestMistral3ForwardEquivalence:
    def test_text_forward_matches_fresh_hf(self, mistral3_bridge, sample_tokens):
        from transformers import AutoModelForImageTextToText

        fresh = AutoModelForImageTextToText.from_pretrained(
            MODEL, dtype=torch.float32, attn_implementation="eager"
        )
        fresh.eval()
        with torch.no_grad():
            bridge_out = mistral3_bridge(sample_tokens)
            hf_out = fresh(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs fresh HF max diff = {max_diff}"

    def test_multimodal_forward_matches_fresh_hf(self, mistral3_bridge):
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor

        fresh = AutoModelForImageTextToText.from_pretrained(
            MODEL, dtype=torch.float32, attn_implementation="eager"
        )
        fresh.eval()
        proc = AutoProcessor.from_pretrained(MODEL)
        img = Image.new("RGB", (64, 64), "red")
        image_token = getattr(proc, "image_token", "[IMG]")
        inputs = proc(text=f"{image_token}Describe", images=img, return_tensors="pt")
        # Drive the bridge's own forward (input_ids positional, pixel_values/attention_mask
        # as kwargs) so the multimodal path — not just the wrapped HF model — is exercised.
        bridge_inputs = {k: v for k, v in inputs.items() if k != "input_ids"}
        with torch.no_grad():
            bridge_out = mistral3_bridge(inputs["input_ids"], **bridge_inputs)
            hf_out = fresh(**inputs).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs fresh HF max diff = {max_diff}"


class TestMistral3Hooks:
    def test_hooks_fire(self, mistral3_bridge, sample_tokens):
        d_model = mistral3_bridge.cfg.d_model
        seq = sample_tokens.shape[1]
        expected = {
            "blocks.0.attn.hook_out": (1, seq, d_model),
            "blocks.0.mlp.hook_out": (1, seq, d_model),
            "blocks.1.mlp.hook_out": (1, seq, d_model),
        }
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        with torch.no_grad():
            mistral3_bridge.run_with_hooks(
                sample_tokens, fwd_hooks=[(name, grab) for name in expected]
            )
        for name, shape in expected.items():
            assert captured.get(name) == shape, f"{name}: {captured.get(name)}"


class TestMistral3Generation:
    def test_generate(self, mistral3_bridge):
        text = mistral3_bridge.generate("Hello", max_new_tokens=5, do_sample=False, verbose=False)
        assert isinstance(text, str)
        assert text.startswith("Hello")


def _tiny_ministral3_vlm_pair():
    """Seeded tiny Mistral3 VLM with a Ministral-3 text stack (no hub access).

    original_max_position_embeddings=4 puts most of a 12-token prompt past the
    llama-4 query-scale threshold, so a bridge that skips the scale diverges.
    """
    import copy

    from transformers import (
        Ministral3Config,
        Mistral3Config,
        Mistral3ForConditionalGeneration,
        PixtralVisionConfig,
    )

    from transformer_lens.model_bridge.sources._bridge_builder import (
        build_bridge_from_module,
    )

    text_cfg = Ministral3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "llama_4_scaling_beta": 0.5,
            "original_max_position_embeddings": 4,
        },
    )
    vision_cfg = PixtralVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=32,
        patch_size=16,
    )
    cfg = Mistral3Config(
        text_config=text_cfg,
        vision_config=vision_cfg,
        image_token_index=3,
        vision_feature_layer=-1,
    )
    cfg._attn_implementation = "eager"

    torch.manual_seed(42)
    ref = Mistral3ForConditionalGeneration(cfg).eval()
    hf = Mistral3ForConditionalGeneration(copy.deepcopy(cfg)).eval()
    hf.load_state_dict(ref.state_dict())
    bridge = build_bridge_from_module(
        hf,
        "Mistral3ForConditionalGeneration",
        hf_config=copy.deepcopy(cfg),
        tokenizer=None,
        device="cpu",
    ).eval()
    return bridge, ref


class TestMinistral3QueryScale:
    """Ministral-3 text stacks multiply Q by 1 + beta*log(1 + floor(pos/max))
    after RoPE; the reimplemented VLM attention path must apply it."""

    def test_forward_matches_hf_past_scale_threshold(self) -> None:
        bridge, ref = _tiny_ministral3_vlm_pair()
        torch.manual_seed(0)
        ids = torch.randint(4, 128, (1, 12))
        with torch.no_grad():
            out = bridge(ids)
            expected = ref(input_ids=ids).logits
        max_diff = (out - expected).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"
