"""Integration tests for the Idefics3 architecture adapter."""

from functools import partial

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge

MODEL = "ibm-granite/granite-docling-258M"


@pytest.fixture(scope="module")
def bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


class TestIdefics3BridgeCreation:
    def test_adapter_and_components(self, bridge):
        from transformer_lens.model_bridge.supported_architectures.idefics3 import (
            Idefics3ArchitectureAdapter,
        )

        assert isinstance(bridge.adapter, Idefics3ArchitectureAdapter)
        assert bridge.cfg.is_multimodal is True
        assert hasattr(bridge, "vision_encoder")
        assert hasattr(bridge, "vision_projector")

    def test_vision_tower_is_live(self, bridge):
        hf_model = bridge.original_model
        assert bridge.vision_encoder is hf_model.model.vision_model
        assert bridge.vision_projector is hf_model.model.connector


class TestIdefics3ForwardEquivalence:
    def test_text_forward_matches_hf(self, bridge, sample_tokens):
        hf_model = bridge.original_model
        with torch.no_grad():
            bridge_out = bridge(sample_tokens)
            hf_out = hf_model(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"


class TestIdefics3Hooks:
    def test_text_hooks_fire(self, bridge, sample_tokens):
        d_model = bridge.cfg.d_model
        captured = {}

        def grab(tensor, hook):
            captured[hook.name] = tuple(tensor.shape)

        hooks = ["blocks.0.attn.hook_out", "blocks.0.mlp.hook_out"]
        with torch.no_grad():
            bridge.run_with_hooks(sample_tokens, fwd_hooks=[(name, grab) for name in hooks])
        seq = sample_tokens.shape[1]
        for name in hooks:
            assert captured.get(name) == (1, seq, d_model), f"{name}: {captured.get(name)}"


class TestSiglipVisionLayerHooks:
    """The vision layer bridge declares hook_attn_*/hook_mlp_* aliases, so the
    attn/mlp submodules those point at have to exist. They did not: the layer was
    built with no submodules, leaving the four hooks silently inaccessible on
    every SigLIP layer and 48 "did not resolve" warnings at boot.
    """

    ALIASES = ("hook_attn_in", "hook_attn_out", "hook_mlp_in", "hook_mlp_out")

    def test_declared_aliases_resolve_on_every_layer(self, bridge):
        for index, layer in enumerate(bridge.vision_encoder.encoder_layers):
            for alias in self.ALIASES:
                assert hasattr(layer, alias), f"layer {index} is missing {alias}"

    def test_q_fires_split_by_the_vision_towers_head_count(self, bridge):
        """The payoff, asserted on a fired tensor rather than on the config.

        q/k/v reshape to [batch, pos, n_heads, d_head] by ``config.n_heads``. Two
        regressions leave the config assert green while the shape goes wrong: the
        raw HF vision config reads as n_heads=1 (it spells the field
        num_attention_heads), and dropping the vision tower from the bridge's
        hook-compatibility pass makes the hook fire flat [batch, pos, d_model].
        The text model's own count (9 over 576) is wrong here too.
        """
        heads = bridge.cfg.vision_num_heads
        d_head = bridge.cfg.vision_hidden_size // heads
        assert heads != bridge.cfg.n_heads, "fixture no longer distinguishes the towers"

        fired: dict[str, tuple[int, ...]] = {}

        def record(tensor, hook, name):
            fired[name] = tuple(tensor.shape)
            return tensor

        attn = bridge.vision_encoder.encoder_layers[0].attn
        for projection in ("q", "k", "v"):
            getattr(attn, projection).hook_out.add_hook(partial(record, name=projection))

        torch.manual_seed(0)
        with torch.no_grad():
            bridge(
                torch.randint(0, bridge.cfg.d_vocab - 10, (1, 8)),
                pixel_values=torch.randn(1, 1, 3, 512, 512),
            )

        for projection in ("q", "k", "v"):
            assert projection in fired, f"{projection} never fired"
            batch, pos, *per_head = fired[projection]
            assert per_head == [heads, d_head], (
                f"{projection} fired {fired[projection]}, expected "
                f"[{batch}, {pos}, {heads}, {d_head}]"
            )

        # The shape above cannot catch the text config leaking in on this model:
        # the reshape takes d_head from the config and infers the head count from
        # the tensor width, and both towers happen to use d_head=64 (576/9 and
        # 768/12), so a leaked text config produces the same [.., 12, 64]. Assert
        # the dims directly for the case the fixture cannot express behaviourally.
        config = attn.config
        assert config.n_heads == heads
        assert config.d_model == bridge.cfg.vision_hidden_size

    def test_vision_hooks_fire_with_tower_shapes(self, bridge, sample_tokens):
        fired: dict[str, tuple[int, ...]] = {}
        layer = bridge.vision_encoder.encoder_layers[0]

        def record(tensor, hook, name):
            fired[name] = tuple(tensor.shape)
            return tensor

        for alias in self.ALIASES:
            getattr(layer, alias).add_hook(partial(record, name=alias))
        torch.manual_seed(0)
        with torch.no_grad():
            bridge(sample_tokens, pixel_values=torch.randn(1, 1, 3, 512, 512))
        for alias in self.ALIASES:
            assert alias in fired, f"{alias} never fired"
            assert fired[alias][-1] == bridge.cfg.vision_hidden_size
