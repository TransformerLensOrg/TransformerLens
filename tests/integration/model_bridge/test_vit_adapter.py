"""Integration tests for the ViT / DeiT architecture adapter.

Modeled on tests/integration/model_bridge/test_mamba_adapter.py — verifies
wrap-don't-reimplement behavior against real HF checkpoints:
- Forward pass matches HF (bridge substitutes hooked submodules into the real
  HF model and calls its real forward — see bridge.py's `is_visual_model`
  branch, which does `output = self.original_model(**kwargs)`)
- Submodule hooks fire with expected shapes (embed, attn q/k/v/o, mlp in/out,
  ln_final, unembed/classifier)
- DeiT's distillation token is invisible to the adapter (sequence length
  differs from ViT's by one extra token, entirely inside VisionEmbeddingsBridge)
- Bare ViTModel (no classifier) return-value shape via return_type="logits"
- DeiTForImageClassificationWithTeacher is rejected, not silently mishandled

Three checkpoints, matching the ones already sanity-checked manually:
- google/vit-base-patch16-224          (ViTForImageClassification, prefix "vit.")
- google/vit-base-patch16-224-in21k    (bare ViTModel, no prefix, no classifier)
- facebook/deit-small-patch16-224      (DeiTForImageClassification, prefix "deit.")

NOTE ON RUNNING THESE: they download real weights from HuggingFace (a few
hundred MB total) — same cost class as the existing Mamba/SmolLM3 integration
tests, no special marker needed, but expect the first run to be slow.
"""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components.vision_classifier_head import (
    VisionClassifierHeadBridge,
)

MODEL_VIT_CLASSIFIER = "google/vit-base-patch16-224"
MODEL_VIT_BARE = "google/vit-base-patch16-224-in21k"
MODEL_DEIT_CLASSIFIER = "facebook/deit-small-patch16-224"


def _pixel_values(batch: int = 1, image_size: int = 224) -> torch.Tensor:
    return torch.randn(batch, 3, image_size, image_size)


# ---------------------------------------------------------------------------
# ViTForImageClassification (the "normal" case)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vit_bridge():
    return TransformerBridge.boot_transformers(MODEL_VIT_CLASSIFIER, device="cpu")


class TestViTClassifierBridgeCreation:
    def test_block_count(self, vit_bridge):
        assert len(vit_bridge.blocks) == vit_bridge.original_model.config.num_hidden_layers

    def test_config_flags(self, vit_bridge):
        assert vit_bridge.cfg.normalization_type == "LN"
        assert vit_bridge.cfg.positional_embedding_type == "standard"
        assert vit_bridge.cfg.is_visual_model is True
        assert vit_bridge.cfg.gated_mlp is False
        assert vit_bridge.cfg.attn_only is False

    def test_has_core_components(self, vit_bridge):
        assert hasattr(vit_bridge, "embed")
        assert hasattr(vit_bridge, "blocks")
        assert hasattr(vit_bridge, "ln_final")
        assert hasattr(vit_bridge, "unembed")

    def test_unembed_is_vision_classifier_head(self, vit_bridge):
        assert isinstance(vit_bridge.unembed, VisionClassifierHeadBridge)

    def test_d_model_matches_hf_config(self, vit_bridge):
        assert vit_bridge.cfg.d_model == vit_bridge.original_model.config.hidden_size

    def test_n_heads_matches_hf_config(self, vit_bridge):
        assert vit_bridge.cfg.n_heads == vit_bridge.original_model.config.num_attention_heads


class TestViTClassifierForwardPass:
    def test_forward_returns_correct_shape(self, vit_bridge):
        pixel_values = _pixel_values()
        with torch.no_grad():
            output = vit_bridge(pixel_values)
        num_labels = vit_bridge.original_model.config.num_labels
        assert output.shape == (1, num_labels)
        assert not torch.isnan(output).any()

    def test_forward_matches_hf(self, vit_bridge):
        """Wrap-don't-reimplement: bridge substitutes hooked submodules into the
        real HF model and calls its real forward, so this should match HF very
        tightly (mamba's equivalent test asserts bit-exact; using a tight atol
        here to be safe against BLAS/backend-dependent float noise across
        machines rather than assuming bit-exactness untested)."""
        pixel_values = _pixel_values()
        hf_model = vit_bridge.original_model
        with torch.no_grad():
            bridge_out = vit_bridge(pixel_values)
            hf_out = hf_model(pixel_values).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"Bridge vs HF max diff = {max_diff}"

    def test_batch_of_images(self, vit_bridge):
        pixel_values = _pixel_values(batch=3)
        with torch.no_grad():
            output = vit_bridge(pixel_values)
        assert output.shape[0] == 3


class TestViTClassifierHookCoverage:
    @pytest.fixture(scope="class")
    def cache(self, vit_bridge):
        pixel_values = _pixel_values()
        with torch.no_grad():
            _, cache = vit_bridge.run_with_cache(pixel_values)
        return cache

    def test_embed_hooks_fire(self, cache, vit_bridge):
        assert "embed.hook_in" in cache
        assert "embed.hook_out" in cache
        assert cache["embed.hook_in"].shape == (1, 3, 224, 224)
        seq_len = cache["embed.hook_out"].shape[1]
        d_model = vit_bridge.cfg.d_model
        assert cache["embed.hook_out"].shape == (1, seq_len, d_model)
        # ViT: CLS + patches, no distillation token (contrast with DeiT below).
        assert seq_len == 1 + (224 // 16) ** 2

    def test_block_submodule_hooks_fire(self, cache, vit_bridge):
        n_layers = vit_bridge.cfg.n_layers
        for i in [0, n_layers - 1]:
            for submod in ("q", "k", "v", "o"):
                assert f"blocks.{i}.attn.{submod}.hook_in" in cache
                assert f"blocks.{i}.attn.{submod}.hook_out" in cache
            assert f"blocks.{i}.mlp.in.hook_out" in cache
            assert f"blocks.{i}.mlp.out.hook_out" in cache

    def test_mlp_hook_aliases_resolve(self, cache):
        """hook_mlp_in/out are redirected via hook_alias_overrides in vit.py."""
        assert torch.equal(cache["blocks.0.hook_mlp_in"], cache["blocks.0.mlp.in.hook_in"])
        assert torch.equal(cache["blocks.0.hook_mlp_out"], cache["blocks.0.mlp.out.hook_out"])

    def test_ln_final_hook_fires(self, cache):
        assert "ln_final.hook_normalized" in cache

    def test_unembed_hooks_fire_with_pooled_shape(self, cache, vit_bridge):
        """VisionClassifierHeadBridge only ever sees the pooled (batch, hidden)
        CLS-token tensor — HF's ViTForImageClassification.forward() does the
        `sequence_output[:, 0, :]` slicing itself before calling self.classifier,
        and that real forward path is what's running here."""
        d_model = vit_bridge.cfg.d_model
        num_labels = vit_bridge.original_model.config.num_labels
        assert cache["unembed.hook_in"].shape == (1, d_model)
        assert cache["unembed.hook_out"].shape == (1, num_labels)


# ---------------------------------------------------------------------------
# Bare ViTModel (no classifier) — google/vit-base-patch16-224-in21k
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vit_bare_bridge():
    return TransformerBridge.boot_transformers(MODEL_VIT_BARE, device="cpu")


class TestViTBareModel:
    def test_no_unembed_component(self, vit_bare_bridge):
        assert "unembed" not in vit_bare_bridge.adapter.component_mapping

    def test_forward_does_not_crash(self, vit_bare_bridge):
        pixel_values = _pixel_values()
        with torch.no_grad():
            vit_bare_bridge(pixel_values)  # must not raise

    def test_forward_return_value_is_not_a_plain_tensor(self, vit_bare_bridge):
        """KNOWN GAP — flagging, not necessarily desired behavior.

        For a bare ViTModel, HF's own forward returns BaseModelOutputWithPooling,
        which has no `.logits` and is NOT a tuple subclass (confirmed directly:
        `isinstance(BaseModelOutputWithPooling(...), tuple) is False` on
        transformers 5.8.1). bridge.py's forward() only special-cases `.logits`
        or `isinstance(output, tuple)` before falling through to
        `logits = output` — so `bridge(pixel_values)` with the default
        return_type="logits" currently returns the raw HF output object, not a
        tensor. If that's intentional (there's no real "logits" for a bare
        encoder), consider documenting it explicitly and/or exposing
        `last_hidden_state`/`pooler_output` some other way; if not, this is the
        regression test to flip once fixed.
        """
        pixel_values = _pixel_values()
        with torch.no_grad():
            output = vit_bare_bridge(pixel_values)
        assert not isinstance(output, torch.Tensor)
        assert hasattr(output, "last_hidden_state")


# ---------------------------------------------------------------------------
# DeiTForImageClassification — distillation token must stay invisible
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def deit_bridge():
    return TransformerBridge.boot_transformers(MODEL_DEIT_CLASSIFIER, device="cpu")


class TestDeiTBridge:
    def test_prefix_is_deit(self, deit_bridge):
        assert deit_bridge.adapter.component_mapping["embed"].name == "deit.embeddings"

    def test_forward_matches_hf(self, deit_bridge):
        pixel_values = _pixel_values()
        hf_model = deit_bridge.original_model
        with torch.no_grad():
            bridge_out = deit_bridge(pixel_values)
            hf_out = hf_model(pixel_values).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"Bridge vs HF max diff = {max_diff}"

    def test_sequence_length_includes_distillation_token(self, deit_bridge):
        """DeiT embeddings prepend CLS *and* distillation tokens (vs. ViT's CLS
        only) — invisible to the adapter per vision_embeddings.py's docstring,
        but it should still show up in the actual hidden-state shape, one token
        longer than the equivalent ViT sequence length."""
        pixel_values = _pixel_values()
        with torch.no_grad():
            _, cache = deit_bridge.run_with_cache(pixel_values)
        seq_len = cache["embed.hook_out"].shape[1]
        num_patches = (224 // 16) ** 2
        assert seq_len == num_patches + 2  # CLS + distillation + patches

    def test_unembed_pooled_shape(self, deit_bridge):
        pixel_values = _pixel_values()
        with torch.no_grad():
            _, cache = deit_bridge.run_with_cache(pixel_values)
        d_model = deit_bridge.cfg.d_model
        assert cache["unembed.hook_in"].shape == (1, d_model)


# ---------------------------------------------------------------------------
# cfg.n_ctx regression (see unit test file for the narrower, no-network version)
# ---------------------------------------------------------------------------


class TestViTConfigNCtx:
    @pytest.mark.xfail(
        reason=(
            "prepare_loading() in vit.py never derives n_ctx from "
            "image_size/patch_size, and ViTConfig/DeiTConfig have none of the "
            "field names the generic fallback chain in "
            "model_bridge/sources/transformers.py checks (n_positions / "
            "max_position_embeddings / max_context_length / max_length / "
            "seq_length), so it silently defaults to 2048. Flip this to a plain "
            "assert once prepare_loading() sets n_ctx correctly."
        ),
        strict=True,
    )
    def test_n_ctx_matches_real_patch_sequence_length(self, vit_bridge):
        expected = 1 + (224 // 16) ** 2  # CLS + patches, for a 224px/patch16 model
        assert vit_bridge.cfg.n_ctx == expected


# ---------------------------------------------------------------------------
# DeiTForImageClassificationWithTeacher — must raise, not silently mishandle
# ---------------------------------------------------------------------------


class TestDeiTWithTeacherRejected:
    def test_loading_raises_not_implemented(self):
        """No small public DeiTForImageClassificationWithTeacher checkpoint is
        assumed here — this drives the same code path prepare_model() exercises
        via a HF class instance check, without a full weights download.
        """
        from transformers import DeiTConfig, DeiTForImageClassificationWithTeacher

        from transformer_lens.model_bridge.supported_architectures.vit import (
            ViTArchitectureAdapter,
        )

        config = DeiTConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            image_size=32,
            patch_size=16,
            num_labels=10,
        )
        hf_model = DeiTForImageClassificationWithTeacher(config)

        adapter = ViTArchitectureAdapter.__new__(ViTArchitectureAdapter)
        adapter.cfg = None  # prepare_model() doesn't touch cfg before the raise
        with pytest.raises(NotImplementedError):
            ViTArchitectureAdapter.prepare_model(adapter, hf_model)
