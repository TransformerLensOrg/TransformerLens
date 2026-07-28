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
- prepare_model() doesn't assume a `.encoder` wrapper exists on the HF model
  (transformers now puts blocks directly on `<prefix>.layers`)

Four checkpoints, matching the ones already sanity-checked manually (added
facebook/deit-small-patch16-224 as a real DeiTForImageClassification load —
previously declared as MODEL_DEIT_CLASSIFIER but never actually used):
- google/vit-base-patch16-224          (ViTForImageClassification, prefix "vit.")
- google/vit-base-patch16-224-in21k    (bare ViTModel, no prefix, no classifier)
- facebook/deit-small-patch16-224      (DeiTForImageClassification, prefix "deit.")
- facebook/deit-small-distilled-patch16-224 (bare DeiTModel, distillation token)

NOTE ON RUNNING THESE: they download real weights from HuggingFace (a few
hundred MB total) — same cost class as the existing Mamba/SmolLM3 integration
tests, no special marker needed, but expect the first run to be slow.
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import DeiTForImageClassification, DeiTModel, ViTForImageClassification, ViTModel

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
    # Explicitly load the classification model to bypass boot_transformers auto-loader logic
    hf_model = ViTForImageClassification.from_pretrained(MODEL_VIT_CLASSIFIER)
    return TransformerBridge.boot_transformers(
        MODEL_VIT_CLASSIFIER, hf_model=hf_model, device="cpu"
    )


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

    def test_blocks_point_at_flat_layers_attribute(self, vit_bridge):
        """Current transformers has no ViTEncoder wrapper any more — blocks
        live directly on `vit.layers`, not `vit.encoder.layer`."""
        assert vit_bridge.adapter.component_mapping["blocks"].name == "vit.layers"


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
    # Explicitly load the bare model
    hf_model = ViTModel.from_pretrained(MODEL_VIT_BARE)
    return TransformerBridge.boot_transformers(MODEL_VIT_BARE, hf_model=hf_model, device="cpu")


class TestViTBareModel:
    def test_no_unembed_component(self, vit_bare_bridge):
        assert "unembed" not in vit_bare_bridge.adapter.component_mapping

    def test_blocks_point_at_flat_layers_attribute_no_prefix(self, vit_bare_bridge):
        assert vit_bare_bridge.adapter.component_mapping["blocks"].name == "layers"

    def test_forward_does_not_crash(self, vit_bare_bridge):
        pixel_values = _pixel_values()
        with torch.no_grad():
            vit_bare_bridge(pixel_values)  # must not raise

    def test_forward_return_value_is_not_a_plain_tensor(self, vit_bare_bridge):
        """KNOWN GAP — flagging, not necessarily desired behavior.
 
        For a bare ViTModel, HF's own forward returns BaseModelOutputWithPooling,
        which has no `.logits` and is NOT a tuple subclass. bridge.py's forward()
        only special-cases `.logits` or `isinstance(output, tuple)` before
        falling through to `logits = output` — so `bridge(pixel_values)` with
        the default return_type="logits" currently returns the raw HF output
        object, not a tensor. (Confirmed against transformers 5.8.1; this is a
        general bridge.py behavior, not vision-specific, so it's worth
        rechecking whether a later transformers release — e.g. 5.13.0 — changes
        the shape of BaseModelOutputWithPooling in a way that affects this.)
 
        Per review discussion, the desired fix lives in bridge.py, not this
        adapter, and there are two reasonable directions:
          (a) fall back to `output.last_hidden_state` when return_type="logits"
              but the output has no `.logits` attribute, or
          (b) raise a clear, actionable error instead of returning the raw HF
              output object silently.
        This test currently pins the raw-object behavior as a known gap so a
        bridge.py fix doesn't land silently unnoticed here. It should be
        updated to assert whichever of (a)/(b) is chosen once bridge.py is
        fixed, rather than left pinning the current gap indefinitely.
        """
        pixel_values = _pixel_values()
        with torch.no_grad():
            output = vit_bare_bridge(pixel_values)
        assert not isinstance(output, torch.Tensor)
        assert hasattr(output, "last_hidden_state")


# ---------------------------------------------------------------------------
# DeiTForImageClassification — distillation token must stay invisible
# Previously MODEL_DEIT_CLASSIFIER was declared but never used, and the only
# DeiT fixture (deit_bridge, below) loaded a bare DeiTModel — so the
# "deit."-prefix + real classifier-head path had no coverage. This fixture
# and test class close that gap with a real DeiTForImageClassification load.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def deit_classifier_bridge():
    hf_model = DeiTForImageClassification.from_pretrained(MODEL_DEIT_CLASSIFIER)
    return TransformerBridge.boot_transformers(
        MODEL_DEIT_CLASSIFIER, hf_model=hf_model, device="cpu"
    )
 
 
class TestDeiTClassifierBridge:
    def test_prefix_is_deit_for_classifier_model(self, deit_classifier_bridge):
        assert deit_classifier_bridge.adapter.component_mapping["blocks"].name == "deit.layers"
 
    def test_unembed_is_vision_classifier_head(self, deit_classifier_bridge):
        assert isinstance(deit_classifier_bridge.unembed, VisionClassifierHeadBridge)
 
    def test_forward_matches_hf(self, deit_classifier_bridge):
        pixel_values = _pixel_values()
        hf_model = deit_classifier_bridge.original_model
        with torch.no_grad():
            bridge_out = deit_classifier_bridge(pixel_values)
            hf_out = hf_model(pixel_values).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-4, f"Bridge vs HF max diff = {max_diff}"
 
    def test_forward_returns_correct_shape(self, deit_classifier_bridge):
        pixel_values = _pixel_values()
        with torch.no_grad():
            output = deit_classifier_bridge(pixel_values)
        num_labels = deit_classifier_bridge.original_model.config.num_labels
        assert output.shape == (1, num_labels)
      
# ---------------------------------------------------------------------------
# DeiT (bare model) — distillation token must stay invisible
# ---------------------------------------------------------------------------
# 1. Use the distilled checkpoint to actually test distillation token logic
MODEL_DEIT_DISTILLED = "facebook/deit-small-distilled-patch16-224"


@pytest.fixture(scope="module")
def deit_bridge():
    # 2. Load the bare Hugging Face model manually to bypass the
    # DeiTForImageClassificationWithTeacher dual-head that your adapter rejects.
    hf_model = DeiTModel.from_pretrained(MODEL_DEIT_DISTILLED)
    hf_model.config.architectures = ["DeiTModel"]

    # 3. Pass the instantiated HF model to boot_transformers.
    # (Assuming your boot_transformers accepts an hf_model kwarg like HookedTransformer)
    return TransformerBridge.boot_transformers(
        MODEL_DEIT_DISTILLED, hf_model=hf_model, device="cpu"
    )


class TestDeiTBridge:
    def test_prefix_is_empty_for_bare_model(self, deit_bridge):
        # Because we loaded a bare DeiTModel, the components are at the root.
        # Your prepare_model should assign prefix="" (not "deit.")
        assert deit_bridge.adapter.component_mapping["embed"].name == "embeddings"

    def test_forward_matches_hf(self, deit_bridge):
        pixel_values = _pixel_values()
        hf_model = deit_bridge.original_model

        with torch.no_grad():
            bridge_out = deit_bridge(pixel_values)
            # Bare models return a BaseModelOutput object with last_hidden_state
            hf_out = hf_model(pixel_values).last_hidden_state

        # Account for your adapter returning the raw HF object for bare models
        # (as you documented in TestViTBareModel)
        if not isinstance(bridge_out, torch.Tensor):
            bridge_out = bridge_out.last_hidden_state

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


# ---------------------------------------------------------------------------
# Regression test: prepare_model() must not assume `.encoder` exists.
# ---------------------------------------------------------------------------


class TestPrepareModelDoesNotAssumeEncoderWrapper:
    def test_minimal_bare_model_stub_does_not_crash(self):
        """Reproduces the AttributeError seen after upgrading transformers:
        'types.SimpleNamespace' object has no attribute 'encoder'.

        A stub with none of vit/deit/classifier/cls_classifier/
        distillation_classifier set should be treated like a bare, no-prefix
        model — prepare_model() must not reach for `hf_model.encoder.layer`
        (that wrapper module no longer exists in current transformers; blocks
        live directly on `<prefix>.layers`).
        """
        from transformer_lens.model_bridge.supported_architectures.vit import (
            ViTArchitectureAdapter,
        )

        hf_model = SimpleNamespace()

        adapter = ViTArchitectureAdapter.__new__(ViTArchitectureAdapter)
        adapter.cfg = None

        ViTArchitectureAdapter.prepare_model(adapter, hf_model)  # must not raise

        assert adapter.component_mapping["blocks"].name == "layers"
        assert "unembed" not in adapter.component_mapping
