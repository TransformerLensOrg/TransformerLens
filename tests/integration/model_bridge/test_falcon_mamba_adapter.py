"""Integration tests for the FalconMamba architecture adapter."""

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components.ssm_mixer import (
    SSMMixerBridge,
)

MODEL = "tiiuae/falcon-mamba-tiny-dev"


@pytest.fixture(scope="module")
def fm_bridge():
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(fm_bridge):
    torch.manual_seed(0)
    return torch.randint(0, fm_bridge.cfg.d_vocab - 10, (1, 8))


def _mixers(bridge):
    return [b.mixer for b in bridge.blocks if isinstance(getattr(b, "mixer", None), SSMMixerBridge)]


class TestFalconMambaBridgeCreation:
    def test_adapter_selected(self, fm_bridge):
        from transformer_lens.model_bridge.supported_architectures.falcon_mamba import (
            FalconMambaArchitectureAdapter,
        )

        assert isinstance(fm_bridge.adapter, FalconMambaArchitectureAdapter)

    def test_mixer_carries_rms_eps(self, fm_bridge):
        """The FalconMamba quirk: parameter-free RMS on B/C/dt, keyed on rms_eps."""
        mixers = _mixers(fm_bridge)
        assert mixers, "no SSM mixers found"
        assert getattr(mixers[0].original_component, "rms_eps", None) is not None


class TestFalconMambaForwardEquivalence:
    def test_forward_matches_hf(self, fm_bridge, sample_tokens):
        hf_model = fm_bridge.original_model
        with torch.no_grad():
            bridge_out = fm_bridge(sample_tokens)
            hf_out = hf_model(input_ids=sample_tokens).logits
        max_diff = (bridge_out - hf_out).abs().max().item()
        assert max_diff < 1e-5, f"Bridge vs HF max diff = {max_diff}"


class TestFalconMambaEagerScan:
    def test_eager_scan_applies_bcdt_rms(self, fm_bridge, sample_tokens):
        """The eager S6 scan must reproduce the fused path, which requires
        applying FalconMamba's B/C/dt RMS before dt_proj."""
        mixers = _mixers(fm_bridge)
        with torch.no_grad():
            base = fm_bridge(sample_tokens, use_cache=False)
            for m in mixers:
                m.eager_scan = True
            try:
                eager = fm_bridge(sample_tokens, use_cache=False)
            finally:
                for m in mixers:
                    m.eager_scan = False
        rel = (eager - base).abs().max().item() / max(base.abs().max().item(), 1e-8)
        assert rel < 1e-3, f"eager vs fused rel diff {rel:.2e}"

    def test_rms_is_load_bearing(self, fm_bridge, sample_tokens):
        """Counterfactual: hiding rms_eps from the eager scan must degrade parity,
        proving the RMS branch is actually engaged for FalconMamba."""
        mixers = _mixers(fm_bridge)
        saved = [m.original_component.rms_eps for m in mixers]
        with torch.no_grad():
            base = fm_bridge(sample_tokens, use_cache=False)
            for m in mixers:
                m.eager_scan = True
                m.original_component.rms_eps = None
            try:
                broken = fm_bridge(sample_tokens, use_cache=False)
            finally:
                for m, eps in zip(mixers, saved):
                    m.eager_scan = False
                    m.original_component.rms_eps = eps
        rel = (broken - base).abs().max().item() / max(base.abs().max().item(), 1e-8)
        assert rel > 1e-3, f"eager scan without RMS still matched (rel {rel:.2e})"


class TestFalconMambaStateReconstruction:
    def test_ssm_state_reconstruction_runs(self, fm_bridge, sample_tokens):
        with torch.no_grad():
            _, cache = fm_bridge.run_with_cache(sample_tokens, use_cache=False)
        mixer = _mixers(fm_bridge)[0]
        S = mixer.compute_ssm_state(cache, layer_idx=0)
        assert S.ndim == 4
        assert S.shape[2] == sample_tokens.shape[1]
        assert torch.isfinite(S).all()
