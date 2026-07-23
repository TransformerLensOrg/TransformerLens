"""n_devices=2 validation for boot_transformers — real model, >= 2 CUDA devices.

Run on a multi-GPU box (no extras needed beyond the default install):

    uv run pytest tests/acceptance/model_bridge/test_bridge_multigpu.py -m multigpu -v

Every test compares n_devices=2 (accelerate "balanced" dispatch with concrete
per-device memory budgets) against the single-device cuda:0 path on the same
model, so a pass means the split introduces no logits/cache/hook/generation
drift. device_map variants run separately (test_bridge_multigpu_device_map.py)
in their own process.
"""
from __future__ import annotations

import warnings

import pytest
import torch

from ._bridge_multigpu_common import (
    MULTIGPU_MARKS,
    N_LAYERS,
    PROMPT,
    assert_caches_match,
    assert_logits_match,
    bridge_pair_fixture,
    cuda_indices,
)

pytestmark = MULTIGPU_MARKS

bridges = bridge_pair_fixture(n_devices=2)


class TestNDevicesPlacement:
    def test_config_and_params_span_both_devices(self, bridges):
        single, multi = bridges
        assert single.cfg.n_devices == 1
        assert multi.cfg.n_devices == 2
        assert cuda_indices(multi) == {0, 1}
        # cfg.device is the embedding device — where inputs and sampling tensors live.
        emb_device = next(multi.original_model.get_input_embeddings().parameters()).device
        assert torch.device(multi.cfg.device) == emb_device

    def test_stacked_weights_and_accumulated_bias_gather(self, bridges):
        """Stacked-weight properties must gather per-block tensors onto cfg.device
        (torch.stack requires one device) without raising across the split."""
        _, multi = bridges
        W_Q = multi.W_Q
        assert W_Q.shape[0] == multi.cfg.n_layers
        assert W_Q.device == torch.device(multi.cfg.device)
        bias = multi.accumulated_bias(layer=multi.cfg.n_layers - 1)
        assert bias.shape == (multi.cfg.d_model,)

    def test_to_device_move_is_refused_but_split_survives(self, bridges):
        """.to(device) on a dispatched bridge must warn and keep the split —
        collapsing accelerate's placement would silently break the layer map."""
        _, multi = bridges
        with pytest.warns(UserWarning):
            multi.to("cuda:0")
        assert cuda_indices(multi) == {0, 1}


class TestNDevicesForwardParity:
    def test_logits_match_single_device(self, bridges):
        single, multi = bridges
        assert_logits_match(single, multi)

    def test_loss_matches_single_device(self, bridges):
        single, multi = bridges
        toks = single.to_tokens(PROMPT)
        loss1 = single(toks, return_type="loss").item()
        loss2 = multi(toks.to(multi.cfg.device), return_type="loss").item()
        assert loss2 == pytest.approx(loss1, abs=1e-4)

    def test_cache_parity_and_per_layer_devices(self, bridges):
        """Every cached hook must match the single-device run, and the multi-device
        cache must actually hold tensors on more than one device (run_with_cache
        leaves activations where their layer ran)."""
        single, multi = bridges
        toks = single.to_tokens(PROMPT)
        _, cache1 = single.run_with_cache(toks)
        _, cache2 = multi.run_with_cache(toks.to(multi.cfg.device))
        assert_caches_match(cache1, cache2)
        devices = {cache2[name].device for name in cache2}
        assert len(devices) > 1, f"expected per-layer devices in the cache, got {devices}"

    def test_run_with_cache_device_kwarg_warned_and_ignored(self, bridges):
        _, multi = bridges
        toks = multi.to_tokens(PROMPT)
        with pytest.warns(UserWarning, match="per-layer devices"):
            multi.run_with_cache(toks, device="cuda:0")


class TestNDevicesHookParity:
    @pytest.mark.parametrize(
        "hook_name",
        [
            "blocks.0.hook_mlp_out",  # first device
            f"blocks.{N_LAYERS - 1}.hook_mlp_out",  # second device under a balanced split
        ],
    )
    def test_editing_hook_bites_and_matches_single_device(self, bridges, hook_name):
        """A mutating hook must fire on whichever device owns its layer and steer
        the logits identically to the single-device run."""
        single, multi = bridges
        toks = single.to_tokens(PROMPT)

        def zero_out(tensor, hook):
            return torch.zeros_like(tensor)

        clean2 = multi(toks.to(multi.cfg.device)).detach().float().cpu()
        l1 = single.run_with_hooks(toks, fwd_hooks=[(hook_name, zero_out)])
        l2 = multi.run_with_hooks(toks.to(multi.cfg.device), fwd_hooks=[(hook_name, zero_out)])
        l1, l2 = l1.detach().float().cpu(), l2.detach().float().cpu()
        assert not torch.allclose(
            clean2, l2, atol=1e-4, rtol=1e-4
        ), f"{hook_name}: hook edit was a no-op under n_devices=2"
        assert torch.equal(l1.argmax(dim=-1), l2.argmax(dim=-1))
        assert torch.allclose(
            l1, l2, atol=1e-4, rtol=1e-4
        ), f"{hook_name}: max abs diff {(l1 - l2).abs().max().item():.3e}"


class TestNDevicesGeneration:
    def test_greedy_generate_matches_single_device(self, bridges):
        single, multi = bridges
        out1 = single.generate(PROMPT, max_new_tokens=5, do_sample=False, verbose=False)
        out2 = multi.generate(PROMPT, max_new_tokens=5, do_sample=False, verbose=False)
        assert isinstance(out2, str) and len(out2) > len(PROMPT)
        assert out1 == out2

    def test_greedy_generate_without_kv_cache(self, bridges):
        """The no-cache path re-runs the full prefix each step — every step crosses
        the device boundary; must agree with the cached path."""
        _, multi = bridges
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # HF may warn about cacheless generation
            out_cached = multi.generate(PROMPT, max_new_tokens=5, do_sample=False, verbose=False)
            out_nocache = multi.generate(
                PROMPT, max_new_tokens=5, do_sample=False, use_past_kv_cache=False, verbose=False
            )
        assert out_cached == out_nocache
