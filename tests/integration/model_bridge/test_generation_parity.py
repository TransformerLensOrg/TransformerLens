"""Generation parity for architectures that speak no KV-cache protocol.

RWKV (bespoke recurrent ``state``) and HyenaDNA (kwarg-less FFT convolution)
generate through the bridge's full-prefix recompute path. These pin that the
path is exact — not merely non-crashing — and that unsound batching raises.
"""

import pytest
import torch

pytest.importorskip("transformers")


def _tiny_rwkv_pair():
    """Seeded tiny RWKV: bridge + an INDEPENDENT reference with the same weights.

    The reference must be a separate instance — build_bridge_from_module wraps
    its argument in place, so driving that module with HF APIs afterwards runs
    through hooked components instead of the plain model.
    """
    import copy

    from transformers import RwkvConfig
    from transformers.models.rwkv.modeling_rwkv import RwkvForCausalLM

    from transformer_lens.model_bridge.sources._bridge_builder import (
        build_bridge_from_module,
    )

    cfg = RwkvConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        context_length=64,
        intermediate_size=64,
    )
    torch.manual_seed(0)
    ref = RwkvForCausalLM(cfg).eval()
    wrapped = RwkvForCausalLM(copy.deepcopy(cfg)).eval()
    wrapped.load_state_dict(ref.state_dict())
    bridge = build_bridge_from_module(
        wrapped, "RwkvForCausalLM", hf_config=copy.deepcopy(cfg), tokenizer=None, device="cpu"
    ).eval()
    return bridge, ref


class TestRwkvGeneration:
    def test_greedy_matches_hf(self) -> None:
        """HF's own cache-free generate is the reference; recompute must equal it."""
        bridge, hf = _tiny_rwkv_pair()
        ids = torch.tensor([[5, 12, 7, 33, 2]])
        with torch.no_grad():
            expected = hf.generate(ids.clone(), max_new_tokens=8, do_sample=False, use_cache=False)
            actual = bridge.generate(ids.clone(), max_new_tokens=8, temperature=0.0)
        assert torch.equal(actual, expected), f"{actual.tolist()} != {expected.tolist()}"

    def test_cache_request_is_overridden_not_obeyed(self) -> None:
        """use_past_kv_cache=True must not reach a forward that cannot honor it."""
        bridge, hf = _tiny_rwkv_pair()
        ids = torch.tensor([[5, 12, 7, 33, 2]])
        with torch.no_grad():
            expected = hf.generate(ids.clone(), max_new_tokens=6, do_sample=False, use_cache=False)
            actual = bridge.generate(
                ids.clone(), max_new_tokens=6, temperature=0.0, use_past_kv_cache=True
            )
        assert torch.equal(actual, expected)

    def test_batched_generation_rejected(self) -> None:
        """RwkvModel ignores attention_mask, so padded rows would corrupt state."""
        bridge, _ = _tiny_rwkv_pair()
        with pytest.raises(NotImplementedError, match="Batched generation is not supported"):
            bridge.generate(["one", "two"], max_new_tokens=2)


class TestHyenaDnaGeneration:
    """HyenaDNA ships remote code only; skip when it is unavailable offline."""

    @staticmethod
    def _bridge():
        from transformer_lens.model_bridge import TransformerBridge

        try:
            bridge = TransformerBridge.boot_transformers(
                "LongSafari/hyenadna-tiny-1k-seqlen-hf", device="cpu", trust_remote_code=True
            )
        except Exception as exc:  # offline / hub failure
            pytest.skip(f"hyenadna unavailable: {exc}")
        return bridge.eval()

    def test_greedy_matches_manual_recompute(self) -> None:
        """HF cannot drive this model (no GenerationMixin, kwarg-less forward),
        so the reference is a hand-rolled greedy loop on the wrapped module."""
        bridge = self._bridge()
        ids = torch.tensor([[7, 8, 9, 10, 7, 8, 9, 10]])
        with torch.no_grad():
            actual = bridge.generate(ids.clone(), max_new_tokens=6, temperature=0.0)
            expected = ids.clone()
            for _ in range(6):
                logits = bridge.original_model(expected).logits
                expected = torch.cat([expected, logits[:, -1, :].argmax(-1, keepdim=True)], dim=1)
        assert torch.equal(actual, expected), f"{actual.tolist()} != {expected.tolist()}"

    def test_batched_generation_rejected(self) -> None:
        bridge = self._bridge()
        with pytest.raises(NotImplementedError, match="Batched generation is not supported"):
            bridge.generate(["ACGT", "TGCA"], max_new_tokens=2)
