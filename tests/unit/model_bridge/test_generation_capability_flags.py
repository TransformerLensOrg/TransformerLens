"""Adapter-driven generation capability plumbing.

Architectures whose wrapped forward speaks no KV-cache protocol must be routed
to the full-prefix recompute path, and batching must be rejected outright where
padding cannot be masked (silent corruption otherwise).
"""

from types import SimpleNamespace

import pytest

from transformer_lens.model_bridge.bridge import TransformerBridge


class TestResolveGenerationCaching:
    def _resolve(self, *, kv, batched_ok, requested, batched):
        obj = SimpleNamespace(
            adapter=SimpleNamespace(supports_kv_cache=kv, supports_batched_generation=batched_ok),
            cfg=SimpleNamespace(architecture="FakeForCausalLM"),
        )
        return TransformerBridge._resolve_generation_caching(obj, requested, batched)

    def test_default_adapter_keeps_requested_cache_flag(self) -> None:
        assert self._resolve(kv=True, batched_ok=True, requested=True, batched=False) is True
        assert self._resolve(kv=True, batched_ok=True, requested=False, batched=False) is False

    def test_cache_free_arch_forces_recompute(self) -> None:
        """RWKV/HyenaDNA: the KV branch would pass use_cache into a forward that
        cannot honor it, so the request must be overridden, not obeyed."""
        assert self._resolve(kv=False, batched_ok=True, requested=True, batched=False) is False

    def test_batched_rejected_when_unsupported(self) -> None:
        with pytest.raises(NotImplementedError, match="Batched generation is not supported"):
            self._resolve(kv=False, batched_ok=False, requested=True, batched=True)

    def test_batched_allowed_when_supported(self) -> None:
        assert self._resolve(kv=True, batched_ok=True, requested=True, batched=True) is True

    def test_missing_attrs_default_permissive(self) -> None:
        """Adapters predating the flags must be unaffected."""
        obj = SimpleNamespace(adapter=SimpleNamespace(), cfg=SimpleNamespace(architecture="X"))
        assert TransformerBridge._resolve_generation_caching(obj, True, True) is True


class TestAdapterFlagDeclarations:
    """The Tier-A architectures declare the capability set generation relies on."""

    @pytest.mark.parametrize(
        "module_name,class_name",
        [
            ("rwkv", "RwkvArchitectureAdapter"),
            ("hyenadna", "HyenaDNAArchitectureAdapter"),
        ],
    )
    def test_generation_enabled_without_kv_cache(self, module_name, class_name) -> None:
        import importlib

        module = importlib.import_module(
            f"transformer_lens.model_bridge.supported_architectures.{module_name}"
        )
        adapter_cls = getattr(module, class_name)
        assert adapter_cls.supports_generation is True
        assert adapter_cls.supports_kv_cache is False
        assert adapter_cls.supports_batched_generation is False
        # P4 exercises generate(); leaving it out would re-introduce a carve-out.
        assert 4 in adapter_cls.applicable_phases


class TestMlpTupleHookSemantics:
    """RWKV's channel-mix returns (hidden, state); hook_out must see the tensor."""

    def _bridge(self, returns_tuple: bool):
        import torch

        from transformer_lens.model_bridge.generalized_components.mlp import MLPBridge

        class _Inner(torch.nn.Module):
            def forward(self, x):
                doubled = x * 2
                return (doubled, "state") if returns_tuple else doubled

        bridge = MLPBridge(name="mlp")
        bridge.set_original_component(_Inner())
        return bridge

    def test_hook_out_receives_tensor_not_tuple(self) -> None:
        """Pre-fix this handed hook functions the raw (hidden, state) tuple."""
        import torch

        seen = []
        bridge = self._bridge(returns_tuple=True)
        bridge.hook_out.add_hook(lambda t, hook: seen.append(t) or t)
        out = bridge(torch.ones(1, 2, 4))
        assert isinstance(out, tuple) and out[1] == "state", "tuple contract broken"
        assert len(seen) == 1 and isinstance(seen[0], torch.Tensor)

    def test_intervention_on_tuple_path_reaches_the_caller(self) -> None:
        """The re-packed tuple must carry the hook's edit, not the original."""
        import torch

        bridge = self._bridge(returns_tuple=True)
        bridge.hook_out.add_hook(lambda t, hook: torch.zeros_like(t))
        out = bridge(torch.ones(1, 2, 4))
        assert torch.equal(out[0], torch.zeros(1, 2, 4))

    def test_tensor_path_unchanged(self) -> None:
        import torch

        bridge = self._bridge(returns_tuple=False)
        out = bridge(torch.ones(1, 2, 4))
        assert isinstance(out, torch.Tensor)
        assert torch.equal(out, torch.full((1, 2, 4), 2.0))


class TestRopeOptionalOptOut:
    """NoPE architectures null position_embeddings by design; the missing-RoPE
    warning must stay silent for them and loud for everyone else."""

    def test_nope_adapters_declare_the_opt_out(self) -> None:
        import importlib

        for module_name, class_name in (
            ("exaone4", "_Exaone4AttentionBridge"),
            ("smollm3", "_SmolLM3AttentionBridge"),
        ):
            module = importlib.import_module(
                f"transformer_lens.model_bridge.supported_architectures.{module_name}"
            )
            bridge_cls = getattr(module, class_name)
            assert bridge_cls.rope_optional is True, f"{class_name} must opt out"

    def test_base_does_not_opt_out(self) -> None:
        from transformer_lens.model_bridge.generalized_components import (
            PositionEmbeddingsAttentionBridge,
        )

        assert PositionEmbeddingsAttentionBridge.rope_optional is False
