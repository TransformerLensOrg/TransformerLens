"""Native-sampler delegation for non-autoregressive architectures.

Diffusion LMs sample by iterative denoising, so the bridge delegates to the
model's own sampler rather than reusing the left-to-right loop. These pin the
routing, the per-architecture budget translation, and the error paths.
"""

import importlib


def _adapter_cls(module_name: str, class_name: str):
    module = importlib.import_module(
        f"transformer_lens.model_bridge.supported_architectures.{module_name}"
    )
    return getattr(module, class_name)


class TestNativeSamplerDeclarations:
    def test_autoregressive_adapters_declare_no_sampler(self) -> None:
        """The attribute must not leak onto ordinary decoders."""
        from transformer_lens.model_bridge.architecture_adapter import (
            ArchitectureAdapter,
        )

        assert ArchitectureAdapter.native_sampler is None


def _diffusion_adapter_classes() -> dict[str, type]:
    """Every registered adapter class that declares a native sampler, by module name."""
    from transformer_lens.factories.architecture_adapter_factory import (
        SUPPORTED_ARCHITECTURES,
    )

    return {
        cls.__module__.rsplit(".", 1)[-1]: cls
        for cls in set(SUPPORTED_ARCHITECTURES.values())
        if getattr(cls, "native_sampler", None) is not None
    }


def _uncovered_diffusion_adapters() -> list[str]:
    """Sampler-declaring adapters whose test file lacks a bound contract subclass.

    A missing or unimportable test module counts as uncovered — that is the
    state a brand-new adapter starts in.
    """
    from tests.unit.model_bridge.supported_architectures.helpers import (
        DiffusionContractTests,
    )

    missing = []
    for module_name, adapter_cls in sorted(_diffusion_adapter_classes().items()):
        try:
            test_module = importlib.import_module(
                f"tests.unit.model_bridge.supported_architectures.test_{module_name}_adapter"
            )
        except ModuleNotFoundError:
            missing.append(f"{adapter_cls.__name__} (no test_{module_name}_adapter.py)")
            continue
        covered = any(
            isinstance(obj, type)
            and issubclass(obj, DiffusionContractTests)
            and obj is not DiffusionContractTests
            and getattr(obj, "adapter_cls", None) is adapter_cls
            for obj in vars(test_module).values()
        )
        if not covered:
            missing.append(f"{adapter_cls.__name__} (test_{module_name}_adapter.py)")
    return missing


class TestContractCompleteness:
    """Per-adapter contract tests live in each adapter's own test file
    (subclassing helpers.DiffusionContractTests); this derives the required set
    from the factory so a new diffusion adapter cannot ship without them."""

    def test_every_diffusion_adapter_has_contract_coverage(self) -> None:
        missing = _uncovered_diffusion_adapters()
        assert not missing, (
            "Diffusion adapters without contract coverage: "
            f"{missing}. Add a DiffusionContractTests subclass (see "
            "helpers.DiffusionContractTests) to each adapter's test file."
        )

    def test_checker_flags_uncovered_adapter(self, monkeypatch) -> None:
        """The completeness check must bite, not vacuously pass."""
        from transformer_lens.factories import architecture_adapter_factory

        fake = type(
            "FakeDiffusionAdapter",
            (),
            {
                "native_sampler": "generate",
                "__module__": "transformer_lens.model_bridge.supported_architectures._fake_diffusion",
            },
        )
        patched = dict(architecture_adapter_factory.SUPPORTED_ARCHITECTURES)
        patched["FakeDiffusionLM"] = fake
        monkeypatch.setattr(architecture_adapter_factory, "SUPPORTED_ARCHITECTURES", patched)
        assert any("FakeDiffusionAdapter" in m for m in _uncovered_diffusion_adapters())


class TestSamplerBudgetTranslation:
    """Each sampler names the budget differently; adapters translate, not callers."""

    def _kwargs(self, module_name, class_name, max_new_tokens, prompt_len):
        adapter_cls = _adapter_cls(module_name, class_name)
        return adapter_cls.native_sampler_kwargs(adapter_cls, max_new_tokens, prompt_len)

    def test_dream_uses_max_new_tokens_and_steps(self) -> None:
        kwargs = self._kwargs("dream", "DreamArchitectureAdapter", 24, 6)
        assert kwargs["max_new_tokens"] == 24
        assert kwargs["steps"] == 24

    def test_llada2_rounds_gen_length_to_whole_blocks(self) -> None:
        """gen_length must cover whole blocks or the sampler's canvas is ragged."""
        kwargs = self._kwargs("llada2_moe", "LLaDA2MoeArchitectureAdapter", 40, 8)
        assert kwargs["gen_length"] % kwargs["block_length"] == 0
        assert kwargs["gen_length"] >= 40

    def test_llada2_block_length_never_exceeds_budget(self) -> None:
        kwargs = self._kwargs("llada2_moe", "LLaDA2MoeArchitectureAdapter", 4, 8)
        assert kwargs["block_length"] <= 4

    def test_gidd_max_length_is_generated_tokens_not_canvas(self) -> None:
        """Gidd's windows start at prompt_length and span max_length, so folding
        the prompt into it would over-generate by prompt_len tokens."""
        kwargs = self._kwargs("gidd", "GiddArchitectureAdapter", 16, 8)
        assert kwargs["max_length"] == 16

    def test_base_default_is_plain_max_new_tokens(self) -> None:
        from transformer_lens.model_bridge.architecture_adapter import (
            ArchitectureAdapter,
        )

        assert ArchitectureAdapter.native_sampler_kwargs(ArchitectureAdapter, 12, 3) == {
            "max_new_tokens": 12
        }


class TestBenchmarkRouting:
    """Verification must exercise diffusion sampling, not skip these architectures."""

    def test_resolver_picks_diffusion_generate(self) -> None:
        from types import SimpleNamespace

        from transformer_lens.benchmarks.generation import resolve_text_generator

        bridge = SimpleNamespace(
            adapter=SimpleNamespace(supports_generation=False, native_sampler="generate"),
            generate="AR",
            diffusion_generate="DIFFUSION",
        )
        assert resolve_text_generator(bridge) == "DIFFUSION"

    def test_resolver_picks_generate_for_autoregressive(self) -> None:
        from types import SimpleNamespace

        from transformer_lens.benchmarks.generation import resolve_text_generator

        bridge = SimpleNamespace(
            adapter=SimpleNamespace(supports_generation=True, native_sampler=None),
            generate="AR",
            diffusion_generate="DIFFUSION",
        )
        assert resolve_text_generator(bridge) == "AR"

    def test_resolver_returns_none_when_neither(self) -> None:
        from types import SimpleNamespace

        from transformer_lens.benchmarks.generation import resolve_text_generator

        bridge = SimpleNamespace(
            adapter=SimpleNamespace(supports_generation=False, native_sampler=None),
            generate="AR",
            diffusion_generate="DIFFUSION",
        )
        assert resolve_text_generator(bridge) is None


class TestPromptNormalization:
    """Samplers disagree on whether the prompt is returned; the bridge must not."""

    def _run(self, sampler_return, prompt):
        import torch

        from transformer_lens.model_bridge.bridge import TransformerBridge

        class _Adapter:
            native_sampler = "generate"
            supports_generation = False

            def native_sampler_kwargs(self, max_new_tokens, prompt_len):
                return {}

        class _Model:
            def generate(self, tokens, **kwargs):
                return sampler_return

        class _Cfg:
            architecture = "FakeDiffusionLM"
            device = "cpu"

        obj = type(
            "B",
            (),
            {
                "adapter": _Adapter(),
                "original_model": _Model(),
                "cfg": _Cfg(),
                "tokenizer": None,
            },
        )()
        return TransformerBridge.diffusion_generate.__wrapped__(obj, prompt, max_new_tokens=2)

    def test_continuation_only_gets_prompt_prepended(self) -> None:
        """LLaDA2 slices the prompt off its return."""
        import torch

        prompt = torch.tensor([[1, 2, 3]])
        out = self._run(torch.tensor([[7, 8]]), prompt)
        assert out.tolist() == [[1, 2, 3, 7, 8]]

    def test_full_canvas_is_left_alone(self) -> None:
        """Dream/Gidd already include the prompt — must not be duplicated."""
        import torch

        prompt = torch.tensor([[1, 2, 3]])
        out = self._run(torch.tensor([[1, 2, 3, 7, 8]]), prompt)
        assert out.tolist() == [[1, 2, 3, 7, 8]]


class TestRouterNestedProcessedWeights:
    """JetMoe's TopKGating holds its Linear at router.layer — processing hands the
    router bridge nested keys and the wrapped module has no .weight of its own."""

    def _router(self):
        import torch

        from transformer_lens.model_bridge.generalized_components.moe import (
            MoERouterBridge,
        )

        class _Gating(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(8, 4, bias=False)

        bridge = MoERouterBridge(name="router", logits_index=-1)
        gating = _Gating()
        bridge.set_original_component(gating)
        return bridge, gating

    def test_nested_keys_written_by_dotted_path(self) -> None:
        import torch

        bridge, gating = self._router()
        new = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        bridge.set_processed_weights({"layer.weight": new.clone()})
        assert torch.equal(gating.layer.weight.detach(), new)

    def test_shape_mismatch_raises(self) -> None:
        import pytest as _pytest
        import torch

        bridge, _ = self._router()
        with _pytest.raises(ValueError, match="does not match"):
            bridge.set_processed_weights({"layer.weight": torch.zeros(3, 3)})
