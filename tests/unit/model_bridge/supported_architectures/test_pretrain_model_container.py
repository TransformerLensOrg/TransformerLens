"""Container, builder, and wrapped-model lifecycle behavior for
PretrainModelContainer and build_pretrain_bridge -- kwarg filtering,
output-contract normalization, train/eval mode propagation, and
dtype/tied-weight preservation. Hook registration tests for the hidden
MLP delegate
(implemented by DenseOrMoEFeedForwardBridge) live here too, since what
they verify -- real parameters staying reachable through
`original_model.inner` -- is a lifecycle concern, not a mapping one.
Adapter-mapping-level behavior lives in test_pretrain_adapter.py.

Fully self-contained (see _pretrain_mocks.py): tiny mock `nn.Module`s, no
external package dependency, no network access.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from transformer_lens.model_bridge.supported_architectures.pretrain import (
    PretrainModelContainer,
    _LogitsAttrDict,
    build_pretrain_bridge,
)

from ._pretrain_mocks import (
    ForwardStrict,
    ForwardVarKwargs,
    TinyPretrainModel,
    make_cfg,
)

_make_cfg = make_cfg  # local alias, matches call sites below


class TestPretrainModelContainerKwargFiltering:
    def test_output_attentions_is_stripped_for_strict_signature(self) -> None:
        """The one TransformerBridge-injected kwarg (run_with_cache
        forces output_attentions=True) must not reach a source forward with
        no **kwargs catch-all."""
        model = ForwardStrict()
        container = PretrainModelContainer(model)
        container(torch.tensor([[1, 2, 3]]), targets=None, output_attentions=True)
        assert model.seen_kwargs == {"targets": None}

    def test_output_attentions_is_stripped_even_for_var_kwargs_signature(self) -> None:
        """Stripped unconditionally, regardless of whether the source would
        have accepted it anyway -- keeps behavior uniform across source
        forward signatures."""
        model = ForwardVarKwargs()
        container = PretrainModelContainer(model)
        container(torch.tensor([[1, 2, 3]]), output_attentions=True, some_other_kwarg=42)
        assert model.seen_kwargs == {"some_other_kwarg": 42}

    def test_unrelated_kwarg_typo_is_not_silently_swallowed(self) -> None:
        """A genuine caller mistake (e.g. `target=` instead of `targets=`)
        must raise, not be silently discarded alongside the one kwarg this
        container is actually responsible for stripping."""
        model = ForwardStrict()
        container = PretrainModelContainer(model)
        with pytest.raises(TypeError):
            container(torch.tensor([[1, 2, 3]]), target=torch.tensor([1]))


class TestPretrainModelContainerHookRegistration:
    """The delegate is hidden from nn.Module registration (see
    DenseOrMoEFeedForwardBridge), so blocks.{i}.mlp.hook_in/out are the
    only publicly registered MLP hook points. Forward, gradients, dtype
    conversion, and state_dict() still reach the hidden delegate's real
    weights through the raw wrapped model."""

    def _build(self):
        cfg = _make_cfg(n_layers=1)
        model = TinyPretrainModel(d_model=16, n_heads=2, n_layers=1, d_ff=32, vocab_size=64)
        return build_pretrain_bridge(model, cfg)

    def test_only_the_dispatcher_hook_out_is_registered(self) -> None:
        bridge = self._build()
        tokens = torch.randint(0, 64, (1, 4))
        with torch.no_grad():
            _, cache = bridge.run_with_cache(tokens)
        assert "blocks.0.mlp.hook_out" in cache
        assert "blocks.0.mlp._delegate.hook_out" not in cache

    def test_broad_hook_selector_applies_intervention_exactly_once(self) -> None:
        bridge = self._build()
        tokens = torch.randint(0, 64, (1, 4))
        calls: list[str] = []

        def recording_add_one(value: torch.Tensor, hook) -> torch.Tensor:
            calls.append(hook.name)
            return value + 1.0

        def broad_filter(name: str) -> bool:
            return "mlp" in name and "hook_out" in name

        with torch.no_grad():
            bridge.run_with_hooks(tokens, fwd_hooks=[(broad_filter, recording_add_one)])

        assert calls == ["blocks.0.mlp.hook_out"]

    def test_gradients_still_reach_the_hidden_delegates_weights(self) -> None:
        bridge = self._build()
        tokens = torch.randint(0, 64, (1, 4))
        bridge(tokens).sum().backward()
        mlp = bridge.original_model.inner.blocks[0].mlp
        assert mlp.gate.weight.grad is not None
        assert mlp.gate.weight.grad.abs().sum().item() > 0

    def test_bridge_to_dtype_reaches_the_hidden_delegates_weights(self) -> None:
        bridge = self._build()
        bridge.to(dtype=torch.float64)

        assert bridge.original_model.inner.embed.weight.dtype == torch.float64
        assert bridge.original_model.inner.blocks[0].mlp.gate.weight.dtype == torch.float64

        tokens = torch.randint(0, 64, (1, 4))
        with torch.no_grad():
            output = bridge(tokens)
        assert output.dtype == torch.float64

    def test_state_dict_includes_the_hidden_delegates_weights(self) -> None:
        bridge = self._build()
        source_weight = bridge.original_model.inner.blocks[0].mlp.gate.weight

        state = bridge.state_dict()
        matching = [v for k, v in state.items() if k.endswith("blocks.0.mlp.gate.weight")]

        assert len(matching) == 1
        torch.testing.assert_close(matching[0], source_weight)

    def test_repeated_hook_lifecycle_still_works(self) -> None:
        """Guards against a future assumption that every internally-used
        GeneralizedComponent must be registered -- the dispatcher's own
        hooks are, and that's what reset_hooks()/run_with_cache rely on."""
        bridge = self._build()
        tokens = torch.randint(0, 64, (1, 4))
        with torch.no_grad():
            bridge.run_with_cache(tokens)
            bridge.reset_hooks()
            _, cache = bridge.run_with_cache(tokens)
        assert "blocks.0.mlp.hook_out" in cache


class TestTrainEvalModePropagation:
    """bridge.train()/.eval() propagate to the wrapped model:
    build_pretrain_bridge reassigns the returned bridge's class to a small
    generated TransformerBridge subclass (see
    _get_mode_propagating_bridge_class) whose overridden train() also sets
    mode on original_model; inherited eval() calls train(False), so it
    reaches the override too without needing its own override. This
    closes a gap in TransformerBridge's own train()/eval(), which only
    walk component_mapping and never reach original_model on their own."""

    def test_bridge_eval_propagates_to_wrapped_model(self) -> None:
        cfg = _make_cfg(n_layers=1)
        model = TinyPretrainModel(d_model=16, n_heads=2, n_layers=1, d_ff=32, vocab_size=64)
        bridge = build_pretrain_bridge(model, cfg)

        assert model.training is True
        bridge.eval()
        assert model.training is False

    def test_bridge_train_propagates_to_wrapped_model(self) -> None:
        cfg = _make_cfg(n_layers=1)
        model = TinyPretrainModel(d_model=16, n_heads=2, n_layers=1, d_ff=32, vocab_size=64)
        bridge = build_pretrain_bridge(model, cfg)

        model.eval()
        assert model.training is False
        bridge.train()
        assert model.training is True

    def test_caller_can_still_set_mode_directly_on_their_own_model_reference(self) -> None:
        """Setting mode via the raw model reference still works and stays
        in sync -- both paths write to the same underlying module."""
        cfg = _make_cfg(n_layers=1)
        model = TinyPretrainModel(d_model=16, n_heads=2, n_layers=1, d_ff=32, vocab_size=64)
        build_pretrain_bridge(model, cfg)

        model.eval()
        assert model.training is False
        model.train()
        assert model.training is True

    def test_train_and_eval_return_the_bridge_itself(self) -> None:
        """nn.Module convention: train()/eval() return self, so callers can
        chain (`model.train().to(device)`, etc). The generated subclass
        must preserve this, not just the mode-propagation side effect."""
        cfg = _make_cfg(n_layers=1)
        model = TinyPretrainModel(d_model=16, n_heads=2, n_layers=1, d_ff=32, vocab_size=64)
        bridge = build_pretrain_bridge(model, cfg)

        assert bridge.train(False) is bridge
        assert bridge.train(True) is bridge
        assert bridge.eval() is bridge

    def test_bridge_can_be_reconstructed_from_supported_state(self) -> None:
        """Rebuild the bridge from the source model's *raw* state_dict and
        confirm behavior survives the round trip.

        The state_dict must be snapshotted BEFORE `build_pretrain_bridge`
        wraps the model's submodules in place; taken after, keys become
        `embed._original_component.weight` and a fresh unwrapped model
        can't load them.
        """
        cfg = _make_cfg(n_layers=1)
        model = TinyPretrainModel(d_model=16, n_heads=2, n_layers=1, d_ff=32, vocab_size=64)

        # Snapshot BEFORE wrapping -- see docstring.
        raw_model_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

        bridge = build_pretrain_bridge(model, cfg)
        bridge.eval()

        restored_model = TinyPretrainModel(
            d_model=16, n_heads=2, n_layers=1, d_ff=32, vocab_size=64
        )
        restored_model.load_state_dict(raw_model_state)

        restored_bridge = build_pretrain_bridge(restored_model, cfg)
        restored_bridge.eval()

        tokens = torch.randint(0, 64, (1, 4))
        with torch.no_grad():
            expected = bridge(tokens)
            actual = restored_bridge(tokens)

        expected_logits = expected.logits if hasattr(expected, "logits") else expected
        actual_logits = actual.logits if hasattr(actual, "logits") else actual
        torch.testing.assert_close(actual_logits, expected_logits, atol=0, rtol=0)

        assert restored_bridge.training is False
        assert restored_model.training is False

        restored_bridge.train()
        assert restored_model.training is True

    def test_repeated_construction_reuses_the_same_generated_class(self) -> None:
        """Guards the cache's stated purpose: build_pretrain_bridge should
        not generate a new mode-propagating subclass on every call -- two
        independently-built bridges from the same base bridge class must
        share one generated class."""
        cfg_a = _make_cfg(n_layers=1)
        model_a = TinyPretrainModel(d_model=16, n_heads=2, n_layers=1, d_ff=32, vocab_size=64)
        bridge_a = build_pretrain_bridge(model_a, cfg_a)

        cfg_b = _make_cfg(n_layers=1)
        model_b = TinyPretrainModel(d_model=16, n_heads=2, n_layers=1, d_ff=32, vocab_size=64)
        bridge_b = build_pretrain_bridge(model_b, cfg_b)

        assert type(bridge_a) is type(bridge_b)


class TestDtypeAndTiedWeightPreservation:
    def test_non_default_dtype_and_tied_weights_survive_construction(self) -> None:
        torch.manual_seed(0)
        model = TinyPretrainModel(
            d_model=16, n_heads=2, n_layers=1, d_ff=32, vocab_size=64, tie_embeddings=True
        ).to(torch.float64)
        assert model.lm_head.weight is model.embed.weight

        cfg = _make_cfg(n_layers=1)
        bridge = build_pretrain_bridge(model, cfg)

        assert next(model.parameters()).dtype == torch.float64
        assert model.lm_head.weight is model.embed.weight

        tokens = torch.randint(0, 64, (1, 3))
        with torch.no_grad():
            output = bridge(tokens)
        assert output.dtype == torch.float64


class TestBuildPretrainBridgeErgonomics:
    def test_explicit_model_name_is_forwarded(self) -> None:
        cfg = _make_cfg(n_layers=1)
        model = TinyPretrainModel(d_model=16, n_heads=2, n_layers=1, d_ff=32, vocab_size=64)
        bridge = build_pretrain_bridge(model, cfg, model_name="my-custom-name")
        assert bridge.cfg.model_name == "my-custom-name"

    def test_omitting_optional_kwargs_still_works(self) -> None:
        """The default (no device/dtype/model_name passed) path, exercised
        everywhere else in this file, must keep working unchanged."""
        cfg = _make_cfg(n_layers=1)
        model = TinyPretrainModel(d_model=16, n_heads=2, n_layers=1, d_ff=32, vocab_size=64)
        bridge = build_pretrain_bridge(model, cfg)
        tokens = torch.randint(0, 64, (1, 3))
        with torch.no_grad():
            output = bridge(tokens)
        assert output.shape == (1, 3, 64)


class TestPretrainModelContainerOutputContract:
    def test_container_normalizes_dict_output_to_logits_attr_dict(self) -> None:
        model = TinyPretrainModel(
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            vocab_size=64,
        )
        container = PretrainModelContainer(model)
        tokens = torch.randint(0, 64, (1, 3))

        with torch.no_grad():
            output = container(tokens)

        assert isinstance(output, _LogitsAttrDict)
        assert output.logits is output["logits"]
        assert output.logits.shape == (1, 3, 64)

    def test_container_returns_a_fresh_wrapper_on_each_call(self) -> None:
        model = TinyPretrainModel(
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            vocab_size=64,
        )
        container = PretrainModelContainer(model)
        tokens = torch.randint(0, 64, (1, 3))

        with torch.no_grad():
            out1 = container(tokens)
            out2 = container(tokens)

        assert isinstance(out1, _LogitsAttrDict)
        assert isinstance(out2, _LogitsAttrDict)
        assert out1 is not out2
        assert out1["logits"] is not out2["logits"]
        torch.testing.assert_close(out1["logits"], out2["logits"])

    def test_container_does_not_reuse_or_mutate_output_mapping(self) -> None:
        model = TinyPretrainModel(
            d_model=16,
            n_heads=2,
            n_layers=1,
            d_ff=32,
            vocab_size=64,
        ).eval()
        container = PretrainModelContainer(model)
        tokens = torch.randint(0, 64, (1, 3))

        with torch.no_grad():
            out1 = container(tokens)

        out1["sentinel"] = True

        with torch.no_grad():
            out2 = container(tokens)

        assert out1 is not out2
        assert "sentinel" not in out2
        torch.testing.assert_close(out1["logits"], out2["logits"])

    def test_dict_missing_logits_key_raises_value_error(self) -> None:
        class ForwardWrongKey(nn.Module):
            def forward(self, tokens: torch.Tensor) -> dict:
                return {"scores": tokens.float()}

        container = PretrainModelContainer(ForwardWrongKey())
        with pytest.raises(ValueError, match="'logits'"):
            container(torch.tensor([[1, 2, 3]]))

    def test_dict_with_heterogeneous_keys_still_raises_value_error(self) -> None:
        """sorted(output.keys()) would raise TypeError on non-comparable
        mixed key types (e.g. str and int); the error message must use
        list() instead, so an invalid model output produces the intended
        ValueError, not an unrelated sorting error."""

        class ForwardHeterogeneousKeys(nn.Module):
            def forward(self, tokens: torch.Tensor) -> dict:
                return {"scores": tokens.float(), 1: "metadata"}

        container = PretrainModelContainer(ForwardHeterogeneousKeys())
        with pytest.raises(ValueError, match="'logits'"):
            container(torch.tensor([[1, 2, 3]]))

    def test_dict_with_non_tensor_logits_raises_type_error(self) -> None:
        class ForwardBadLogitsType(nn.Module):
            def forward(self, tokens: torch.Tensor) -> dict:
                return {"logits": "not a tensor"}

        container = PretrainModelContainer(ForwardBadLogitsType())
        with pytest.raises(TypeError, match="torch.Tensor"):
            container(torch.tensor([[1, 2, 3]]))

    def test_already_logits_attr_dict_passes_through_by_identity(self) -> None:
        """An already-wrapped output must not be re-wrapped."""
        already_wrapped = _LogitsAttrDict({"logits": torch.zeros(1)})

        class ForwardAlreadyWrapped(nn.Module):
            def forward(self, tokens: torch.Tensor) -> _LogitsAttrDict:
                return already_wrapped

        container = PretrainModelContainer(ForwardAlreadyWrapped())
        output = container(torch.tensor([[1, 2, 3]]))
        assert output is already_wrapped

    def test_malformed_logits_attr_dict_raises_value_error_not_keyerror(self) -> None:
        """Not a realistic target-model failure (_LogitsAttrDict is
        private), but closes the contract: a malformed instance gets the
        same clear ValueError an ordinary dict missing 'logits' gets,
        rather than a raw KeyError leaking out."""
        malformed = _LogitsAttrDict({"scores": torch.zeros(1)})

        class ForwardMalformedWrapped(nn.Module):
            def forward(self, tokens: torch.Tensor) -> _LogitsAttrDict:
                return malformed

        container = PretrainModelContainer(ForwardMalformedWrapped())
        with pytest.raises(ValueError, match="'logits'"):
            container(torch.tensor([[1, 2, 3]]))

    def test_hf_style_object_with_logits_attribute_passes_through_unchanged(self) -> None:
        """A source model that already satisfies hasattr(output, 'logits')
        on its own (an HF-ModelOutput-like object, not a dict) needs no
        normalization at all."""

        class FakeModelOutput:
            def __init__(self, logits: torch.Tensor):
                self.logits = logits

        already_hf_style = FakeModelOutput(torch.zeros(1))

        class ForwardHFStyle(nn.Module):
            def forward(self, tokens: torch.Tensor) -> FakeModelOutput:
                return already_hf_style

        container = PretrainModelContainer(ForwardHFStyle())
        output = container(torch.tensor([[1, 2, 3]]))
        assert output is already_hf_style

    def test_property_backed_logits_is_only_evaluated_once(self) -> None:
        """.logits must be read once and cached locally, not re-evaluated
        on every access -- matters for a property-backed .logits, which
        could otherwise do redundant (or side-effecting) work per access."""
        access_count = 0

        class PropertyBackedOutput:
            @property
            def logits(self) -> torch.Tensor:
                nonlocal access_count
                access_count += 1
                return torch.zeros(1)

        class ForwardPropertyBacked(nn.Module):
            def forward(self, tokens: torch.Tensor) -> PropertyBackedOutput:
                return PropertyBackedOutput()

        container = PretrainModelContainer(ForwardPropertyBacked())
        container(torch.tensor([[1, 2, 3]]))
        assert access_count == 1

    def test_object_with_non_tensor_logits_attribute_raises_type_error(self) -> None:
        class BadModelOutput:
            logits = "not a tensor"

        class ForwardBadModelOutput(nn.Module):
            def forward(self, tokens: torch.Tensor) -> BadModelOutput:
                return BadModelOutput()

        container = PretrainModelContainer(ForwardBadModelOutput())
        with pytest.raises(TypeError, match=r"\.logits"):
            container(torch.tensor([[1, 2, 3]]))

    def test_bare_tensor_output_passes_through_unchanged(self) -> None:
        class ForwardBareTensor(nn.Module):
            def forward(self, tokens: torch.Tensor) -> torch.Tensor:
                return tokens.float()

        container = PretrainModelContainer(ForwardBareTensor())
        tokens = torch.tensor([[1, 2, 3]])
        output = container(tokens)
        assert torch.equal(output, tokens.float())

    def test_tuple_with_tensor_first_element_passes_through_unchanged(self) -> None:
        """TransformerBridge extracts output[0] as logits for tuple
        returns -- no normalization needed."""

        class ForwardTuple(nn.Module):
            def forward(self, tokens: torch.Tensor) -> tuple:
                return (tokens.float(), "some_aux_value")

        container = PretrainModelContainer(ForwardTuple())
        output = container(torch.tensor([[1, 2, 3]]))
        assert isinstance(output, tuple)
        assert output[1] == "some_aux_value"

    def test_tuple_with_non_tensor_first_element_raises_type_error(self) -> None:
        class ForwardBadTuple(nn.Module):
            def forward(self, tokens: torch.Tensor) -> tuple:
                return ("not a tensor", tokens)

        container = PretrainModelContainer(ForwardBadTuple())
        with pytest.raises(TypeError, match="torch.Tensor"):
            container(torch.tensor([[1, 2, 3]]))

    def test_list_output_raises_type_error(self) -> None:
        class ForwardList(nn.Module):
            def forward(self, tokens: torch.Tensor) -> list:
                return [tokens.float()]

        container = PretrainModelContainer(ForwardList())
        with pytest.raises(TypeError, match="must return"):
            container(torch.tensor([[1, 2, 3]]))

    def test_string_output_raises_type_error(self) -> None:
        class ForwardString(nn.Module):
            def forward(self, tokens: torch.Tensor) -> str:
                return "wrong"

        container = PretrainModelContainer(ForwardString())
        with pytest.raises(TypeError, match="must return"):
            container(torch.tensor([[1, 2, 3]]))
