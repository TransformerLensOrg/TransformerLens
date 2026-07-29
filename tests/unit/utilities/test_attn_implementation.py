"""Unit tests for transformer_lens.utilities.attn_implementation."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

from transformer_lens.utilities.attn_implementation import force_eager_attention


def _config(nested: bool = False) -> SimpleNamespace:
    cfg = SimpleNamespace(_attn_implementation="sdpa")
    if nested:
        cfg.text_config = SimpleNamespace(_attn_implementation="sdpa")
    return cfg


def _layer(with_attn: bool = True, with_attn_config: bool = True) -> SimpleNamespace:
    layer = SimpleNamespace()
    if with_attn:
        layer.self_attn = SimpleNamespace()
        if with_attn_config:
            layer.self_attn.config = SimpleNamespace(_attn_implementation="sdpa")
    return layer


class TestPublicSetterPath:
    def test_public_setter_is_called_and_trusted(self):
        """A working set_attn_implementation is preferred; no config fallback fires."""

        class Model:
            def __init__(self):
                self.config = _config()
                self.calls = []

            def set_attn_implementation(self, impl):
                self.calls.append(impl)

        model = Model()
        force_eager_attention(model)
        assert model.calls == ["eager"]
        # Setter succeeded, so the fallback must not have touched the config.
        assert model.config._attn_implementation == "sdpa"

    def test_raising_setter_falls_back_to_config(self):
        """HF validates on some models; the raise is swallowed and config writes land."""

        class Model:
            def __init__(self):
                self.config = _config(nested=True)

            def set_attn_implementation(self, impl):
                raise ValueError("attn implementation not supported")

        model = Model()
        force_eager_attention(model)
        assert model.config._attn_implementation == "eager"
        assert model.config.text_config._attn_implementation == "eager"


class TestConfigFallback:
    def test_no_setter_writes_config_and_text_config(self):
        model = SimpleNamespace(config=_config(nested=True))
        force_eager_attention(model)
        assert model.config._attn_implementation == "eager"
        assert model.config.text_config._attn_implementation == "eager"

    def test_config_without_attr_is_left_alone(self):
        """The attribute is only flipped, never created on foreign configs."""
        model = SimpleNamespace(config=SimpleNamespace())
        force_eager_attention(model)
        assert not hasattr(model.config, "_attn_implementation")


class TestPerLayerWalk:
    def test_walks_plain_fake_model_layers(self):
        """Fakes without modules() are reached via the model.layers chain."""
        layers = [_layer(), _layer(with_attn=False), _layer(with_attn_config=False), _layer()]
        model = SimpleNamespace(config=_config(), model=SimpleNamespace(layers=layers))
        force_eager_attention(model, per_layer=True)
        assert model.config._attn_implementation == "eager"
        assert layers[0].self_attn.config._attn_implementation == "eager"
        assert layers[3].self_attn.config._attn_implementation == "eager"

    def test_walks_modules_for_nested_stacks(self):
        """modules() traversal reaches layers outside model.layers (e.g. nested stacks)."""
        nested_layer = _layer()

        class Model:
            def __init__(self):
                self.config = _config()
                self.stack = SimpleNamespace(layers=[nested_layer])

            def modules(self):
                return [self, self.stack, nested_layer]

        model = Model()
        force_eager_attention(model, per_layer=True)
        assert nested_layer.self_attn.config._attn_implementation == "eager"

    def test_per_layer_runs_even_when_public_setter_succeeds(self):
        """Per-layer config copies may be distinct objects the public API misses."""
        layer = _layer()

        class Model:
            def __init__(self):
                self.config = _config()
                self.model = SimpleNamespace(layers=[layer])

            def set_attn_implementation(self, impl):
                self.config._attn_implementation = impl

        model = Model()
        force_eager_attention(model, per_layer=True)
        assert model.config._attn_implementation == "eager"
        assert layer.self_attn.config._attn_implementation == "eager"

    def test_mock_models_do_not_raise(self):
        force_eager_attention(MagicMock(), per_layer=True)
        # Plain Mock's modules() returns a non-iterable — the walk must absorb it.
        force_eager_attention(Mock(), per_layer=True)


class TestNoOp:
    def test_object_with_neither_api_nor_config_is_silent(self):
        force_eager_attention(SimpleNamespace())
        force_eager_attention(SimpleNamespace(), per_layer=True)
        force_eager_attention(object())
