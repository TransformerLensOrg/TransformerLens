"""Conv hooks of the Mamba-family mixers fire on transformers >= 5.15.

transformers 5.15 rewrote these mixers to pass ``self.conv1d.weight`` to a
module-level ``causal_conv1d_fn`` instead of calling ``self.conv1d(...)``, so the
``DepthwiseConv1DBridge`` swapped in for the conv no longer ran. The mixer bridges
now route that call through the conv bridge (``route_causal_conv1d``).

The first group of tests uses a stand-in mixer shaped like transformers >= 5.15,
so it checks the routing on any installed transformers. The second group wires
the real HF mixers: on transformers < 5.15 they call the conv module themselves,
on >= 5.15 the routing does; either way the hooks fire with the same shapes and
the output stays bit-exact with HF's (torch path; no causal-conv1d kernel in CI),
and calls with packed-sequence arguments behave exactly as HF's.
"""

from __future__ import annotations

import copy
import sys
import threading

import pytest
import torch
import torch.nn.functional as F

from tests.unit.model_bridge.supported_architectures.helpers import make_bridge_cfg
from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.component_setup import setup_submodules
from transformer_lens.model_bridge.generalized_components.depthwise_conv1d import (
    DepthwiseConv1DBridge,
    route_causal_conv1d,
)


def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
    """Copy of the torch path of transformers' ``causal_conv1d_fn``, which refuses packed sequences."""
    if kwargs.get("seq_idx") is not None or kwargs.get("cu_seqlens") is not None:
        raise ValueError("can't keep packed sequences apart without the kernel")
    seq_len = hidden_states.shape[-1]
    out = F.conv1d(
        hidden_states.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=bias,
        padding=weight.shape[-1] - 1,
        groups=hidden_states.shape[1],
    )[:, :, :seq_len]
    if activation is not None:
        out = F.silu(out)
    return out.to(hidden_states.dtype)


class _KernelStyleMixer(torch.nn.Module):
    """Calls the module-level conv function with its conv's weight, as transformers >= 5.15."""

    def __init__(self, channels: int = 6, kernel: int = 4) -> None:
        super().__init__()
        self.conv1d = torch.nn.Conv1d(
            channels, channels, kernel, groups=channels, padding=kernel - 1
        )
        self.other = torch.nn.Conv1d(
            channels, channels, kernel, groups=channels, padding=kernel - 1
        )

    def forward(self, x: torch.Tensor, use_other: bool = False, **kwargs) -> torch.Tensor:
        conv = self.other if use_other else self.conv1d
        return causal_conv1d_fn(x, conv.weight.squeeze(1), conv.bias, activation="silu", **kwargs)


def _bridged_kernel_style_mixer():
    torch.manual_seed(0)
    mixer = _KernelStyleMixer()
    reference = copy.deepcopy(mixer)
    conv = DepthwiseConv1DBridge(name="conv1d")
    conv.set_original_component(mixer.conv1d)
    mixer.conv1d = conv
    return mixer, reference, conv


def _record(store: dict, name: str):
    def hook(tensor, hook):
        store[name] = tensor.detach().clone()
        return tensor

    return hook


def test_route_runs_the_conv_bridge_and_matches_the_function() -> None:
    mixer, reference, conv = _bridged_kernel_style_mixer()
    seen: dict = {}
    conv.hook_in.add_hook(_record(seen, "in"))
    conv.hook_out.add_hook(_record(seen, "out"))
    x = torch.randn(2, 6, 5)
    with torch.no_grad(), route_causal_conv1d(mixer, "conv1d"):
        out = mixer(x)
    torch.testing.assert_close(out, reference(x), rtol=0.0, atol=0.0)
    assert seen["in"].shape == (2, 6, 5)
    assert seen["out"].shape == (2, 6, 5 + 4 - 1)


def test_route_patch_on_conv_hook_out_reaches_the_output() -> None:
    mixer, reference, conv = _bridged_kernel_style_mixer()
    conv.hook_out.add_hook(lambda tensor, hook: torch.zeros_like(tensor))
    x = torch.randn(2, 6, 5)
    with torch.no_grad(), route_causal_conv1d(mixer, "conv1d"):
        out = mixer(x)
    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0.0, atol=0.0)
    assert not torch.equal(out, reference(x))


def test_route_leaves_other_convs_and_the_module_function_alone() -> None:
    mixer, reference, conv = _bridged_kernel_style_mixer()
    seen: dict = {}
    conv.hook_in.add_hook(_record(seen, "in"))
    x = torch.randn(2, 6, 5)
    function_before = sys.modules[__name__].causal_conv1d_fn
    with torch.no_grad(), route_causal_conv1d(mixer, "conv1d"):
        out = mixer(x, use_other=True)
    torch.testing.assert_close(out, reference(x, use_other=True), rtol=0.0, atol=0.0)
    assert seen == {}
    assert sys.modules[__name__].causal_conv1d_fn is function_before


def test_route_restores_the_module_function_after_an_error() -> None:
    mixer, _, conv = _bridged_kernel_style_mixer()

    def fail(tensor, hook):
        raise RuntimeError("hook failed")

    conv.hook_in.add_hook(fail)
    function_before = sys.modules[__name__].causal_conv1d_fn
    with pytest.raises(RuntimeError, match="hook failed"):
        with torch.no_grad(), route_causal_conv1d(mixer, "conv1d"):
            mixer(torch.randn(2, 6, 5))
    assert sys.modules[__name__].causal_conv1d_fn is function_before


def test_route_is_a_no_op_without_a_conv_bridge() -> None:
    torch.manual_seed(0)
    mixer = _KernelStyleMixer()
    x = torch.randn(2, 6, 5)
    function_before = sys.modules[__name__].causal_conv1d_fn
    with torch.no_grad():
        expected = mixer(x)
        with route_causal_conv1d(mixer, "conv1d"):
            assert sys.modules[__name__].causal_conv1d_fn is function_before
            torch.testing.assert_close(mixer(x), expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("name", ["seq_idx", "cu_seqlens"])
def test_route_leaves_packed_sequences_to_the_function(name) -> None:
    mixer, _, conv = _bridged_kernel_style_mixer()
    seen: dict = {}
    conv.hook_in.add_hook(_record(seen, "in"))
    packed = {name: torch.zeros(2, 5, dtype=torch.int32)}
    with torch.no_grad(), route_causal_conv1d(mixer, "conv1d"):
        with pytest.raises(ValueError, match="packed sequences"):
            mixer(torch.randn(2, 6, 5), **packed)
    assert seen == {}


def test_route_keeps_concurrent_forwards_apart() -> None:
    """Thread B's route outlives thread A's: B's conv hook still fires and the function is put back."""
    mixer_a, reference, conv_a = _bridged_kernel_style_mixer()
    mixer_b, _, conv_b = _bridged_kernel_style_mixer()
    seen: dict = {}
    conv_a.hook_in.add_hook(_record(seen, "a"))
    conv_b.hook_in.add_hook(_record(seen, "b"))
    x = torch.randn(2, 6, 5)
    function_before = sys.modules[__name__].causal_conv1d_fn
    a_entered, b_entered, a_left = threading.Event(), threading.Event(), threading.Event()
    outputs: dict = {}
    errors: list = []

    def run_a() -> None:
        try:
            with torch.no_grad(), route_causal_conv1d(mixer_a, "conv1d"):
                a_entered.set()
                assert b_entered.wait(10)
                outputs["a"] = mixer_a(x)
        except BaseException as exc:  # surfaced in the main thread below
            errors.append(exc)
        finally:
            a_left.set()

    def run_b() -> None:
        try:
            assert a_entered.wait(10)
            with torch.no_grad(), route_causal_conv1d(mixer_b, "conv1d"):
                b_entered.set()
                assert a_left.wait(10)
                outputs["b"] = mixer_b(x)
        except BaseException as exc:  # surfaced in the main thread below
            errors.append(exc)
        finally:
            b_entered.set()

    threads = [threading.Thread(target=run_a), threading.Thread(target=run_b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert not errors, errors
    assert set(seen) == {"a", "b"}
    with torch.no_grad():
        expected = reference(x)
    torch.testing.assert_close(outputs["a"], expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(outputs["b"], expected, rtol=0.0, atol=0.0)
    assert sys.modules[__name__].causal_conv1d_fn is function_before


# --- Real HF mixers, wired as the adapters wire them --------------------------------


def _wire(arch: str, key: str, hf_module):
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(
        make_bridge_cfg(arch, d_model=32, n_heads=4, d_head=8)
    )
    bridge = copy.deepcopy(adapter.component_mapping["blocks"].submodules[key])
    bridge.set_original_component(hf_module)
    setup_submodules(bridge, adapter, hf_module)
    return bridge


def _mamba():
    from transformers import MambaConfig
    from transformers.models.mamba.modeling_mamba import MambaMixer

    cfg = MambaConfig(
        hidden_size=32, state_size=8, expand=2, conv_kernel=4, num_hidden_layers=2, vocab_size=64
    )
    return MambaMixer(cfg, layer_idx=0)


def _mamba2():
    from transformers import Mamba2Config
    from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer

    cfg = Mamba2Config(
        hidden_size=32,
        num_heads=8,
        head_dim=8,
        expand=2,
        n_groups=1,
        state_size=8,
        conv_kernel=4,
        chunk_size=8,
        num_hidden_layers=2,
        vocab_size=64,
    )
    return Mamba2Mixer(cfg, layer_idx=0)


def _nemotron_h():
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHMamba2Mixer

    cfg = NemotronHConfig(
        hidden_size=32,
        mamba_num_heads=4,
        mamba_head_dim=8,
        ssm_state_size=8,
        conv_kernel=4,
        n_groups=1,
        num_hidden_layers=2,
        vocab_size=64,
        intermediate_size=64,
        expand=2,
        hybrid_override_pattern="MM",
    )
    return NemotronHMamba2Mixer(cfg, layer_idx=0)


def _lfm2():
    from transformers import Lfm2Config
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ShortConv

    cfg = Lfm2Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        conv_L_cache=3,
        layer_types=["conv", "full_attention"],
    )
    return Lfm2ShortConv(cfg, layer_idx=0)


HF_MIXERS = [
    pytest.param("MambaForCausalLM", "mixer", "conv1d", _mamba, id="mamba"),
    pytest.param("Mamba2ForCausalLM", "mixer", "conv1d", _mamba2, id="mamba2"),
    pytest.param("NemotronHForCausalLM", "mixer", "conv1d", _nemotron_h, id="nemotron_h"),
    pytest.param("Lfm2ForCausalLM", "conv", "conv", _lfm2, id="lfm2"),
]


def _first(output):
    return output[0] if isinstance(output, tuple) else output


@pytest.mark.parametrize("arch,key,conv_attr,build", HF_MIXERS)
def test_hf_mixer_conv_hooks_fire_and_output_is_bit_exact(arch, key, conv_attr, build) -> None:
    torch.manual_seed(0)
    hf_module = build().eval()
    reference = copy.deepcopy(hf_module)
    bridge = _wire(arch, key, hf_module)
    conv = getattr(bridge, conv_attr)
    module = sys.modules[type(hf_module).__module__]
    function_before = getattr(module, "causal_conv1d_fn", None)
    seen: dict = {}
    conv.hook_in.add_hook(_record(seen, "in"))
    conv.hook_out.add_hook(_record(seen, "out"))

    x = torch.randn(2, 7, 32)
    with torch.no_grad():
        out = _first(bridge(x))
        expected = _first(reference(x))

    torch.testing.assert_close(out, expected, rtol=0.0, atol=0.0)
    channels = conv.original_component.in_channels
    kernel = conv.original_component.kernel_size[0]
    assert seen["in"].shape == (2, channels, 7)
    assert seen["out"].shape == (2, channels, 7 + kernel - 1)
    assert getattr(module, "causal_conv1d_fn", None) is function_before


@pytest.mark.parametrize("arch,key,conv_attr,build", HF_MIXERS)
def test_hf_mixer_patch_on_conv_hook_out_reaches_the_output(arch, key, conv_attr, build) -> None:
    torch.manual_seed(0)
    hf_module = build().eval()
    reference = copy.deepcopy(hf_module)
    bridge = _wire(arch, key, hf_module)
    getattr(bridge, conv_attr).hook_out.add_hook(lambda tensor, hook: torch.zeros_like(tensor))

    x = torch.randn(2, 7, 32)
    with torch.no_grad():
        assert not torch.allclose(_first(bridge(x)), _first(reference(x)))


def _outcome(call):
    """The call's first output, or the type and message of what it raised."""
    try:
        return _first(call())
    except Exception as exc:
        return type(exc), str(exc)


PACKED_SEQUENCES = {
    "seq_idx": torch.zeros(2, 7, dtype=torch.int32),
    "cu_seqlens": torch.tensor([0, 7, 14], dtype=torch.int32),
}


@pytest.mark.parametrize("name", list(PACKED_SEQUENCES))
@pytest.mark.parametrize("arch,key,conv_attr,build", HF_MIXERS)
def test_hf_mixer_packed_sequences_behave_as_hf(arch, key, conv_attr, build, name) -> None:
    torch.manual_seed(0)
    hf_module = build().eval()
    reference = copy.deepcopy(hf_module)
    bridge = _wire(arch, key, hf_module)
    x = torch.randn(2, 7, 32)
    packed = {name: PACKED_SEQUENCES[name]}
    with torch.no_grad():
        expected = _outcome(lambda: reference(x, **packed))
        got = _outcome(lambda: bridge(x, **packed))
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)
    else:
        assert got == expected
