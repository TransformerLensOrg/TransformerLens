"""Unembedding compatibility must preserve the checkpoint's trainable parameters."""

import copy
from collections import ChainMap
from pathlib import Path

import pytest
import torch
from accelerate import cpu_offload, disk_offload
from accelerate.hooks import remove_hook_from_module

from transformer_lens.config.transformer_lens_config import TransformerLensConfig
from transformer_lens.model_bridge.generalized_components.unembedding import (
    UnembeddingBridge,
)
from transformer_lens.weight_processing import ProcessWeights


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_unembedding_preserves_optimizer_updates(bias: bool, dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    reference = torch.nn.Linear(5, 3, bias=bias, dtype=dtype)
    layer = copy.deepcopy(reference)
    original_parameters = {id(parameter) for parameter in layer.parameters()}
    bridge = UnembeddingBridge(name="lm_head")
    bridge.set_original_component(layer)
    bridge.requires_grad_(True)
    assert {id(parameter) for parameter in bridge.parameters()} == original_parameters

    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    bridge_optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-3)
    inputs = torch.randn(4, 5, dtype=dtype)
    labels = torch.tensor([0, 2, 1, 2])
    for _ in range(3):
        reference_inputs = inputs.clone().requires_grad_()
        bridge_inputs = inputs.clone().requires_grad_()
        reference_logits = reference(reference_inputs)
        bridge_logits = bridge(bridge_inputs)
        torch.testing.assert_close(bridge_logits, reference_logits)
        torch.nn.functional.cross_entropy(reference_logits, labels).backward()
        torch.nn.functional.cross_entropy(bridge_logits, labels).backward()
        torch.testing.assert_close(bridge_inputs.grad, reference_inputs.grad)
        torch.testing.assert_close(layer.weight.grad, reference.weight.grad)
        if bias:
            torch.testing.assert_close(layer.bias.grad, reference.bias.grad)
        reference_optimizer.step()
        bridge_optimizer.step()
        reference_optimizer.zero_grad()
        bridge_optimizer.zero_grad()
        torch.testing.assert_close(bridge(inputs), reference(inputs))
    if not bias:
        assert not bridge.b_U.requires_grad
        torch.testing.assert_close(bridge.b_U, torch.zeros(3, dtype=dtype))


def test_processed_unembedding_bias_survives_checkpoint_and_dtype_changes() -> None:
    layer = torch.nn.Linear(5, 3, bias=False)
    bridge = UnembeddingBridge(name="lm_head")
    bridge.set_original_component(layer)
    processed_bias = torch.tensor([0.2, -0.4, 0.6], requires_grad=True)
    bridge.set_processed_weights({"bias": processed_bias})
    bridge.double()
    inputs = torch.randn(2, 5, dtype=torch.float64)
    expected = torch.nn.functional.linear(inputs, layer.weight, processed_bias.double())
    torch.testing.assert_close(bridge(inputs), expected)
    assert not bridge.b_U.requires_grad
    assert "_original_component.bias" in bridge.state_dict()

    restored = UnembeddingBridge(name="lm_head")
    restored.set_original_component(torch.nn.Linear(5, 3, bias=False).double())
    restored.load_state_dict(bridge.state_dict())
    torch.testing.assert_close(restored(inputs), expected)
    assert {name for name, _ in restored.named_parameters()} == {"_original_component.weight"}


def test_processed_unembedding_bias_rejects_wrong_shape() -> None:
    bridge = UnembeddingBridge(name="lm_head")
    bridge.set_original_component(torch.nn.Linear(5, 3, bias=False))
    with pytest.raises(ValueError, match="Shape mismatch"):
        bridge.set_processed_weights({"bias": torch.zeros(4)})


def test_folded_layer_norm_bias_updates_unembedding_buffer() -> None:
    torch.manual_seed(42)
    layer = torch.nn.Linear(4, 3, bias=False, dtype=torch.float64)
    norm = torch.nn.LayerNorm(4, dtype=torch.float64)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([0.5, 1.5, 2.0, 0.8]))
        norm.bias.copy_(torch.tensor([0.1, -0.2, 0.3, 0.4]))
    bridge = UnembeddingBridge(name="lm_head")
    bridge.set_original_component(layer)
    inputs = torch.randn(2, 4, dtype=torch.float64)
    expected = bridge(norm(inputs)).detach()
    state = {
        "ln_final.w": norm.weight.detach(),
        "ln_final.b": norm.bias.detach(),
        "unembed.W_U": bridge.W_U.detach(),
        "unembed.b_U": bridge.b_U.detach(),
    }
    config = TransformerLensConfig(
        d_model=4, d_head=4, n_layers=0, n_ctx=2, d_vocab=3, device="cpu", dtype=torch.float64
    )
    folded = ProcessWeights.fold_layer_norm(state, config, center_weights=False)
    bridge.set_processed_weights({"weight": folded["unembed.W_U"].T, "bias": folded["unembed.b_U"]})
    normalized = torch.nn.functional.layer_norm(inputs, (4,), eps=norm.eps)
    torch.testing.assert_close(bridge(normalized), expected, rtol=1e-12, atol=1e-12)
    assert bridge.b_U.abs().max() > 0
    assert not bridge.b_U.requires_grad


@pytest.mark.parametrize("offload_buffers", [False, True])
@pytest.mark.parametrize("root_module", [False, True])
@pytest.mark.parametrize("offload_method", ["cpu", "disk"])
def test_unembedding_bias_with_offloaded_weights(
    offload_buffers: bool, root_module: bool, offload_method: str, tmp_path: Path
) -> None:
    torch.manual_seed(42)
    reference = torch.nn.Linear(5, 3, bias=False)
    layer = copy.deepcopy(reference)
    offloaded_module = layer if root_module else torch.nn.Sequential(layer)

    def apply_offload(directory: Path) -> None:
        if offload_method == "cpu":
            cpu_offload(
                offloaded_module,
                execution_device=torch.device("cpu"),
                offload_buffers=offload_buffers,
            )
        else:
            disk_offload(
                offloaded_module,
                offload_dir=str(directory),
                execution_device=torch.device("cpu"),
                offload_buffers=offload_buffers,
            )

    apply_offload(tmp_path / "zero")
    bridge = UnembeddingBridge(name="lm_head")
    bridge.set_original_component(layer)
    inputs = torch.randn(2, 5)
    for _ in range(2):
        torch.testing.assert_close(bridge(inputs), reference(inputs))
    remove_hook_from_module(offloaded_module, recurse=True)
    torch.testing.assert_close(bridge(inputs), reference(inputs))
    assert bridge.b_U.device == layer.weight.device == reference.weight.device

    apply_offload(tmp_path / "processed")
    processed_bias = torch.tensor([0.2, -0.4, 0.6], requires_grad=True)
    bridge.set_processed_weights({"bias": processed_bias})
    for _ in range(2):
        torch.testing.assert_close(bridge(inputs), reference(inputs) + processed_bias)
    remove_hook_from_module(offloaded_module, recurse=True)
    torch.testing.assert_close(bridge(inputs), reference(inputs) + processed_bias)
    assert bridge.b_U.device == layer.weight.device == reference.weight.device
    assert not bridge.b_U.requires_grad


@pytest.mark.parametrize("offload_method", ["cpu", "disk"])
def test_post_wrap_bias_fold_updates_offload_map(offload_method: str, tmp_path: Path) -> None:
    """A final_logits_bias fold after wrapping must re-sync accelerate's offload map."""
    from types import SimpleNamespace

    from transformer_lens.model_bridge.supported_architectures.bart import (
        BartArchitectureAdapter,
    )

    torch.manual_seed(42)
    reference = torch.nn.Linear(5, 3, bias=False)
    layer = copy.deepcopy(reference)
    # offload_buffers=True so each forward reads the bias back from the map.
    if offload_method == "cpu":
        cpu_offload(layer, execution_device=torch.device("cpu"), offload_buffers=True)
    else:
        disk_offload(
            layer,
            offload_dir=str(tmp_path),
            execution_device=torch.device("cpu"),
            offload_buffers=True,
        )
    bridge = UnembeddingBridge(name="lm_head")
    bridge.set_original_component(layer)

    # On a CPU execution device the map entry aliases the bias storage, which would
    # hide a stale snapshot; clone it the way a device-to-cpu offload copy would.
    hook = UnembeddingBridge._offload_hook(layer)
    assert hook is not None and hook.weights_map is not None
    hook.weights_map = ChainMap(
        {"bias": torch.as_tensor(hook.weights_map["bias"]).clone()}, hook.weights_map
    )

    folded = torch.tensor([0.2, -0.4, 0.6])
    fake_bridge = SimpleNamespace(
        original_model=SimpleNamespace(
            final_logits_bias=folded.reshape(1, -1).clone(), lm_head=layer
        )
    )
    adapter = object.__new__(BartArchitectureAdapter)
    adapter.setup_hook_compatibility(fake_bridge)

    torch.testing.assert_close(torch.as_tensor(hook.weights_map["bias"]), folded)
    inputs = torch.randn(2, 5)
    for _ in range(2):
        torch.testing.assert_close(bridge(inputs), reference(inputs) + folded)
