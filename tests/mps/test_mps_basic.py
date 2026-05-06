"""Apple Silicon MPS smoke tests for TransformerLens.

Design principles:
- All tests skip automatically on non-MPS runners (Linux, Windows, CPU-only Macs)
- Only float32 is used (bfloat16 is unsupported on MPS)
- Only small models are loaded (roneneldan/TinyStories-1M, ~50MB)
- torch.mps.empty_cache() + gc.collect() between tests to stay within memory budget
- TRANSFORMERLENS_ALLOW_MPS=1 must be set for get_device() to return "mps"

CI: These tests are run via the `mps-checks` job in .github/workflows/checks.yml
which sets TRANSFORMERLENS_ALLOW_MPS=1 and runs on macos-latest.
"""

import gc
import os
import warnings

import pytest
import torch

# Skip the entire module on non-MPS runners (Linux CI, CPU-only Macs)
pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available on this runner — skipping Apple Silicon tests",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SMALL_MODEL = "roneneldan/TinyStories-1M"  # ~50MB, safe for 1GB runner budget


def _load_tiny_model(device: str = "mps"):
    """Load TinyStories-1M on the given device with float32 (bfloat16 unsupported on MPS)."""
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained(SMALL_MODEL, device=device, dtype=torch.float32)


def _cleanup(model=None):
    """Free GPU memory between tests."""
    if model is not None:
        del model
    torch.mps.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# 1. Device detection (no model load — instant)
# ---------------------------------------------------------------------------


def test_mps_device_available():
    """Sanity check: MPS backend is present and built on this runner."""
    assert torch.backends.mps.is_available(), "MPS not available"
    assert torch.backends.mps.is_built(), "MPS not built into this PyTorch"


def test_mps_get_device_returns_mps_with_env_var():
    """get_device() auto-selects MPS when TRANSFORMERLENS_ALLOW_MPS=1 is set."""
    from transformer_lens.utilities.devices import get_device

    original = os.environ.get("TRANSFORMERLENS_ALLOW_MPS", "")
    try:
        os.environ["TRANSFORMERLENS_ALLOW_MPS"] = "1"
        device = get_device()
        assert isinstance(device, str)
        assert device == "mps", f"Expected 'mps', got '{device}'"
    finally:
        if original:
            os.environ["TRANSFORMERLENS_ALLOW_MPS"] = original
        else:
            os.environ.pop("TRANSFORMERLENS_ALLOW_MPS", None)


def test_mps_get_device_falls_back_to_cpu_without_env_var():
    """get_device() falls back to CPU when TRANSFORMERLENS_ALLOW_MPS is unset (safety default)."""
    from transformer_lens.utilities.devices import get_device

    original = os.environ.get("TRANSFORMERLENS_ALLOW_MPS", "")
    try:
        os.environ.pop("TRANSFORMERLENS_ALLOW_MPS", None)
        device = get_device()
        # On a Mac with no CUDA, should return cpu (safe default)
        assert isinstance(device, str)
        assert (
            device == "cpu"
        ), f"Without TRANSFORMERLENS_ALLOW_MPS=1, get_device() should return 'cpu' not '{device}'"
    finally:
        if original:
            os.environ["TRANSFORMERLENS_ALLOW_MPS"] = original


def test_mps_warn_if_mps_emits_warning_without_env_var():
    """warn_if_mps() emits a UserWarning when MPS is used without the env var."""
    import transformer_lens.utilities.devices as devices_module
    from transformer_lens.utilities import warn_if_mps

    original = os.environ.get("TRANSFORMERLENS_ALLOW_MPS", "")
    original_warned = devices_module._mps_warned
    try:
        os.environ.pop("TRANSFORMERLENS_ALLOW_MPS", None)
        devices_module._mps_warned = False  # reset so warning fires
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_if_mps("mps")
        assert any(
            "MPS backend" in str(warning.message) for warning in w
        ), "Expected MPS warning but got: " + str([str(x.message) for x in w])
    finally:
        if original:
            os.environ["TRANSFORMERLENS_ALLOW_MPS"] = original
        devices_module._mps_warned = original_warned


# ---------------------------------------------------------------------------
# 2. Raw tensor operations on Metal (no model load)
# ---------------------------------------------------------------------------


def test_mps_tensor_basic_operations():
    """Basic tensor arithmetic runs on the Metal GPU without errors."""
    x = torch.randn(16, 32, device="mps", dtype=torch.float32)
    y = torch.randn(16, 32, device="mps", dtype=torch.float32)

    z = x + y
    assert z.device.type == "mps"

    w = torch.matmul(x, y.T)
    assert w.device.type == "mps"
    assert w.shape == (16, 16)

    # Verify result comes back to CPU correctly
    z_cpu = z.cpu()
    assert z_cpu.device.type == "cpu"

    _cleanup()


def test_mps_softmax_and_layernorm():
    """Softmax and LayerNorm — core transformer ops — work on MPS."""
    x = torch.randn(4, 16, 64, device="mps", dtype=torch.float32)

    softmax_out = torch.nn.functional.softmax(x, dim=-1)
    assert softmax_out.device.type == "mps"
    assert torch.allclose(softmax_out.sum(dim=-1), torch.ones(4, 16, device="mps"), atol=1e-5)

    ln = torch.nn.LayerNorm(64).to("mps")
    ln_out = ln(x)
    assert ln_out.device.type == "mps"

    _cleanup()


# ---------------------------------------------------------------------------
# 3. Model loading and forward pass on Metal
# ---------------------------------------------------------------------------


def test_mps_model_forward_pass():
    """TinyStories-1M loads and runs a forward pass on the Metal GPU."""
    model = _load_tiny_model(device="mps")

    tokens = model.to_tokens("Once upon a time")
    assert tokens.device.type == "mps", f"Tokens should be on MPS, got {tokens.device}"

    logits = model(tokens)
    assert logits.device.type == "mps", f"Logits should be on MPS, got {logits.device}"
    assert logits.shape[-1] == model.cfg.d_vocab
    assert not torch.isnan(logits).any(), "NaN values in logits — possible MPS compute error"

    _cleanup(model)


def test_mps_run_with_cache():
    """run_with_cache() returns cache tensors on the Metal GPU."""
    model = _load_tiny_model(device="mps")
    tokens = model.to_tokens("The quick brown fox")

    logits, cache = model.run_with_cache(tokens)

    assert logits.device.type == "mps"

    # Check a representative set of cache keys
    hook_q = cache["blocks.0.attn.hook_q"]
    assert hook_q.device.type == "mps", f"Cache tensor not on MPS: {hook_q.device}"
    assert not torch.isnan(hook_q).any(), "NaN in attention query cache"

    _cleanup(model)


def test_mps_activation_hook_fires_on_metal():
    """run_with_hooks() fires hooks and hook tensors are on the Metal GPU."""
    model = _load_tiny_model(device="mps")
    tokens = model.to_tokens("Apple Silicon rocks")

    hook_devices = []
    hook_shapes = []

    def capture_hook(value, hook):
        hook_devices.append(value.device.type)
        hook_shapes.append(value.shape)
        return value

    model.run_with_hooks(
        tokens,
        fwd_hooks=[
            ("blocks.0.attn.hook_q", capture_hook),
            ("blocks.0.mlp.hook_post", capture_hook),
        ],
    )

    assert len(hook_devices) == 2, f"Expected 2 hooks to fire, got {len(hook_devices)}"
    for device in hook_devices:
        assert device == "mps", f"Hook tensor not on MPS: {device}"

    _cleanup(model)


def test_mps_float32_inference():
    """Explicit float32 model loads and infers correctly on MPS."""
    model = _load_tiny_model(device="mps")

    # Verify all parameters are float32
    for name, param in model.named_parameters():
        assert param.dtype == torch.float32, f"Parameter {name} has wrong dtype: {param.dtype}"

    tokens = model.to_tokens("Testing float32 on Metal")
    logits = model(tokens)
    assert logits.dtype == torch.float32

    _cleanup(model)


def test_mps_loss_computation():
    """Loss computation (return_type='loss') works on MPS."""
    model = _load_tiny_model(device="mps")

    loss = model("Once upon a time in a land", return_type="loss")
    assert isinstance(loss, torch.Tensor)
    assert loss.device.type == "mps"
    assert not torch.isnan(loss), f"NaN loss — possible MPS compute error: {loss}"
    assert loss.item() > 0, "Loss should be positive"

    _cleanup(model)
