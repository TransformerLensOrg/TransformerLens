"""Integration tests for OLMo 2 residual-branch hook semantics.

OLMo 2 is post-norm: RMSNorm applies to each sublayer output before the
residual add, so the compatibility aliases hook_attn_out / hook_mlp_out must
expose the norm outputs (the additive contributions), not the raw module
outputs. See issue #1648.
"""

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge

MODEL = "hf-internal-testing/tiny-random-Olmo2ForCausalLM"


@pytest.fixture(scope="module")
def olmo2_bridge() -> TransformerBridge:
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(olmo2_bridge: TransformerBridge) -> torch.Tensor:
    return olmo2_bridge.to_tokens("The capital of France is Paris.")


def test_residual_branch_hooks_decompose_stream(
    olmo2_bridge: TransformerBridge, sample_tokens: torch.Tensor
) -> None:
    with torch.no_grad():
        _, cache = olmo2_bridge.run_with_cache(sample_tokens)

    for layer in range(olmo2_bridge.cfg.n_layers):
        torch.testing.assert_close(
            cache[f"blocks.{layer}.hook_resid_mid"],
            cache[f"blocks.{layer}.hook_resid_pre"] + cache[f"blocks.{layer}.hook_attn_out"],
        )
        torch.testing.assert_close(
            cache[f"blocks.{layer}.hook_resid_post"],
            cache[f"blocks.{layer}.hook_resid_mid"] + cache[f"blocks.{layer}.hook_mlp_out"],
        )


def test_contribution_hooks_differ_from_raw_module_outputs(
    olmo2_bridge: TransformerBridge, sample_tokens: torch.Tensor
) -> None:
    """The architecture-shaped hooks stay raw; the aliases are post-norm."""
    with torch.no_grad():
        _, cache = olmo2_bridge.run_with_cache(sample_tokens)

    for layer in range(olmo2_bridge.cfg.n_layers):
        assert not torch.allclose(
            cache[f"blocks.{layer}.attn.hook_out"],
            cache[f"blocks.{layer}.hook_attn_out"],
        )
        assert not torch.allclose(
            cache[f"blocks.{layer}.mlp.hook_out"],
            cache[f"blocks.{layer}.hook_mlp_out"],
        )


def test_attn_out_ablation_collapses_residual_step(
    olmo2_bridge: TransformerBridge, sample_tokens: torch.Tensor
) -> None:
    """Zeroing hook_attn_out must yield resid_mid == resid_pre — pins that writes
    land on the additive contribution, with no norm applied afterwards."""
    captured = {}

    def grab(key: str):
        def hook_fn(tensor: torch.Tensor, hook) -> torch.Tensor:
            captured[key] = tensor.detach().clone()
            return tensor

        return hook_fn

    with torch.no_grad():
        baseline = olmo2_bridge(sample_tokens)
        ablated = olmo2_bridge.run_with_hooks(
            sample_tokens,
            fwd_hooks=[
                ("blocks.0.hook_attn_out", lambda tensor, hook: torch.zeros_like(tensor)),
                ("blocks.0.hook_resid_pre", grab("resid_pre")),
                ("blocks.0.hook_resid_mid", grab("resid_mid")),
            ],
        )

    assert not torch.equal(ablated, baseline)
    torch.testing.assert_close(captured["resid_mid"], captured["resid_pre"])


def test_attn_out_write_lands_unmodified(
    olmo2_bridge: TransformerBridge, sample_tokens: torch.Tensor
) -> None:
    """Writing v to hook_attn_out must make the contribution exactly v — a
    post-hook norm would distort it (zero-ablation can't catch this: zero is a
    fixed point of RMSNorm)."""
    torch.manual_seed(1)
    replacement = torch.randn(1, sample_tokens.shape[1], olmo2_bridge.cfg.d_model)
    captured = {}

    def grab(key: str):
        def hook_fn(tensor: torch.Tensor, hook) -> torch.Tensor:
            captured[key] = tensor.detach().clone()
            return tensor

        return hook_fn

    with torch.no_grad():
        olmo2_bridge.run_with_hooks(
            sample_tokens,
            fwd_hooks=[
                ("blocks.0.hook_attn_out", lambda tensor, hook: replacement.clone()),
                ("blocks.0.hook_resid_pre", grab("resid_pre")),
                ("blocks.0.hook_resid_mid", grab("resid_mid")),
            ],
        )

    torch.testing.assert_close(captured["resid_mid"] - captured["resid_pre"], replacement)


def test_mlp_out_ablation_collapses_residual_step(
    olmo2_bridge: TransformerBridge, sample_tokens: torch.Tensor
) -> None:
    """Zeroing hook_mlp_out must yield resid_post == resid_mid."""
    captured = {}

    def grab(key: str):
        def hook_fn(tensor: torch.Tensor, hook) -> torch.Tensor:
            captured[key] = tensor.detach().clone()
            return tensor

        return hook_fn

    with torch.no_grad():
        baseline = olmo2_bridge(sample_tokens)
        ablated = olmo2_bridge.run_with_hooks(
            sample_tokens,
            fwd_hooks=[
                ("blocks.0.hook_mlp_out", lambda tensor, hook: torch.zeros_like(tensor)),
                ("blocks.0.hook_resid_mid", grab("resid_mid")),
                ("blocks.0.hook_resid_post", grab("resid_post")),
            ],
        )

    assert not torch.equal(ablated, baseline)
    torch.testing.assert_close(captured["resid_post"], captured["resid_mid"])


def test_hook_mlp_in_exposes_mlp_input(
    olmo2_bridge: TransformerBridge, sample_tokens: torch.Tensor
) -> None:
    """Post-norm OLMo 2 has no pre-MLP norm, so hook_mlp_in must capture the
    mid-residual (the true MLP input), not ln2's input (the raw MLP output)."""
    olmo2_bridge.set_use_hook_mlp_in(True)
    try:
        with torch.no_grad():
            baseline = olmo2_bridge(sample_tokens)
            _, cache = olmo2_bridge.run_with_cache(sample_tokens)
            ablated = olmo2_bridge.run_with_hooks(
                sample_tokens,
                fwd_hooks=[("blocks.0.hook_mlp_in", lambda tensor, hook: torch.zeros_like(tensor))],
            )

        for layer in range(olmo2_bridge.cfg.n_layers):
            torch.testing.assert_close(
                cache[f"blocks.{layer}.hook_mlp_in"],
                cache[f"blocks.{layer}.hook_resid_mid"],
            )
        # Writing lands on the MLP input, so it must change the output.
        assert not torch.equal(ablated, baseline)
    finally:
        olmo2_bridge.set_use_hook_mlp_in(False)


def test_compatibility_mode_preserves_residual_semantics() -> None:
    """The identities must survive enable_compatibility_mode (no LN folding for
    post-norm OLMo 2, but processing must not move the aliases)."""
    bridge = TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)
    tokens = bridge.to_tokens("The capital of France is Paris.")
    bridge.enable_compatibility_mode()

    with torch.no_grad():
        _, cache = bridge.run_with_cache(tokens)

    for layer in range(bridge.cfg.n_layers):
        torch.testing.assert_close(
            cache[f"blocks.{layer}.hook_resid_mid"],
            cache[f"blocks.{layer}.hook_resid_pre"] + cache[f"blocks.{layer}.hook_attn_out"],
        )
        torch.testing.assert_close(
            cache[f"blocks.{layer}.hook_resid_post"],
            cache[f"blocks.{layer}.hook_resid_mid"] + cache[f"blocks.{layer}.hook_mlp_out"],
        )


@pytest.mark.parametrize("use_split_qkv", [True, False], ids=["split_qkv", "attn_in"])
def test_attention_input_forks_are_stateless(use_split_qkv: bool) -> None:
    """OLMo 2 has no pre-attention norm, so input forks must read resid_pre
    directly and must not retain the preceding forward's post-norm output.
    """
    bridge = TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)
    tokens = torch.tensor([[1, 2, 3, 4, 5]])

    with torch.no_grad():
        expected = bridge(tokens).clone()

    if use_split_qkv:
        bridge.set_use_split_qkv_input(True)
        hook_name = "blocks.1.attn.hook_q_input"
    else:
        bridge.set_use_attn_in(True)
        hook_name = "blocks.1.attn.hook_attn_in"

    try:
        with torch.no_grad():
            first_logits, first_cache = bridge.run_with_cache(tokens)
            second_logits, second_cache = bridge.run_with_cache(tokens)
    finally:
        if use_split_qkv:
            bridge.set_use_split_qkv_input(False)
        else:
            bridge.set_use_attn_in(False)

    for cache in (first_cache, second_cache):
        expected_input = cache["blocks.1.hook_resid_pre"].unsqueeze(2)
        torch.testing.assert_close(cache[hook_name], expected_input.expand_as(cache[hook_name]))

    torch.testing.assert_close(first_logits, expected, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(second_logits, expected, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(second_logits, first_logits, atol=0, rtol=0)
