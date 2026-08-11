"""Integration tests for BLOOM residual-branch hook semantics."""

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge

MODEL = "trl-internal-testing/tiny-BloomForCausalLM"


@pytest.fixture(scope="module")
def bloom_bridge() -> TransformerBridge:
    return TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)


def test_residual_branch_hooks_decompose_stream(bloom_bridge: TransformerBridge) -> None:
    tokens = bloom_bridge.to_tokens("The capital of France is Paris.")

    with torch.no_grad():
        _, cache = bloom_bridge.run_with_cache(tokens)

    for layer in range(bloom_bridge.cfg.n_layers):
        torch.testing.assert_close(
            cache[f"blocks.{layer}.hook_resid_mid"],
            cache[f"blocks.{layer}.hook_resid_pre"] + cache[f"blocks.{layer}.hook_attn_out"],
        )
        torch.testing.assert_close(
            cache[f"blocks.{layer}.hook_resid_post"],
            cache[f"blocks.{layer}.hook_resid_mid"] + cache[f"blocks.{layer}.hook_mlp_out"],
        )


def test_attn_out_ablation_collapses_residual_step(bloom_bridge: TransformerBridge) -> None:
    """Zeroing hook_attn_out must yield resid_mid == resid_pre — pins that writes
    land on the additive contribution, not the residual-added module output."""
    tokens = bloom_bridge.to_tokens("The capital of France is Paris.")
    captured = {}

    def grab(key: str):
        def hook_fn(tensor: torch.Tensor, hook) -> torch.Tensor:
            captured[key] = tensor.detach().clone()
            return tensor

        return hook_fn

    with torch.no_grad():
        baseline = bloom_bridge(tokens)
        ablated = bloom_bridge.run_with_hooks(
            tokens,
            fwd_hooks=[
                ("blocks.0.hook_attn_out", lambda tensor, hook: torch.zeros_like(tensor)),
                ("blocks.0.hook_resid_pre", grab("resid_pre")),
                ("blocks.0.hook_resid_mid", grab("resid_mid")),
            ],
        )

    assert not torch.equal(ablated, baseline)
    torch.testing.assert_close(captured["resid_mid"], captured["resid_pre"])


def test_mlp_out_ablation_collapses_residual_step(bloom_bridge: TransformerBridge) -> None:
    """Zeroing hook_mlp_out must yield resid_post == resid_mid."""
    tokens = bloom_bridge.to_tokens("The capital of France is Paris.")
    captured = {}

    def grab(key: str):
        def hook_fn(tensor: torch.Tensor, hook) -> torch.Tensor:
            captured[key] = tensor.detach().clone()
            return tensor

        return hook_fn

    with torch.no_grad():
        baseline = bloom_bridge(tokens)
        ablated = bloom_bridge.run_with_hooks(
            tokens,
            fwd_hooks=[
                ("blocks.0.hook_mlp_out", lambda tensor, hook: torch.zeros_like(tensor)),
                ("blocks.0.hook_resid_mid", grab("resid_mid")),
                ("blocks.0.hook_resid_post", grab("resid_post")),
            ],
        )

    assert not torch.equal(ablated, baseline)
    torch.testing.assert_close(captured["resid_post"], captured["resid_mid"])


def test_slow_but_exact_checkpoint_still_fires_mlp_out() -> None:
    """Checkpoints shipping pretraining_tp>1 + slow_but_exact=True (e.g. this
    bigscience testing model) make HF's BloomMLP bypass the dense_4h_to_h module
    call, so hook_mlp_out would silently vanish from the cache. The bridge must
    force the module-call path its hooks attach to."""
    bridge = TransformerBridge.boot_transformers(
        "bigscience/bigscience-small-testing", device="cpu", dtype=torch.float32
    )
    tokens = bridge.to_tokens("The capital of France is Paris.")

    with torch.no_grad():
        _, cache = bridge.run_with_cache(tokens)

    for layer in range(bridge.cfg.n_layers):
        torch.testing.assert_close(
            cache[f"blocks.{layer}.hook_resid_post"],
            cache[f"blocks.{layer}.hook_resid_mid"] + cache[f"blocks.{layer}.hook_mlp_out"],
        )


def test_compatibility_mode_preserves_residual_semantics() -> None:
    """Compat-mode weight processing must not crash on BLOOM and must keep the
    residual identities and the model function (log_softmax; raw logits shift
    under center_unembed). Regression: the tiny fixture's stray
    num_key_value_heads sent fold_value_biases down a GQA branch."""
    bridge = TransformerBridge.boot_transformers(MODEL, device="cpu", dtype=torch.float32)
    tokens = bridge.to_tokens("The capital of France is Paris.")

    with torch.no_grad():
        base_log_probs = torch.log_softmax(bridge(tokens), dim=-1)

    bridge.enable_compatibility_mode()

    with torch.no_grad():
        logits, cache = bridge.run_with_cache(tokens)

    torch.testing.assert_close(
        torch.log_softmax(logits, dim=-1), base_log_probs, atol=1e-4, rtol=1e-5
    )
    for layer in range(bridge.cfg.n_layers):
        torch.testing.assert_close(
            cache[f"blocks.{layer}.hook_resid_mid"],
            cache[f"blocks.{layer}.hook_resid_pre"] + cache[f"blocks.{layer}.hook_attn_out"],
        )
        torch.testing.assert_close(
            cache[f"blocks.{layer}.hook_resid_post"],
            cache[f"blocks.{layer}.hook_resid_mid"] + cache[f"blocks.{layer}.hook_mlp_out"],
        )
