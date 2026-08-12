"""Integration tests for Granite residual-branch hook semantics.

Granite's HF blocks compute ``residual + sublayer_out * residual_multiplier``,
so hook_attn_out / hook_mlp_out must expose the scaled contribution (issue
#1648). The hub tiny-random Granite ships residual_multiplier=1.0, which cannot
catch a missing scale, so the fixtures build local checkpoints with 0.22 (the
granite-3.3 value). Norm-based comparisons throughout — the wrong tensor is
collinear with the right one, so cosine checks are blind here.
"""

import pytest
import torch

from transformer_lens.model_bridge import TransformerBridge

RESIDUAL_MULTIPLIER = 0.22
TOKENIZER_SOURCE = "hf-internal-testing/tiny-random-GraniteForCausalLM"


@pytest.fixture(scope="module")
def granite_path(tmp_path_factory):
    from transformers import AutoTokenizer, GraniteConfig, GraniteForCausalLM

    tok = AutoTokenizer.from_pretrained(TOKENIZER_SOURCE)
    torch.manual_seed(0)
    cfg = GraniteConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=len(tok),
        max_position_embeddings=512,
        pad_token_id=tok.pad_token_id or 0,
        residual_multiplier=RESIDUAL_MULTIPLIER,
    )
    path = tmp_path_factory.mktemp("granite") / "tiny-granite"
    model = GraniteForCausalLM(cfg).to(torch.float32)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def granite_bridge(granite_path) -> TransformerBridge:
    return TransformerBridge.boot_transformers(granite_path, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="module")
def sample_tokens(granite_bridge: TransformerBridge) -> torch.Tensor:
    return granite_bridge.to_tokens("The capital of France is Paris.")


def test_forward_matches_fresh_hf(granite_bridge, granite_path, sample_tokens) -> None:
    from transformers import AutoModelForCausalLM

    fresh = AutoModelForCausalLM.from_pretrained(
        granite_path, dtype=torch.float32, attn_implementation="eager"
    )
    fresh.eval()
    with torch.no_grad():
        bridge_out = granite_bridge(sample_tokens)
        hf_out = fresh(input_ids=sample_tokens).logits
    max_diff = (bridge_out - hf_out).abs().max().item()
    assert max_diff < 1e-5, f"Bridge vs fresh HF max diff = {max_diff}"


def test_residual_branch_hooks_decompose_stream(granite_bridge, sample_tokens) -> None:
    with torch.no_grad():
        _, cache = granite_bridge.run_with_cache(sample_tokens)

    for layer in range(granite_bridge.cfg.n_layers):
        torch.testing.assert_close(
            cache[f"blocks.{layer}.hook_resid_mid"],
            cache[f"blocks.{layer}.hook_resid_pre"] + cache[f"blocks.{layer}.hook_attn_out"],
        )
        torch.testing.assert_close(
            cache[f"blocks.{layer}.hook_resid_post"],
            cache[f"blocks.{layer}.hook_resid_mid"] + cache[f"blocks.{layer}.hook_mlp_out"],
        )


def test_contribution_is_scaled_raw_output(granite_bridge, sample_tokens) -> None:
    """The contribution must be raw * residual_multiplier; raw stays on the
    architecture-shaped hooks."""
    with torch.no_grad():
        _, cache = granite_bridge.run_with_cache(sample_tokens)

    for layer in range(granite_bridge.cfg.n_layers):
        torch.testing.assert_close(
            cache[f"blocks.{layer}.hook_attn_out"],
            cache[f"blocks.{layer}.attn.hook_out"] * RESIDUAL_MULTIPLIER,
        )
        torch.testing.assert_close(
            cache[f"blocks.{layer}.hook_mlp_out"],
            cache[f"blocks.{layer}.mlp.hook_out"] * RESIDUAL_MULTIPLIER,
        )


def test_read_only_hooks_leave_forward_bit_exact(granite_bridge, sample_tokens) -> None:
    """Cache-style hooks must not perturb the forward — the rewrite (with its
    divide-multiply rounding) only happens when a hook changes the tensor."""
    grabbed = {}

    def grab(tensor: torch.Tensor, hook) -> torch.Tensor:
        grabbed[hook.name] = tensor.detach().clone()
        return tensor

    with torch.no_grad():
        baseline = granite_bridge(sample_tokens)
        hooked = granite_bridge.run_with_hooks(
            sample_tokens,
            fwd_hooks=[
                ("blocks.0.hook_attn_out", grab),
                ("blocks.0.hook_mlp_out", grab),
            ],
        )

    assert torch.equal(baseline, hooked)
    assert "blocks.0.hook_attn_out" in grabbed


def test_attn_out_ablation_collapses_residual_step(granite_bridge, sample_tokens) -> None:
    captured = {}

    def grab(key: str):
        def hook_fn(tensor: torch.Tensor, hook) -> torch.Tensor:
            captured[key] = tensor.detach().clone()
            return tensor

        return hook_fn

    with torch.no_grad():
        baseline = granite_bridge(sample_tokens)
        ablated = granite_bridge.run_with_hooks(
            sample_tokens,
            fwd_hooks=[
                ("blocks.0.hook_attn_out", lambda tensor, hook: torch.zeros_like(tensor)),
                ("blocks.0.hook_resid_pre", grab("resid_pre")),
                ("blocks.0.hook_resid_mid", grab("resid_mid")),
            ],
        )

    assert not torch.equal(ablated, baseline)
    torch.testing.assert_close(captured["resid_mid"], captured["resid_pre"])


def test_attn_out_write_lands_unmodified(granite_bridge, sample_tokens) -> None:
    """Writing v must make the contribution v itself — the multiplier may not be
    applied on top of the write (the issue's Granite failure mode)."""
    torch.manual_seed(1)
    replacement = torch.randn(1, sample_tokens.shape[1], granite_bridge.cfg.d_model)
    captured = {}

    def grab(key: str):
        def hook_fn(tensor: torch.Tensor, hook) -> torch.Tensor:
            captured[key] = tensor.detach().clone()
            return tensor

        return hook_fn

    with torch.no_grad():
        granite_bridge.run_with_hooks(
            sample_tokens,
            fwd_hooks=[
                ("blocks.0.hook_attn_out", lambda tensor, hook: replacement.clone()),
                ("blocks.0.hook_resid_pre", grab("resid_pre")),
                ("blocks.0.hook_resid_mid", grab("resid_mid")),
            ],
        )

    torch.testing.assert_close(captured["resid_mid"] - captured["resid_pre"], replacement)


def test_in_place_mutation_is_not_dropped(granite_bridge, sample_tokens) -> None:
    """A hook that zeroes the tensor in place (returning the same object) must
    ablate the contribution — identity checks would silently drop it."""
    captured = {}

    def grab(key: str):
        def hook_fn(tensor: torch.Tensor, hook) -> torch.Tensor:
            captured[key] = tensor.detach().clone()
            return tensor

        return hook_fn

    def zero_in_place(tensor: torch.Tensor, hook) -> torch.Tensor:
        tensor.zero_()
        return tensor

    with torch.no_grad():
        granite_bridge.run_with_hooks(
            sample_tokens,
            fwd_hooks=[
                ("blocks.0.hook_attn_out", zero_in_place),
                ("blocks.0.hook_resid_pre", grab("resid_pre")),
                ("blocks.0.hook_resid_mid", grab("resid_mid")),
            ],
        )

    torch.testing.assert_close(captured["resid_mid"], captured["resid_pre"])


def test_backward_hook_receives_gradient(granite_bridge, sample_tokens) -> None:
    """Backward hooks must sit on the compute path (the rewrite must happen when
    bwd hooks are attached, or they observe a dead branch)."""
    received = []

    def bwd_hook(grad, hook):
        received.append(grad.detach().clone())
        return grad

    granite_bridge.add_hook("blocks.0.hook_attn_out", bwd_hook, dir="bwd")
    try:
        logits = granite_bridge(sample_tokens)
        logits.sum().backward()
    finally:
        granite_bridge.reset_hooks()

    assert received and received[0].abs().sum() > 0


class TestGraniteMoeHookSemantics:
    """Same contract for GraniteMoe: the MoE output is scaled before the add."""

    @pytest.fixture(scope="class")
    def moe_bridge(self, tmp_path_factory) -> TransformerBridge:
        from transformers import AutoTokenizer, GraniteMoeConfig, GraniteMoeForCausalLM

        tok = AutoTokenizer.from_pretrained(TOKENIZER_SOURCE)
        torch.manual_seed(0)
        cfg = GraniteMoeConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=len(tok),
            max_position_embeddings=512,
            pad_token_id=tok.pad_token_id or 0,
            residual_multiplier=RESIDUAL_MULTIPLIER,
            num_local_experts=4,
            num_experts_per_tok=2,
        )
        path = tmp_path_factory.mktemp("granite_moe") / "tiny-granite-moe"
        model = GraniteMoeForCausalLM(cfg).to(torch.float32)
        model.save_pretrained(path)
        tok.save_pretrained(path)
        return TransformerBridge.boot_transformers(str(path), device="cpu", dtype=torch.float32)

    def test_residual_branch_hooks_decompose_stream(self, moe_bridge) -> None:
        tokens = moe_bridge.to_tokens("The capital of France is Paris.")
        with torch.no_grad():
            _, cache = moe_bridge.run_with_cache(tokens)

        for layer in range(moe_bridge.cfg.n_layers):
            torch.testing.assert_close(
                cache[f"blocks.{layer}.hook_resid_mid"],
                cache[f"blocks.{layer}.hook_resid_pre"] + cache[f"blocks.{layer}.hook_attn_out"],
            )
            torch.testing.assert_close(
                cache[f"blocks.{layer}.hook_resid_post"],
                cache[f"blocks.{layer}.hook_resid_mid"] + cache[f"blocks.{layer}.hook_mlp_out"],
            )
