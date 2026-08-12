"""OLMo 2/3 post-norm hook placement in HookedTransformer's TransformerBlock.

The residual identities are weight-independent, so a random-weight model with
original_architecture="Olmo2ForCausalLM" pins the hook ordering (ln1/ln2 must
apply before hook_attn_out / hook_mlp_out). See issue #1648.
"""

import pytest
import torch

from transformer_lens import HookedTransformer, HookedTransformerConfig

D_VOCAB = 100


@pytest.fixture(scope="module", params=["Olmo2ForCausalLM", "Olmo3ForCausalLM"])
def olmo_model(request) -> HookedTransformer:
    torch.manual_seed(0)
    cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=64,
        n_ctx=32,
        d_head=16,
        n_heads=4,
        d_mlp=128,
        d_vocab=D_VOCAB,
        act_fn="silu",
        gated_mlp=True,
        normalization_type="RMS",
        positional_embedding_type="rotary",
        rotary_dim=16,
        original_architecture=request.param,
    )
    model = HookedTransformer(cfg)
    model.init_weights()
    return model


@pytest.fixture(scope="module")
def sample_tokens() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, D_VOCAB, (1, 10))


def test_residual_branch_hooks_decompose_stream(
    olmo_model: HookedTransformer, sample_tokens: torch.Tensor
) -> None:
    with torch.no_grad():
        _, cache = olmo_model.run_with_cache(sample_tokens)

    for layer in range(olmo_model.cfg.n_layers):
        torch.testing.assert_close(
            cache[f"blocks.{layer}.hook_resid_mid"],
            cache[f"blocks.{layer}.hook_resid_pre"] + cache[f"blocks.{layer}.hook_attn_out"],
        )
        torch.testing.assert_close(
            cache[f"blocks.{layer}.hook_resid_post"],
            cache[f"blocks.{layer}.hook_resid_mid"] + cache[f"blocks.{layer}.hook_mlp_out"],
        )


def test_attn_out_ablation_collapses_residual_step(
    olmo_model: HookedTransformer, sample_tokens: torch.Tensor
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
        baseline = olmo_model(sample_tokens)
        ablated = olmo_model.run_with_hooks(
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
    olmo_model: HookedTransformer, sample_tokens: torch.Tensor
) -> None:
    """Writing v to hook_attn_out must make the contribution exactly v — a
    post-hook norm would distort it (zero-ablation can't catch this: zero is a
    fixed point of RMSNorm)."""
    torch.manual_seed(1)
    replacement = torch.randn(1, sample_tokens.shape[1], olmo_model.cfg.d_model)
    captured = {}

    def grab(key: str):
        def hook_fn(tensor: torch.Tensor, hook) -> torch.Tensor:
            captured[key] = tensor.detach().clone()
            return tensor

        return hook_fn

    with torch.no_grad():
        olmo_model.run_with_hooks(
            sample_tokens,
            fwd_hooks=[
                ("blocks.0.hook_attn_out", lambda tensor, hook: replacement.clone()),
                ("blocks.0.hook_resid_pre", grab("resid_pre")),
                ("blocks.0.hook_resid_mid", grab("resid_mid")),
            ],
        )

    torch.testing.assert_close(captured["resid_mid"] - captured["resid_pre"], replacement)


def test_hook_mlp_in_exposes_mid_residual(
    olmo_model: HookedTransformer, sample_tokens: torch.Tensor
) -> None:
    """With use_hook_mlp_in, hook_mlp_in must equal resid_mid — post-norm OLMo
    has no pre-MLP norm, so the MLP input is the mid-residual itself."""
    olmo_model.cfg.use_hook_mlp_in = True
    try:
        with torch.no_grad():
            _, cache = olmo_model.run_with_cache(sample_tokens)
        for layer in range(olmo_model.cfg.n_layers):
            torch.testing.assert_close(
                cache[f"blocks.{layer}.hook_mlp_in"],
                cache[f"blocks.{layer}.hook_resid_mid"],
            )
    finally:
        olmo_model.cfg.use_hook_mlp_in = False


def test_mlp_out_ablation_collapses_residual_step(
    olmo_model: HookedTransformer, sample_tokens: torch.Tensor
) -> None:
    """Zeroing hook_mlp_out must yield resid_post == resid_mid."""
    captured = {}

    def grab(key: str):
        def hook_fn(tensor: torch.Tensor, hook) -> torch.Tensor:
            captured[key] = tensor.detach().clone()
            return tensor

        return hook_fn

    with torch.no_grad():
        baseline = olmo_model(sample_tokens)
        ablated = olmo_model.run_with_hooks(
            sample_tokens,
            fwd_hooks=[
                ("blocks.0.hook_mlp_out", lambda tensor, hook: torch.zeros_like(tensor)),
                ("blocks.0.hook_resid_mid", grab("resid_mid")),
                ("blocks.0.hook_resid_post", grab("resid_post")),
            ],
        )

    assert not torch.equal(ablated, baseline)
    torch.testing.assert_close(captured["resid_post"], captured["resid_mid"])
