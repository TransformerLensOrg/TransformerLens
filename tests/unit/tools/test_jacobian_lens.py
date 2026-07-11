"""Unit tests for the Jacobian lens: estimator correctness on a closed-form toy
model, artifact round-trips, merge weighting, and the raw-weights guards.

The toy model's blocks are position-independent linear residual updates
``h <- h + W h``, so the exact Jacobian from block ``l``'s output to the final
block's output is the matrix product ``(I + W_{n-1}) ... (I + W_{l+1})`` — the
fit result can be checked against a closed form, matching the reference
implementation's own orientation test.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from transformer_lens import HookedRootModule
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis import JacobianLens

D_MODEL = 6
N_LAYERS = 4
D_VOCAB = 11
SEQ_LEN = 9
SKIP_FIRST = 2


class _ToyBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.linear.weight, std=0.2)
        self.hook_resid_post = HookPoint()

    def forward(self, resid):
        return self.hook_resid_post(resid + self.linear(resid))


class _ToyResidModel(HookedRootModule):
    """Minimal HookedRootModule exposing the surface JacobianLens needs."""

    def __init__(self, normalization_type: str = "LN"):
        super().__init__()
        torch.manual_seed(0)
        self.cfg = SimpleNamespace(
            n_layers=N_LAYERS,
            d_model=D_MODEL,
            d_vocab=D_VOCAB,
            normalization_type=normalization_type,
            output_logits_soft_cap=None,
            model_name="toy",
        )
        self.embed = nn.Embedding(D_VOCAB, D_MODEL)
        self.blocks = nn.ModuleList([_ToyBlock(D_MODEL) for _ in range(N_LAYERS)])
        self.ln_final = nn.Identity()
        self.unembed = nn.Linear(D_MODEL, D_VOCAB, bias=False)
        self.setup()

    @property
    def W_U(self):
        return self.unembed.weight.T  # [d_model, d_vocab]

    def to_tokens(self, prompt: str):
        ids = [(3 * i + len(prompt)) % D_VOCAB for i in range(SEQ_LEN)]
        return torch.tensor([ids], dtype=torch.long)

    def to_single_token(self, string: str) -> int:
        return len(string) % D_VOCAB

    def forward(self, tokens, return_type="logits"):
        resid = self.embed(tokens)
        for block in self.blocks:
            resid = block(resid)
        if return_type is None:
            return None
        return self.unembed(self.ln_final(resid))


def _closed_form_jacobian(model: _ToyResidModel, layer: int) -> torch.Tensor:
    """Exact d h_final / d h_layer for the linear toy: prod of (I + W_k)."""
    jacobian = torch.eye(D_MODEL)
    for k in range(layer + 1, N_LAYERS):
        jacobian = (torch.eye(D_MODEL) + model.blocks[k].linear.weight) @ jacobian
    return jacobian


@pytest.fixture(scope="module")
def toy_model():
    return _ToyResidModel()


@pytest.fixture(scope="module")
def fitted_lens(toy_model):
    return JacobianLens.fit(
        toy_model,
        ["a toy prompt", "another toy prompt"],
        dim_batch=4,  # not a divisor of d_model=6: exercises the partial final pass
        skip_first_positions=SKIP_FIRST,
        show_progress=False,
    )


def test_fit_recovers_closed_form_jacobians(toy_model, fitted_lens):
    assert fitted_lens.source_layers == [0, 1, 2]
    assert fitted_lens.n_prompts == 2
    for layer in fitted_lens.source_layers:
        expected = _closed_form_jacobian(toy_model, layer)
        torch.testing.assert_close(fitted_lens.jacobians[layer], expected, atol=1e-5, rtol=1e-4)


def test_transport_orientation(toy_model, fitted_lens):
    resid = torch.randn(3, D_MODEL)
    expected = resid @ _closed_form_jacobian(toy_model, 2).T
    torch.testing.assert_close(fitted_lens.transport(resid, 2), expected, atol=1e-4, rtol=1e-3)


def test_readout_final_layer_matches_model_output(toy_model, fitted_lens):
    result = fitted_lens.readout(toy_model, "a toy prompt")
    torch.testing.assert_close(result.lens_logits[N_LAYERS - 1], result.model_logits)


def test_transported_readout_approximates_model_output(toy_model, fitted_lens):
    # The toy is exactly linear, so transporting from ANY layer must reproduce
    # the model's own logits.
    result = fitted_lens.readout(toy_model, "a toy prompt", layers=[0, 2])
    for layer in (0, 2):
        torch.testing.assert_close(
            result.lens_logits[layer], result.model_logits, atol=1e-3, rtol=1e-3
        )


def test_save_load_roundtrip(tmp_path, fitted_lens):
    path = str(tmp_path / "lens.pt")
    fitted_lens.save(path)
    loaded = JacobianLens.load(path)
    assert loaded.source_layers == fitted_lens.source_layers
    assert loaded.n_prompts == fitted_lens.n_prompts
    assert loaded.d_model == fitted_lens.d_model
    assert loaded.metadata["model_name"] == "toy"
    for layer in loaded.source_layers:
        # fp16 storage round-trip tolerance, matching the reference tests.
        torch.testing.assert_close(
            loaded.jacobians[layer], fitted_lens.jacobians[layer], atol=2e-3, rtol=2e-3
        )
        assert loaded.jacobians[layer].dtype == torch.float32


def test_load_official_format_without_metadata(tmp_path):
    path = str(tmp_path / "official.pt")
    torch.save(
        {
            "J": {0: torch.eye(D_MODEL, dtype=torch.float16)},
            "n_prompts": 7,
            "source_layers": [0],
            "d_model": D_MODEL,
        },
        path,
    )
    lens = JacobianLens.load(path)
    assert lens.n_prompts == 7
    assert lens.metadata == {}
    assert lens.jacobians[0].dtype == torch.float32


def test_load_rejects_fit_checkpoints(tmp_path):
    path = str(tmp_path / "ckpt.pt")
    torch.save({"jacobian_sum": {}, "n_done": 3}, path)
    with pytest.raises(ValueError, match="no 'J' key"):
        JacobianLens.load(path)


def test_merge_is_n_prompts_weighted():
    lens_a = JacobianLens({0: torch.ones(D_MODEL, D_MODEL)}, n_prompts=2, d_model=D_MODEL)
    lens_b = JacobianLens({0: 4 * torch.ones(D_MODEL, D_MODEL)}, n_prompts=6, d_model=D_MODEL)
    merged = JacobianLens.merge([lens_a, lens_b])
    assert merged.n_prompts == 8
    torch.testing.assert_close(merged.jacobians[0], torch.full((D_MODEL, D_MODEL), 3.25))


def test_merge_rejects_mismatched_lenses():
    lens_a = JacobianLens({0: torch.ones(D_MODEL, D_MODEL)}, n_prompts=1, d_model=D_MODEL)
    lens_b = JacobianLens({1: torch.ones(D_MODEL, D_MODEL)}, n_prompts=1, d_model=D_MODEL)
    with pytest.raises(ValueError, match="disagree"):
        JacobianLens.merge([lens_a, lens_b])
    with pytest.raises(ValueError, match="at least one"):
        JacobianLens.merge([])


def test_validate_model_rejects_wrong_d_model(toy_model):
    lens = JacobianLens({0: torch.ones(3, 3)}, n_prompts=1, d_model=3)
    with pytest.raises(ValueError, match="d_model"):
        lens.validate_model(toy_model)


def test_validate_model_rejects_out_of_range_layers(toy_model):
    lens = JacobianLens({N_LAYERS + 5: torch.ones(D_MODEL, D_MODEL)}, n_prompts=1, d_model=D_MODEL)
    with pytest.raises(ValueError, match="different model"):
        lens.validate_model(toy_model)


def test_folded_layernorm_is_rejected(fitted_lens):
    processed = _ToyResidModel(normalization_type="LNPre")
    with pytest.raises(ValueError, match="fold_ln"):
        fitted_lens.validate_model(processed)
    with pytest.raises(ValueError, match="fold_ln"):
        JacobianLens.fit(processed, ["a toy prompt"], show_progress=False)


def _mock_bridge(compatibility_mode=False):
    bridge = MagicMock(spec=TransformerBridge)
    bridge.cfg = SimpleNamespace(d_model=D_MODEL, n_layers=N_LAYERS, normalization_type="LN")
    bridge.compatibility_mode = compatibility_mode
    return bridge


def test_compatibility_mode_bridge_is_rejected():
    lens = JacobianLens({0: torch.ones(D_MODEL, D_MODEL)}, n_prompts=1, d_model=D_MODEL)
    with pytest.raises(ValueError, match="compatibility mode"):
        lens.validate_model(_mock_bridge(compatibility_mode=True))


def test_raw_bridge_passes_validation():
    lens = JacobianLens({0: torch.ones(D_MODEL, D_MODEL)}, n_prompts=1, d_model=D_MODEL)
    assert lens.validate_model(_mock_bridge()) is lens


def test_fit_skips_short_prompts(toy_model):
    with pytest.raises(ValueError, match="no prompt was long enough"):
        with pytest.warns(UserWarning, match="skipping prompt"):
            JacobianLens.fit(
                toy_model,
                ["a toy prompt"],
                dim_batch=6,
                skip_first_positions=SEQ_LEN,  # every prompt too short -> skipped
                show_progress=False,
            )


def test_intervention_hooks_have_valid_names_and_run(toy_model, fitted_lens):
    swap_hooks = fitted_lens.swap_hooks(toy_model, 3, 5, layers=[1, 2])
    steer_hooks = fitted_lens.steering_hooks(toy_model, 3, layers=[1], alpha=2.0)
    ablate_hooks = fitted_lens.ablation_hooks(toy_model, [3, 5], layers=[2])
    assert [name for name, _ in swap_hooks] == [
        "blocks.1.hook_resid_post",
        "blocks.2.hook_resid_post",
    ]
    tokens = toy_model.to_tokens("a toy prompt")
    baseline = toy_model(tokens)
    for hooks in (swap_hooks, steer_hooks, ablate_hooks):
        with toy_model.hooks(fwd_hooks=hooks):
            intervened = toy_model(tokens)
        assert intervened.shape == baseline.shape
        assert torch.isfinite(intervened).all()
        assert not torch.allclose(intervened, baseline)


def test_swap_leaves_orthogonal_complement_unchanged(toy_model, fitted_lens):
    layer = 2
    vectors = fitted_lens.lens_vectors(toy_model, [3, 5], layer)  # [2, d_model]
    hooks = fitted_lens.swap_hooks(toy_model, 3, 5, layers=[layer])
    tokens = toy_model.to_tokens("a toy prompt")
    _, base_cache = toy_model.run_with_cache(tokens)
    with toy_model.hooks(fwd_hooks=hooks):
        _, swap_cache = toy_model.run_with_cache(tokens)
    name = f"blocks.{layer}.hook_resid_post"
    delta = (swap_cache[name] - base_cache[name]).reshape(-1, D_MODEL)
    # The intervention only moves activations within span{v_s, v_t}.
    basis, _ = torch.linalg.qr(vectors.T)  # orthonormal basis of the span
    orthogonal_part = delta - (delta @ basis) @ basis.T
    assert orthogonal_part.abs().max().item() < 1e-4


def test_exports():
    from transformer_lens.tools import analysis

    assert analysis.JacobianLens is JacobianLens
    assert hasattr(analysis, "JacobianLensReadout")
