"""Hook-based tests for dynamic J-space coordinate patching."""

import warnings
from typing import Any

import pytest
import torch

from tests.unit.tools.conftest import D_MODEL, D_VOCAB, _lens, _ToyBridge
from transformer_lens.tools.analysis import JacobianLens

# The toy vocab dictionary has only D_VOCAB atoms, fewer than the library DEFAULT_K, so every
# real solve in this file passes an explicit k within [1, D_VOCAB].
SOLVE_K = 8


def _active_source_and_distinct_target(
    lens: JacobianLens, model: _ToyBridge, prompt: str, layer: int, position: int
) -> tuple[int, int]:
    """Discover a real active source at (layer, position) instead of guessing a token id --
    coordinate_patch_hooks requires the source to be active, and a toy model's real activations
    are not hand-computable in advance."""
    decomposition = lens.decompose(model, prompt, layer=layer, position=position, k=SOLVE_K)
    source_id = int(decomposition.support[0])
    target_id = (source_id + 1) % D_VOCAB
    return source_id, target_id


def test_coordinate_patch_hooks_shape_mirrors_swap_hooks(toy_model: _ToyBridge) -> None:
    with pytest.warns(UserWarning):
        hooks = _lens().coordinate_patch_hooks(toy_model, 3, 5, layers=[0], positions=[0])
    assert [name for name, _ in hooks] == ["blocks.0.hook_out"]
    assert callable(hooks[0][1])


def test_coordinate_patch_hooks_warns_once_naming_layer_and_position_counts(
    toy_model: _ToyBridge,
) -> None:
    lens = JacobianLens(
        {0: torch.eye(D_MODEL), 1: torch.eye(D_MODEL)}, n_prompts=1, d_model=D_MODEL
    )
    with pytest.warns(UserWarning, match=r"2 layer\(s\) x 2 position\(s\)") as record:
        lens.coordinate_patch_hooks(toy_model, 3, 5, layers=[0, 1], positions=[0, 1])
    assert len(record) == 1


def test_coordinate_patch_hooks_rejects_unfitted_layer(toy_model: _ToyBridge) -> None:
    with pytest.raises(ValueError, match="source layers"):
        _lens().coordinate_patch_hooks(toy_model, 3, 5, layers=[2], positions=[0])


def test_coordinate_patch_hooks_requires_nonempty_positions(toy_model: _ToyBridge) -> None:
    with pytest.raises(ValueError, match="positions"):
        _lens().coordinate_patch_hooks(toy_model, 3, 5, layers=[0], positions=[])


def test_coordinate_patch_hooks_rejects_identical_tokens(toy_model: _ToyBridge) -> None:
    with pytest.raises(ValueError, match="same token|identical|distinct"):
        _lens().coordinate_patch_hooks(toy_model, 3, 3, layers=[0], positions=[0])


def test_coordinate_patch_hooks_changes_only_requested_positions(toy_model: _ToyBridge) -> None:
    lens = _lens()
    prompt = "a toy prompt"
    tokens = toy_model.to_tokens(prompt)
    _, baseline = toy_model.run_with_cache(tokens)
    source_id, target_id = _active_source_and_distinct_target(lens, toy_model, prompt, 0, -1)

    with pytest.warns(UserWarning):
        hooks = lens.coordinate_patch_hooks(
            toy_model, source_id, target_id, layers=[0], positions=[-1], k=SOLVE_K
        )
    with toy_model.hooks(fwd_hooks=hooks):
        _, patched = toy_model.run_with_cache(tokens)

    delta = patched["blocks.0.hook_out"] - baseline["blocks.0.hook_out"]
    torch.testing.assert_close(delta[:, :-1], torch.zeros_like(delta[:, :-1]))


def test_coordinate_patch_hooks_oracle_parity_with_offline_coordinate_patch(
    toy_model: _ToyBridge,
) -> None:
    lens = _lens()
    prompt = "a toy prompt"
    tokens = toy_model.to_tokens(prompt)
    _, baseline = toy_model.run_with_cache(tokens)
    pre_hook_activation = baseline["blocks.0.hook_out"][0, -1, :].float()
    source_id, target_id = _active_source_and_distinct_target(lens, toy_model, prompt, 0, -1)

    with pytest.warns(UserWarning):
        hooks = lens.coordinate_patch_hooks(
            toy_model, source_id, target_id, layers=[0], positions=[-1], k=SOLVE_K
        )
    with toy_model.hooks(fwd_hooks=hooks):
        _, patched = toy_model.run_with_cache(tokens)

    expected = lens.coordinate_patch(
        toy_model,
        pre_hook_activation,
        layer=0,
        source_token=source_id,
        target_token=target_id,
        k=SOLVE_K,
    )
    torch.testing.assert_close(patched["blocks.0.hook_out"][0, -1, :].float(), expected.patched)


def test_decomposition_cache_hit_skips_resolve_across_hook_firings(
    toy_model: _ToyBridge, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transformer_lens.tools.analysis.jacobian_lens_coordinate_patch as core_module

    lens = _lens()
    prompt = "a toy prompt"
    tokens = toy_model.to_tokens(prompt)
    source_id, target_id = _active_source_and_distinct_target(lens, toy_model, prompt, 0, -1)
    cache: dict = {}
    calls = []
    original = core_module.get_sparse_decomposition

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(core_module, "get_sparse_decomposition", spy)
    with pytest.warns(UserWarning):
        hooks = lens.coordinate_patch_hooks(
            toy_model,
            source_id,
            target_id,
            layers=[0],
            positions=[-1],
            decomposition_cache=cache,
            k=SOLVE_K,
        )
    with toy_model.hooks(fwd_hooks=hooks):
        toy_model(tokens)
    first_call_count = len(calls)
    assert first_call_count >= 1

    with toy_model.hooks(fwd_hooks=hooks):
        toy_model(tokens)
    assert len(calls) == first_call_count  # second forward pass is entirely cache hits


def test_coordinate_patch_hooks_propagates_core_errors_uncaught(
    toy_model: _ToyBridge, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fail-fast design decision: an inactive-source error from the core loop must abort the
    whole forward pass, not be caught and turned into a partial/silent patch."""
    import transformer_lens.tools.analysis.jacobian_lens as jacobian_lens_module

    def raise_inactive_source(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("source_idx=3 is not in the decomposition's active support")

    monkeypatch.setattr(
        jacobian_lens_module, "solve_coordinate_patch_positions", raise_inactive_source
    )
    with pytest.warns(UserWarning):
        hooks = _lens().coordinate_patch_hooks(toy_model, 3, 5, layers=[0], positions=[0])
    with pytest.raises(ValueError, match="active support"):
        with toy_model.hooks(fwd_hooks=hooks):
            toy_model(toy_model.to_tokens("a toy prompt"))


def test_coordinate_patch_hooks_warnings_propagate_uncaught(
    toy_model: _ToyBridge, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Warnings raised by the core loop (conditioning, near-parallel) must reach the caller through
    the hook -- not be swallowed or re-wrapped."""
    import transformer_lens.tools.analysis.jacobian_lens as jacobian_lens_module

    def fake_solve(
        activations: torch.Tensor, dictionary, position_labels, source_idx, target_idx, **kwargs
    ):
        warnings.warn(
            "coordinate-patch source and target atoms are near-parallel (stub)", UserWarning
        )
        return activations.clone(), {}

    monkeypatch.setattr(jacobian_lens_module, "solve_coordinate_patch_positions", fake_solve)
    with pytest.warns(UserWarning, match="coordinate_patch_hooks"):
        hooks = _lens().coordinate_patch_hooks(toy_model, 3, 5, layers=[0], positions=[0])
    with pytest.warns(UserWarning, match="near-parallel"):
        with toy_model.hooks(fwd_hooks=hooks):
            toy_model(toy_model.to_tokens("a toy prompt"))
