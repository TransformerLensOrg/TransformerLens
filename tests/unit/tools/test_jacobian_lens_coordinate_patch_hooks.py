"""Hook-based tests for dynamic J-space coordinate patching."""

import warnings
from typing import Any

import pytest
import torch

from tests.unit.tools.conftest import D_MODEL, D_VOCAB, SEQ_LEN, _lens, _ToyBridge
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


def test_coordinate_patch_hooks_plural_install_binds_each_layer_to_its_own_dictionary(
    toy_model: _ToyBridge,
) -> None:
    """First end-to-end test of a *plural* install: two fitted layers, one real forward pass.

    It pins two things no single-layer test above reaches:

    * The per-closure default-argument binding ``layer=layer, dictionary=dictionary`` in
      ``coordinate_patch_hooks``. Without it, Python's late-binding closures make every hook
      capture the *last* loop iteration's ``layer``/``dictionary`` -- so layer 0's hook would
      solve against layer 1's dictionary and key its cache under layer 1. The two layers are
      given deliberately different (non-scalar) dictionaries so a mis-bound hook produces a
      numerically wrong edit; coordinate patching is scale-covariant, so a uniform rescale would
      leave the edit unchanged and hide the mis-binding.
    * The band precondition documented in the ``Raises`` note: the source must stay in the active
      support at *both* layers after the earlier hook has already edited the residual. The
      fixture searches for a source that satisfies it rather than assuming one does.
    """
    lens = JacobianLens(
        {0: torch.eye(D_MODEL), 1: torch.diag(torch.linspace(0.5, 2.0, D_MODEL))},
        n_prompts=1,
        d_model=D_MODEL,
    )
    prompt = "a toy prompt"
    tokens = toy_model.to_tokens(prompt)
    position = -1
    normalized_position = SEQ_LEN - 1  # -1 over the toy model's fixed sequence length

    _, baseline = toy_model.run_with_cache(tokens)
    clean_layer0 = baseline["blocks.0.hook_out"][0, position].float()

    # Find one source that is active at layer 0 *and* still active at layer 1 after layer 0's edit
    # -- exactly the band precondition. Candidates come from layer 0's clean active support.
    layer0_support = [
        int(token)
        for token in lens.decompose(
            toy_model, tokens, layer=0, position=position, k=SOLVE_K
        ).support
    ]
    source_id = -1
    target_id = -1
    layer1_input: torch.Tensor | None = None
    for candidate in layer0_support:
        candidate_target = (candidate + 1) % D_VOCAB
        if candidate_target == candidate:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            layer0_only = lens.coordinate_patch_hooks(
                toy_model, candidate, candidate_target, layers=[0], positions=[position], k=SOLVE_K
            )
            with toy_model.hooks(fwd_hooks=layer0_only):
                _, edited = toy_model.run_with_cache(tokens)
        after_edit = edited["blocks.1.hook_out"][0, position].float()
        support = [
            int(token)
            for token in lens.decompose(toy_model, after_edit, layer=1, k=SOLVE_K).support
        ]
        if candidate in support:
            source_id, target_id, layer1_input = candidate, candidate_target, after_edit
            break
    assert (
        layer1_input is not None
    ), "no source stays active at both layers; fixture needs a new prompt"

    # Install on BOTH fitted layers and run one real forward pass through model.hooks(...).
    cache: dict = {}
    with pytest.warns(UserWarning, match=r"2 layer\(s\) x 1 position\(s\)"):
        hooks = lens.coordinate_patch_hooks(
            toy_model,
            source_id,
            target_id,
            layers=[0, 1],
            positions=[position],
            decomposition_cache=cache,
            k=SOLVE_K,
        )
    with toy_model.hooks(fwd_hooks=hooks):
        _, patched = toy_model.run_with_cache(tokens)

    # Layer 0's hook must solve against layer 0's OWN dictionary. A mis-bound closure would use
    # layer 1's dictionary here and produce a different edit of the clean layer-0 activation.
    expected_layer0 = lens.coordinate_patch(
        toy_model, clean_layer0, layer=0, source_token=source_id, target_token=target_id, k=SOLVE_K
    )
    torch.testing.assert_close(
        patched["blocks.0.hook_out"][0, position].float(), expected_layer0.patched
    )

    # Layer 1's hook must solve against layer 1's own dictionary, on the residual as edited by
    # layer 0 upstream (``layer1_input`` was captured with only layer 0's hook installed).
    expected_layer1 = lens.coordinate_patch(
        toy_model, layer1_input, layer=1, source_token=source_id, target_token=target_id, k=SOLVE_K
    )
    torch.testing.assert_close(
        patched["blocks.1.hook_out"][0, position].float(), expected_layer1.patched
    )

    # Each layer keyed its own cache slot; a late-bound ``layer`` would collapse both onto layer 1.
    assert (0, 0, normalized_position) in cache
    assert (1, 0, normalized_position) in cache
