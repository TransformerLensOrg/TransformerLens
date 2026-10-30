"""Tests for direct path patching on tiny randomly initialised in-memory bridges."""

import contextlib

import pytest
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GraniteConfig,
    GraniteForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    Olmo2Config,
    Olmo2ForCausalLM,
)

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)
from transformer_lens.patching import (
    get_act_patch_direct_path,
    get_act_patch_direct_path_all_sources,
)
from transformer_lens.tools.analysis.attribution_patching import (
    Node,
    _ablate_edges,
    _edge_hook_flags,
    capture_writer_outputs,
)

# float64 models keep implementation rounding far below this; results are stored in float32.
_EXACT_ATOL = 1e-6


def _tiny_gpt2(fold_ln: bool = True) -> TransformerBridge:
    torch.manual_seed(0)
    # A wider init than GPT-2's 0.02 puts path effects near 1e-3, far above the tolerance.
    hf_config = GPT2Config(
        n_layer=3, n_head=4, n_embd=64, vocab_size=100, n_positions=16, initializer_range=0.1
    )
    hf_model = GPT2LMHeadModel(hf_config).to(torch.float64).eval()
    model = build_bridge_from_module(
        hf_model, "GPT2LMHeadModel", hf_config=hf_config, tokenizer=None, device="cpu"
    )
    if fold_ln:
        model.enable_compatibility_mode()
    else:
        generator = torch.Generator().manual_seed(1)
        with torch.no_grad():
            for block in model.blocks:
                ln1 = block.ln1
                ln1.weight.copy_(1 + 0.5 * torch.randn(ln1.weight.shape, generator=generator))
                ln1.bias.copy_(0.2 * torch.randn(ln1.bias.shape, generator=generator))
    return model.eval()


def _tiny_llama_gqa(n_kv_heads=2, fold_ln=False) -> TransformerBridge:
    torch.manual_seed(0)
    hf_config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=n_kv_heads,
        vocab_size=100,
        max_position_embeddings=16,
        initializer_range=0.1,
    )
    hf_config._attn_implementation = "eager"
    hf_model = LlamaForCausalLM(hf_config).to(torch.float64).eval()
    with torch.no_grad():
        for block in hf_model.model.layers:
            block.input_layernorm.weight.copy_(torch.linspace(0.5, 1.5, 64))
    model = build_bridge_from_module(
        hf_model,
        "LlamaForCausalLM",
        hf_config=hf_config,
        tokenizer=None,
        device="cpu",
        dtype=torch.float64,
    )
    if fold_ln:
        model.enable_compatibility_mode()
    return model.eval()


def _tiny_decoder(config_cls, model_cls, architecture, **overrides) -> TransformerBridge:
    torch.manual_seed(0)
    hf_config = config_cls(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        vocab_size=100,
        max_position_embeddings=16,
        **overrides,
    )
    hf_model = model_cls(hf_config).eval()
    model = build_bridge_from_module(
        hf_model, architecture, hf_config=hf_config, tokenizer=None, device="cpu"
    )
    return model.eval()


def _tiny_granite(residual_multiplier: float) -> TransformerBridge:
    return _tiny_decoder(
        GraniteConfig,
        GraniteForCausalLM,
        "GraniteForCausalLM",
        residual_multiplier=residual_multiplier,
    )


def _tiny_olmo2() -> TransformerBridge:
    return _tiny_decoder(Olmo2Config, Olmo2ForCausalLM, "Olmo2ForCausalLM")


def _tokens_and_caches(model, batch=1):
    generator = torch.Generator().manual_seed(42)
    clean_tokens = torch.randint(0, 100, (batch, 8), generator=generator)
    corrupted_tokens = torch.randint(0, 100, (batch, 8), generator=generator)
    with torch.no_grad():
        _, clean_cache = model.run_with_cache(clean_tokens)
        _, corrupted_cache = model.run_with_cache(corrupted_tokens)
    return clean_tokens, corrupted_tokens, clean_cache, corrupted_cache


@pytest.fixture(scope="module")
def tiny_model():
    return _tiny_gpt2()


@pytest.fixture(scope="module")
def unfolded_model():
    return _tiny_gpt2(fold_ln=False)


@pytest.fixture(scope="module")
def tokens_and_caches(tiny_model):
    return _tokens_and_caches(tiny_model)


def logit_diff(logits):
    return logits[0, -1, 17] - logits[0, -1, 42]


def _direct_path(model, corrupted_tokens, clean_cache, corrupted_cache, **kwargs):
    kwargs.setdefault("patching_metric", logit_diff)
    kwargs.setdefault("src_layer", 0)
    kwargs.setdefault("src_head", 0)
    with torch.no_grad():
        return get_act_patch_direct_path(
            model, corrupted_tokens, clean_cache, corrupted_cache, verbose=False, **kwargs
        )


class TestOutputShape:
    def test_single_source_shape(self, tiny_model, tokens_and_caches):
        _, corrupted_tokens, clean_cache, corrupted_cache = tokens_and_caches
        results = _direct_path(tiny_model, corrupted_tokens, clean_cache, corrupted_cache)
        assert results.shape == (tiny_model.cfg.n_layers, tiny_model.cfg.n_heads)

    def test_all_sources_shape(self, tiny_model, tokens_and_caches):
        _, corrupted_tokens, clean_cache, corrupted_cache = tokens_and_caches
        with torch.no_grad():
            results = get_act_patch_direct_path_all_sources(
                tiny_model,
                corrupted_tokens,
                clean_cache,
                corrupted_cache,
                logit_diff,
                verbose=False,
            )
        n, h = tiny_model.cfg.n_layers, tiny_model.cfg.n_heads
        assert results.shape == (n, h, n, h)

    @pytest.mark.parametrize("component", ["q", "k", "v"])
    def test_all_components(self, tiny_model, tokens_and_caches, component):
        _, corrupted_tokens, clean_cache, corrupted_cache = tokens_and_caches
        results = _direct_path(
            tiny_model,
            corrupted_tokens,
            clean_cache,
            corrupted_cache,
            src_head=1,
            component=component,
        )
        assert results.shape == (tiny_model.cfg.n_layers, tiny_model.cfg.n_heads)


class TestCausalStructure:
    def test_earlier_layers_are_zero(self, tiny_model, tokens_and_caches):
        _, corrupted_tokens, clean_cache, corrupted_cache = tokens_and_caches
        results = _direct_path(
            tiny_model, corrupted_tokens, clean_cache, corrupted_cache, src_layer=1
        )
        assert results[:2].eq(0).all()

    def test_later_layers_move_the_metric(self, tiny_model, tokens_and_caches):
        _, corrupted_tokens, clean_cache, corrupted_cache = tokens_and_caches
        with torch.no_grad():
            baseline = logit_diff(tiny_model(corrupted_tokens)).item()
        results = _direct_path(tiny_model, corrupted_tokens, clean_cache, corrupted_cache)
        assert (results[1:] - baseline).abs().max() > 1e-4

    def test_clean_equals_corrupted_gives_baseline(self, tiny_model, tokens_and_caches):
        _, tokens, _, cache = tokens_and_caches
        with torch.no_grad():
            baseline = logit_diff(tiny_model(tokens)).item()
        results = _direct_path(tiny_model, tokens, cache, cache)
        torch.testing.assert_close(
            results[1:], torch.full_like(results[1:], baseline), atol=_EXACT_ATOL, rtol=0
        )


def _ablate_path(model, clean_tokens, corrupted_tokens, src, dst, component):
    """The same path patch through the edge-ablation primitive, one edge per position."""
    edges = [
        (Node("attn_head_out", pos, *src), Node(f"{component}_input", pos, *dst))
        for pos in range(corrupted_tokens.shape[1])
    ]
    with torch.no_grad(), _edge_hook_flags(model):
        replacement = capture_writer_outputs(model, clean_tokens)
        return logit_diff(_ablate_edges(model, corrupted_tokens, edges, [], replacement)).item()


def _fork_reference(model, clean_tokens, corrupted_tokens, src, dst, component):
    """Shift dst's fork input by src's hook_result difference, by hand."""
    result_name = f"blocks.{src[0]}.attn.hook_result"
    model.set_use_attn_result(True)
    try:
        with torch.no_grad():
            _, clean = model.run_with_cache(clean_tokens, names_filter=result_name)
            _, corrupted = model.run_with_cache(corrupted_tokens, names_filter=result_name)
    finally:
        model.set_use_attn_result(False)
    delta = clean[result_name][:, :, src[1]] - corrupted[result_name][:, :, src[1]]

    def shift(fork_input, hook):
        fork_input = fork_input.clone()
        fork_input[:, :, dst[1]] = fork_input[:, :, dst[1]] + delta
        return fork_input

    model.set_use_split_qkv_input(True)
    try:
        with torch.no_grad():
            logits = model.run_with_hooks(
                corrupted_tokens,
                fwd_hooks=[(f"blocks.{dst[0]}.attn.hook_{component}_input", shift)],
            )
    finally:
        model.set_use_split_qkv_input(False)
    return logit_diff(logits).item()


def _frozen_scale_patch(model, corrupted_tokens, clean_cache, corrupted_cache, src, dst, component):
    """Frozen-scale linearisation: divide by the corrupted run's ln1 scale, project, add."""
    W_O = model.blocks[src[0]].attn.W_O[src[1]]
    z_name = f"blocks.{src[0]}.attn.hook_z"
    delta_resid = (clean_cache[z_name][:, :, src[1]] - corrupted_cache[z_name][:, :, src[1]]) @ W_O
    ln_scale = corrupted_cache[f"blocks.{dst[0]}.ln1.hook_scale"]
    W_comp = getattr(model.blocks[dst[0]].attn, f"W_{component.upper()}")[dst[1]]
    delta = (delta_resid / ln_scale) @ W_comp

    def add(value, hook):
        value = value.clone()
        value[:, :, dst[1]] = value[:, :, dst[1]] + delta
        return value

    with torch.no_grad():
        logits = model.run_with_hooks(
            corrupted_tokens, fwd_hooks=[(f"blocks.{dst[0]}.attn.hook_{component}", add)]
        )
    return logit_diff(logits).item()


_SRC, _DST = (0, 0), (2, 1)


class TestCorrectness:
    @pytest.mark.parametrize("component", ["q", "k", "v"])
    def test_matches_hand_written_fork_patch(self, tiny_model, tokens_and_caches, component):
        clean_tokens, corrupted_tokens, clean_cache, corrupted_cache = tokens_and_caches
        ours = _direct_path(
            tiny_model, corrupted_tokens, clean_cache, corrupted_cache, component=component
        )[_DST].item()
        ref = _fork_reference(tiny_model, clean_tokens, corrupted_tokens, _SRC, _DST, component)
        assert abs(ours - ref) < _EXACT_ATOL

    @pytest.mark.parametrize("model_fixture", ["tiny_model", "unfolded_model"])
    @pytest.mark.parametrize("component", ["q", "k", "v"])
    def test_matches_edge_ablation(self, request, model_fixture, component):
        model = request.getfixturevalue(model_fixture)
        clean_tokens, corrupted_tokens, clean_cache, corrupted_cache = _tokens_and_caches(model)
        ours = _direct_path(
            model, corrupted_tokens, clean_cache, corrupted_cache, component=component
        )[_DST].item()
        ablated = _ablate_path(model, clean_tokens, corrupted_tokens, _SRC, _DST, component)
        assert abs(ours - ablated) < _EXACT_ATOL

    @pytest.mark.parametrize("component", ["q", "k", "v"])
    def test_frozen_scale_linearisation_is_resolvably_different(
        self, tiny_model, tokens_and_caches, component
    ):
        """The exactness tolerance resolves the frozen-scale linearisation from the exact patch."""
        _, corrupted_tokens, clean_cache, corrupted_cache = tokens_and_caches
        ours = _direct_path(
            tiny_model, corrupted_tokens, clean_cache, corrupted_cache, component=component
        )[_DST].item()
        frozen = _frozen_scale_patch(
            tiny_model, corrupted_tokens, clean_cache, corrupted_cache, _SRC, _DST, component
        )
        assert abs(ours - frozen) > 100 * _EXACT_ATOL

    def test_batched_sweep_patches_every_example(self, tiny_model):
        _, corrupted_tokens, clean_cache, corrupted_cache = _tokens_and_caches(tiny_model, batch=2)

        def summed_logit_diff(logits):
            return (logits[:, -1, 17] - logits[:, -1, 42]).sum()

        batched = _direct_path(
            tiny_model,
            corrupted_tokens,
            clean_cache,
            corrupted_cache,
            patching_metric=summed_logit_diff,
        )
        per_example = sum(
            _direct_path(
                tiny_model,
                corrupted_tokens[i : i + 1],
                clean_cache.apply_slice_to_batch_dim((i, i + 1)),
                corrupted_cache.apply_slice_to_batch_dim((i, i + 1)),
            )
            for i in range(2)
        )
        torch.testing.assert_close(batched, per_example, atol=_EXACT_ATOL, rtol=0)

    def test_all_sources_consistent_with_single(self, tiny_model, tokens_and_caches):
        _, corrupted_tokens, clean_cache, corrupted_cache = tokens_and_caches
        single = _direct_path(
            tiny_model, corrupted_tokens, clean_cache, corrupted_cache, src_head=2
        )
        with torch.no_grad():
            all_sources = get_act_patch_direct_path_all_sources(
                tiny_model,
                corrupted_tokens,
                clean_cache,
                corrupted_cache,
                logit_diff,
                verbose=False,
            )
        assert torch.equal(single, all_sources[0, 2])


class TestEdgeCases:
    def test_last_layer_source_all_zero(self, tiny_model, tokens_and_caches):
        _, corrupted_tokens, clean_cache, corrupted_cache = tokens_and_caches
        results = _direct_path(
            tiny_model, corrupted_tokens, clean_cache, corrupted_cache, src_layer=2
        )
        assert results.eq(0).all()

    def test_returns_cpu_tensor(self, tiny_model, tokens_and_caches):
        _, corrupted_tokens, clean_cache, corrupted_cache = tokens_and_caches
        results = _direct_path(tiny_model, corrupted_tokens, clean_cache, corrupted_cache)
        assert results.device.type == tiny_model.cfg.device or results.is_cpu


def _fresh_model_and_cache(build=_tiny_gpt2):
    model = build()
    tokens = torch.randint(0, 100, (1, 8), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    return model, tokens, cache


def _count_forwards(model):
    calls: list[int] = []
    # run_with_hooks calls forward() directly, so a pre-hook on the bridge itself never fires.
    model.embed.register_forward_pre_hook(lambda module, args: calls.append(1))
    return calls


def _sweep(model, tokens, cache, src_layer=0, component="q", metric=logit_diff):
    return get_act_patch_direct_path(
        model, tokens, cache, cache, metric, src_layer, 0, component, verbose=False
    )


class TestSplitQkvFlags:
    @pytest.mark.parametrize("component", ["q", "k", "v"])
    @pytest.mark.parametrize("fails", [False, True], ids=["returns", "raises"])
    @pytest.mark.parametrize(
        "split, attn_in",
        [(False, False), (True, False), (False, True)],
        ids=["off", "split", "attn_in"],
    )
    def test_caller_flags_restored(self, split, attn_in, fails, component):
        model, tokens, cache = _fresh_model_and_cache(
            _tiny_gpt2 if component == "q" else _tiny_llama_gqa
        )
        model.set_use_split_qkv_input(split)
        model.set_use_attn_in(attn_in)

        def metric(logits):
            if fails:
                raise RuntimeError("metric failed")
            return logit_diff(logits)

        raised = pytest.raises(RuntimeError, match="metric failed")
        with torch.no_grad(), raised if fails else contextlib.nullcontext():
            _sweep(model, tokens, cache, metric=metric, component=component)
        assert (model.cfg.use_split_qkv_input, model.cfg.use_attn_in) == (split, attn_in)

    def test_destination_without_fork_refused(self, monkeypatch):
        model, tokens, cache = _fresh_model_and_cache()
        attn = model.blocks[2].attn
        fork = attn._fork_and_norm_per_head
        # Route the fork through an unhooked point, as an attention class without the fork would.
        monkeypatch.setattr(
            attn, "_fork_and_norm_per_head", lambda source, hook, n: fork(source, HookPoint(), n)
        )
        with torch.no_grad(), pytest.raises(NotImplementedError, match="did not fire"):
            _sweep(model, tokens, cache, src_layer=1)
        assert not model.cfg.use_split_qkv_input

    def test_model_without_fork_refused_before_any_forward(self):
        cfg = TransformerBridgeConfig(
            n_layers=2, d_model=32, d_head=8, n_heads=4, d_mlp=64, d_vocab=100, n_ctx=16
        )
        model, tokens, cache = _fresh_model_and_cache(
            lambda: TransformerBridge.boot_native(cfg).eval()
        )
        forwards = _count_forwards(model)
        with pytest.raises(NotImplementedError, match="use_split_qkv_input"):
            _sweep(model, tokens, cache)
        assert forwards == []
        assert not model.cfg.use_split_qkv_input


def _non_additive(detail: str) -> str:
    return rf"scaled before the residual add on this architecture \({detail}.*not hook_z @ W_O"


class TestArchitectureSupport:
    @pytest.mark.parametrize("component", ["k", "v"])
    def test_gqa_kv_uses_group_destinations(self, component):
        model, tokens, cache = _fresh_model_and_cache(_tiny_llama_gqa)
        forwards = _count_forwards(model)
        with torch.no_grad():
            result = _sweep(model, tokens, cache, component=component)
            last = _sweep(model, tokens, cache, src_layer=2, component=component)
        assert result.shape == (3, 2)
        assert len(forwards) == 4
        assert result[1:].isfinite().all()
        assert last.shape == (3, 2)
        assert last.eq(0).all()

    def test_gqa_q_still_runs(self):
        model, tokens, cache = _fresh_model_and_cache(_tiny_llama_gqa)
        assert model.cfg.n_key_value_heads == 2
        with torch.no_grad():
            results = _sweep(model, tokens, cache, component="q")
        assert results.shape == (model.cfg.n_layers, model.cfg.n_heads)
        assert results[1:].isfinite().all()

    def test_per_layer_kv_count_checked_on_destination_layers(self):
        model, tokens, cache = _fresh_model_and_cache()
        model.cfg.per_layer_num_key_value_heads = [2, 2, 4]
        with torch.no_grad():
            _sweep(model, tokens, cache, src_layer=1, component="k")
        model.cfg.per_layer_num_key_value_heads = [4, 2, 4]
        forwards = _count_forwards(model)
        with pytest.raises(ValueError, match="destination layers disagree on KV count"):
            _sweep(model, tokens, cache, component="k")
        assert forwards == []

    def test_block_without_attention_refused_before_any_forward(self, monkeypatch):
        model, tokens, cache = _fresh_model_and_cache()
        monkeypatch.delattr(model.blocks[1], "attn")
        forwards = _count_forwards(model)
        for src_layer in (0, 1):
            with pytest.raises(ValueError, match=r"blocks\.1 has no `attn` component"):
                _sweep(model, tokens, cache, src_layer=src_layer)
        with pytest.raises(ValueError, match=r"blocks\.1 has no `attn` component"):
            get_act_patch_direct_path_all_sources(
                model, tokens, cache, cache, logit_diff, verbose=False
            )
        assert forwards == []
        assert _sweep(model, tokens, cache, src_layer=2).eq(0).all()

    @pytest.mark.parametrize(
        "build, detail",
        [
            (lambda: _tiny_granite(0.25), r"blocks\.0 scales its attention output by 0\.25"),
            (_tiny_olmo2, r"blocks\.0\.hook_attn_out reads ln1_post"),
        ],
        ids=["scaled-residual", "post-norm"],
    )
    def test_transformed_attention_write_refused_before_any_forward(self, build, detail):
        model, tokens, cache = _fresh_model_and_cache(build)
        forwards = _count_forwards(model)
        with pytest.raises(ValueError, match=_non_additive(detail)):
            _sweep(model, tokens, cache)
        with pytest.raises(ValueError, match=_non_additive(detail)):
            get_act_patch_direct_path_all_sources(
                model, tokens, cache, cache, logit_diff, verbose=False
            )
        assert forwards == []

    def test_unscaled_granite_runs(self):
        model, tokens, cache = _fresh_model_and_cache(lambda: _tiny_granite(1.0))
        with torch.no_grad():
            results = _sweep(model, tokens, cache)
        assert results[1:].isfinite().all()

    def test_only_source_blocks_need_an_additive_write(self, monkeypatch):
        model, tokens, cache = _fresh_model_and_cache()
        monkeypatch.setattr(model.blocks[1], "residual_contribution_scale", 0.5, raising=False)
        forwards = _count_forwards(model)
        refusal = _non_additive(r"blocks\.1 scales its attention output by 0\.5")
        with pytest.raises(ValueError, match=refusal):
            _sweep(model, tokens, cache, src_layer=1)
        with pytest.raises(ValueError, match=refusal):
            get_act_patch_direct_path_all_sources(
                model, tokens, cache, cache, logit_diff, verbose=False
            )
        assert forwards == []
        with torch.no_grad():
            assert _sweep(model, tokens, cache, src_layer=0)[1:].isfinite().all()

    @pytest.mark.parametrize(
        "build, drop_attn, component, refusal",
        [
            (_tiny_gpt2, True, "q", r"blocks\.2 has no `attn` component"),
        ],
        ids=["no-attn"],
    )
    def test_all_sources_refuses_before_any_forward(
        self, monkeypatch, build, drop_attn, component, refusal
    ):
        model, tokens, cache = _fresh_model_and_cache(build)
        if drop_attn:
            monkeypatch.delattr(model.blocks[2], "attn")
        forwards = _count_forwards(model)
        with pytest.raises(ValueError, match=refusal):
            get_act_patch_direct_path_all_sources(
                model, tokens, cache, cache, logit_diff, component, verbose=False
            )
        assert forwards == []


@pytest.fixture(
    scope="module", params=[(kv, folded) for kv in (1, 2, 4) for folded in (False, True)]
)
def kv_model_and_caches(request):
    kv, folded = request.param
    model = _tiny_llama_gqa(kv, folded)
    return model, _tokens_and_caches(model, batch=2)


@pytest.mark.parametrize("component", ["k", "v"])
def test_kv_group_sweep_matches_projection_oracle(kv_model_and_caches, component):
    model, (clean, corrupt, clean_cache, corrupt_cache) = kv_model_and_caches
    kv = model.cfg.n_key_value_heads or model.cfg.n_heads
    src_head, dst_layer = 3, 1
    observations, current = [], {}

    def observe(name):
        def save(value, hook):
            current[name] = value.detach().clone()

        return save

    def metric(logits):
        observations.append(current.copy())
        return (logits[:, -1, 17] - logits[:, -1, 42]).sum()

    with model.hooks(
        fwd_hooks=[
            (f"blocks.{dst_layer}.attn.hook_{name}", observe(name)) for name in ("q", "k", "v", "z")
        ]
    ):
        result = _direct_path(
            model,
            corrupt,
            clean_cache,
            corrupt_cache,
            component=component,
            src_head=src_head,
            patching_metric=metric,
        )
    assert result.shape == (3, kv)
    assert len(observations) == 2 * kv
    assert result[0].eq(0).all()
    delta = (
        clean_cache["blocks.0.attn.hook_z"][:, :, src_head]
        - corrupt_cache["blocks.0.attn.hook_z"][:, :, src_head]
    ) @ model.blocks[0].attn.W_O[src_head]
    with torch.no_grad():
        normalized = model.blocks[dst_layer].ln1.original_component(
            corrupt_cache[f"blocks.{dst_layer}.ln1.hook_in"] + delta
        )
        projection = getattr(model.blocks[dst_layer].attn, component).original_component
        projected = projection(normalized).reshape(*corrupt.shape, kv, model.cfg.d_head)
        for group in range(kv):

            def replace(value, hook):
                value = value.clone()
                value[:, :, group] = projected[:, :, group]
                return value

            oracle = model.run_with_hooks(
                corrupt, fwd_hooks=[(f"blocks.{dst_layer}.attn.hook_{component}", replace)]
            )
            expected = (oracle[:, -1, 17] - oracle[:, -1, 42]).sum()
            torch.testing.assert_close(
                result[dst_layer, group], expected.float(), atol=1e-6, rtol=0
            )
            snapshot = observations[group]
            for branch in ("q", "v" if component == "k" else "k"):
                torch.testing.assert_close(
                    snapshot[branch],
                    corrupt_cache[f"blocks.{dst_layer}.attn.hook_{branch}"],
                    atol=1e-12,
                    rtol=0,
                )
            others = [g for g in range(kv) if g != group]
            torch.testing.assert_close(
                snapshot[component][:, :, others],
                corrupt_cache[f"blocks.{dst_layer}.attn.hook_{component}"][:, :, others],
                atol=1e-12,
                rtol=0,
            )
            heads_per_group = model.cfg.n_heads // kv
            outside = [h for h in range(model.cfg.n_heads) if h // heads_per_group != group]
            torch.testing.assert_close(
                snapshot["z"][:, :, outside],
                corrupt_cache[f"blocks.{dst_layer}.attn.hook_z"][:, :, outside],
                atol=1e-12,
                rtol=0,
            )
            assert (
                snapshot[component][:, :, group]
                - corrupt_cache[f"blocks.{dst_layer}.attn.hook_{component}"][:, :, group]
            ).abs().max() > 1e-6
    assert not model.cfg.use_split_qkv_input


@pytest.mark.parametrize("component", ["k", "v"])
def test_kv_all_sources_and_zero_delta(kv_model_and_caches, component):
    model, (_, corrupt, clean_cache, corrupt_cache) = kv_model_and_caches
    with torch.no_grad():
        all_sources = get_act_patch_direct_path_all_sources(
            model, corrupt, clean_cache, corrupt_cache, logit_diff, component, verbose=False
        )
        baseline = logit_diff(model(corrupt))
        no_op = _direct_path(model, corrupt, corrupt_cache, corrupt_cache, component=component)
    kv = model.cfg.n_key_value_heads or model.cfg.n_heads
    assert all_sources.shape == (3, 4, 3, kv)
    assert all_sources[2].eq(0).all()
    single = _direct_path(
        model, corrupt, clean_cache, corrupt_cache, src_head=3, component=component
    )
    torch.testing.assert_close(all_sources[0, 3], single, atol=0, rtol=0)
    torch.testing.assert_close(no_op[1:], baseline.float().expand(2, kv), atol=1e-6, rtol=0)


@pytest.mark.parametrize("component", ["k", "v"])
def test_heterogeneous_kv_all_sources_refused_before_forward(component):
    model, tokens, cache = _fresh_model_and_cache(_tiny_llama_gqa)
    model.cfg.per_layer_num_key_value_heads = [2, 2, 1]
    forwards = _count_forwards(model)
    with pytest.raises(ValueError, match="destination layers disagree on KV count"):
        get_act_patch_direct_path_all_sources(
            model, tokens, cache, cache, logit_diff, component, verbose=False
        )
    assert forwards == []


@pytest.mark.parametrize("component", ["q", "k", "v"])
def test_uniform_destination_metadata_ignores_the_source_layer(component):
    model, tokens, cache = _fresh_model_and_cache(_tiny_llama_gqa)
    model.cfg.per_layer_num_key_value_heads = [1, None, 2]
    with torch.no_grad():
        result = get_act_patch_direct_path_all_sources(
            model, tokens, cache, cache, logit_diff, component, verbose=False
        )
    expected = 4 if component == "q" else 2
    assert result.shape == (3, 4, 3, expected)
    assert result[2].eq(0).all()


def test_q_destination_axis_does_not_require_uniform_kv_counts():
    model, tokens, cache = _fresh_model_and_cache(_tiny_llama_gqa)
    model.cfg.per_layer_num_key_value_heads = [2, 1, 2]
    with torch.no_grad():
        result = _sweep(model, tokens, cache, component="q")
    assert result.shape == (3, 4)


def _training_flags(model):
    modules = [*model.named_modules(), *model.original_model.named_modules()]
    return [(name, module.training) for name, module in modules]


class TestEvalModeGuard:
    @pytest.mark.parametrize(
        ("enter_training", "root_training"),
        [
            (lambda model: model.train(), True),
            (lambda model: model.blocks[1].attn.train(), False),
        ],
        ids=["whole-model", "one-attn"],
    )
    def test_both_entry_points_refuse_before_any_forward(self, enter_training, root_training):
        model, tokens, cache = _fresh_model_and_cache()
        enter_training(model)
        flags = _training_flags(model)
        forwards = _count_forwards(model)
        with pytest.raises(ValueError, match=r"get_act_patch_direct_path\(\) .*model\.eval\(\)"):
            _sweep(model, tokens, cache)
        with pytest.raises(
            ValueError, match=r"get_act_patch_direct_path_all_sources\(\) .*model\.eval\(\)"
        ):
            get_act_patch_direct_path_all_sources(
                model, tokens, cache, cache, logit_diff, verbose=False
            )
        assert forwards == []
        assert model.training is root_training
        assert _training_flags(model) == flags


def test_old_module_path_warns_that_results_changed():
    from transformer_lens.tools.analysis import direct_path_patching

    for name, moved in [
        ("get_act_patch_direct_path", get_act_patch_direct_path),
        ("get_act_patch_direct_path_all_sources", get_act_patch_direct_path_all_sources),
    ]:
        with pytest.warns(FutureWarning, match="transformer_lens.patching.*results differ"):
            assert getattr(direct_path_patching, name) is moved


def test_old_module_star_import_binds_the_moved_names_and_warns():
    from transformer_lens.tools.analysis import direct_path_patching

    namespace: dict = {}
    with pytest.warns(FutureWarning, match="transformer_lens.patching"):
        exec("from transformer_lens.tools.analysis.direct_path_patching import *", namespace)
    namespace.pop("__builtins__")
    assert namespace == {
        "get_act_patch_direct_path": get_act_patch_direct_path,
        "get_act_patch_direct_path_all_sources": get_act_patch_direct_path_all_sources,
    }
    assert set(namespace) <= set(dir(direct_path_patching))
