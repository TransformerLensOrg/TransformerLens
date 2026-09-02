"""Real-model integration tests for the TransformerBridge Jacobian lens paths.

The cached GPT-2 model exercises published-artifact loading, top-k and full-logit
readouts, a native fit, and causal steering. A tiny random Gemma2 bridge keeps
RMSNorm and logit-softcap coverage in the regular CI suite, while a slow
Gemma-2-2b-it test checks the published artifact on the real architecture.
"""

import pytest
import torch

PROMPT = "The Eiffel Tower is in the city of"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "a4114d7752d11eb546e6cf372213d7e75526d3a1"
GPT2_LENS_FILE = "gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"
GEMMA_MODEL = "google/gemma-2-2b-it"
GEMMA_LENS_FILE = "gemma-2-2b-it/jlens/Salesforce-wikitext/gemma-2-2b-it_jacobian_lens.pt"
FIT_CORPUS = "jacobian-lens-integration-prompts-v1"

# Long enough to clear the default 16-position source skip.
FIT_PROMPTS = [
    "The industrial revolution transformed the economies of Europe during the "
    "nineteenth century, as steam power, railways, and mechanized factories "
    "reorganized both production and daily life in the growing cities.",
    "Astronomers studying distant galaxies rely on the redshift of spectral "
    "lines to estimate how quickly those galaxies recede, which in turn "
    "constrains the expansion history of the observable universe.",
]


@pytest.fixture(scope="module")
def gpt2_bridge():
    from transformer_lens.model_bridge import TransformerBridge

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TransformerBridge.boot_transformers("gpt2", dtype=torch.float32, device=device)


@pytest.fixture(scope="module")
def published_gpt2_lens(gpt2_bridge):
    from transformer_lens.tools.analysis import JacobianLens

    return JacobianLens.from_pretrained(
        LENS_REPO,
        filename=GPT2_LENS_FILE,
        revision=LENS_REVISION,
        model=gpt2_bridge,
    )


def test_published_lens_loads_and_validates(published_gpt2_lens, gpt2_bridge):
    assert published_gpt2_lens.d_model == 768
    assert published_gpt2_lens.source_layers == list(range(11))
    assert published_gpt2_lens.n_prompts > 0
    assert published_gpt2_lens.validate_model(gpt2_bridge) is published_gpt2_lens


def test_readout_defaults_to_topk_and_final_layer_matches_model(published_gpt2_lens, gpt2_bridge):
    result = published_gpt2_lens.readout(gpt2_bridge, PROMPT, layers=[11], positions=[-1])

    assert result.lens_logits is None
    assert result.model_logits is None
    assert result.lens_topk_values[11].shape == (1, 10)
    assert result.lens_topk_indices[11].shape == (1, 10)
    torch.testing.assert_close(result.lens_topk_values[11], result.model_topk_values)
    assert torch.equal(result.lens_topk_indices[11], result.model_topk_indices)


def test_late_layer_readout_aligns_with_model_output(published_gpt2_lens, gpt2_bridge):
    """The model's top-1 token moves from a poor early rank to first at output."""
    result = published_gpt2_lens.readout(
        gpt2_bridge,
        PROMPT,
        layers=[0, 10, 11],
        positions=[-1],
        return_full_logits=True,
    )
    assert result.lens_logits is not None
    assert result.model_logits is not None
    model_top1 = result.model_logits[-1].argmax().item()

    def rank_of_model_top1(layer):
        logits = result.lens_logits[layer][-1]
        return int((logits > logits[model_top1]).sum().item())

    assert rank_of_model_top1(0) >= 1000
    assert rank_of_model_top1(10) <= 50
    assert rank_of_model_top1(11) == 0


def test_logit_lens_baseline_differs_mid_stack(published_gpt2_lens, gpt2_bridge):
    """The Jacobian and identity transports must diverge in the middle."""
    lens_result = published_gpt2_lens.readout(
        gpt2_bridge,
        PROMPT,
        layers=[5],
        positions=[-1],
        return_full_logits=True,
    )
    baseline = published_gpt2_lens.readout(
        gpt2_bridge,
        PROMPT,
        layers=[5],
        positions=[-1],
        use_jacobian=False,
        return_full_logits=True,
    )
    assert lens_result.lens_logits is not None
    assert baseline.lens_logits is not None
    assert not torch.allclose(lens_result.lens_logits[5], baseline.lens_logits[5])


def test_bridge_steering_raises_target_logit(published_gpt2_lens, gpt2_bridge):
    """A real Bridge run must honor hooks built from a published lens."""
    target = " Paris"
    target_id = gpt2_bridge.to_single_token(target)
    tokens = gpt2_bridge.to_tokens(PROMPT)
    hooks = published_gpt2_lens.steering_hooks(gpt2_bridge, target, layers=[6, 7, 8], alpha=6.0)

    with torch.no_grad():
        baseline = gpt2_bridge(tokens)[0, -1, target_id].item()
        steered = gpt2_bridge.run_with_hooks(tokens, fwd_hooks=hooks)[0, -1, target_id].item()

    assert steered > baseline


@pytest.mark.slow
def test_bridge_fit_merge_consistency_and_finiteness(gpt2_bridge):
    """Joint fitting equals merging per-prompt fits for a real Bridge."""
    from transformer_lens.tools.analysis import JacobianLens

    fit_kwargs = dict(
        source_layers=[0, 5, 10],
        dim_batch=96,
        max_seq_len=32,
        show_progress=False,
    )
    joint = JacobianLens.fit(gpt2_bridge, FIT_PROMPTS, corpus=FIT_CORPUS, **fit_kwargs)
    merged = JacobianLens.merge(
        [
            JacobianLens.fit(gpt2_bridge, [prompt], corpus=FIT_CORPUS, **fit_kwargs)
            for prompt in FIT_PROMPTS
        ]
    )

    assert joint.n_prompts == merged.n_prompts == 2
    assert joint.metadata["corpus"] == FIT_CORPUS
    for layer in joint.source_layers:
        assert torch.isfinite(joint.jacobians[layer]).all()
        torch.testing.assert_close(
            joint.jacobians[layer], merged.jacobians[layer], atol=1e-6, rtol=1e-5
        )


@pytest.mark.slow
def test_small_bridge_fit_converges_toward_published_lens(published_gpt2_lens, gpt2_bridge):
    """A two-prompt Bridge fit already points toward the published estimator."""
    from transformer_lens.tools.analysis import JacobianLens

    fitted = JacobianLens.fit(
        gpt2_bridge,
        FIT_PROMPTS,
        corpus=FIT_CORPUS,
        source_layers=[0, 5, 10],
        dim_batch=96,
        max_seq_len=32,
        show_progress=False,
    )
    cosines = {
        layer: torch.nn.functional.cosine_similarity(
            fitted.jacobians[layer].flatten(),
            published_gpt2_lens.jacobians[layer].flatten(),
            dim=0,
        ).item()
        for layer in fitted.source_layers
    }

    assert all(value >= 0.3 for value in cosines.values()), cosines
    assert cosines[10] >= 0.75, cosines


def test_tiny_gemma_bridge_readout_applies_rmsnorm_softcap_and_topk():
    """A small random Gemma2 keeps the architecture-specific path in CPU CI."""
    from transformers import Gemma2Config, Gemma2ForCausalLM

    from transformer_lens.model_bridge.sources import build_bridge_from_module
    from transformer_lens.tools.analysis import JacobianLens

    torch.manual_seed(0)
    config = Gemma2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        sliding_window=16,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    hf_model = Gemma2ForCausalLM(config).eval()
    with torch.no_grad():
        hf_model.lm_head.weight.mul_(1000)
    model = build_bridge_from_module(
        hf_model,
        "Gemma2ForCausalLM",
        hf_config=config,
        dtype=torch.float32,
        device="cpu",
        model_name="tiny-random-gemma2-jacobian-lens",
    )
    lens = JacobianLens(
        {0: torch.eye(config.hidden_size)},
        n_prompts=1,
        d_model=config.hidden_size,
        metadata={"target_layer": 1},
    )
    tokens = torch.tensor([[1, 7, 11, 3]])
    result = lens.readout(
        model,
        tokens,
        layers=[0, 1],
        positions=[-1],
        top_k=8,
        return_full_logits=True,
    )

    assert result.lens_logits is not None
    assert result.model_logits is not None
    assert model.cfg.output_logits_soft_cap == 30.0

    hook_names = ["blocks.0.hook_out", "blocks.1.hook_out"]
    with torch.no_grad():
        model_logits, cache = model.run_with_cache(tokens, names_filter=hook_names)
        source_residual = cache[hook_names[0]][0, -1]
        source_raw_logits = model.unembed(model.ln_final(source_residual.unsqueeze(0)))[0]
        final_residual = cache[hook_names[1]]
        final_raw_logits = model.unembed(model.ln_final(final_residual))[0, -1]

    soft_cap = float(model.cfg.output_logits_soft_cap)
    assert source_raw_logits.abs().max() > soft_cap
    expected_source_logits = (soft_cap * torch.tanh(source_raw_logits / soft_cap)).float()
    expected_final_logits = (soft_cap * torch.tanh(final_raw_logits / soft_cap)).float()
    torch.testing.assert_close(result.lens_logits[0][0], expected_source_logits)
    torch.testing.assert_close(result.lens_logits[1][0], expected_final_logits)
    torch.testing.assert_close(result.model_logits[0], model_logits[0, -1].float())
    for layer in (0, 1):
        expected_topk = result.lens_logits[layer].topk(8, dim=-1)
        torch.testing.assert_close(result.lens_topk_values[layer], expected_topk.values)
        torch.testing.assert_close(
            result.lens_topk_values[layer],
            result.lens_logits[layer].gather(-1, result.lens_topk_indices[layer]),
        )


def test_tiny_granite_bridge_readout_applies_logits_scaling():
    """A random real Granite bridge divides intermediate logits like HF."""
    from transformers import GraniteConfig, GraniteForCausalLM

    from transformer_lens.model_bridge.sources import build_bridge_from_module
    from transformer_lens.tools.analysis import JacobianLens

    config = GraniteConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
        logits_scaling=4.0,
        tie_word_embeddings=False,
    )
    model = build_bridge_from_module(
        GraniteForCausalLM(config).eval(),
        "GraniteForCausalLM",
        hf_config=config,
        dtype=torch.float32,
        device="cpu",
        model_name="tiny-random-granite-jacobian-lens",
    )
    lens = JacobianLens(
        {0: torch.eye(config.hidden_size)},
        n_prompts=1,
        d_model=config.hidden_size,
        metadata={"target_layer": 1},
    )
    tokens = torch.tensor([[1, 7, 11, 3]])
    result = lens.readout(
        model,
        tokens,
        layers=[0],
        positions=[-1],
        return_full_logits=True,
    )

    assert result.lens_logits is not None
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=["blocks.0.hook_out"])
        residual = cache["blocks.0.hook_out"][0, -1]
        raw_logits = model.unembed(model.ln_final(residual.unsqueeze(0)))[0]
    expected = raw_logits / config.logits_scaling

    torch.testing.assert_close(result.lens_logits[0][0], expected)
    assert not torch.allclose(result.lens_logits[0][0], raw_logits)


def test_bidirectional_bert_bridge_is_rejected_before_readout():
    """The BERT adapter declares the non-causal generation contract explicitly."""
    from transformers import BertConfig, BertForMaskedLM

    from transformer_lens.model_bridge.sources import build_bridge_from_module
    from transformer_lens.tools.analysis import JacobianLens

    config = BertConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=16,
    )
    model = build_bridge_from_module(
        BertForMaskedLM(config).eval(),
        "BertForMaskedLM",
        hf_config=config,
        dtype=torch.float32,
        device="cpu",
        model_name="tiny-random-bert-jacobian-lens-contract",
    )
    lens = JacobianLens(
        {0: torch.eye(config.hidden_size)},
        n_prompts=1,
        d_model=config.hidden_size,
        metadata={"target_layer": 1},
    )

    with pytest.raises(ValueError, match="causal decoder-only"):
        lens.validate_model(model)


@pytest.mark.slow
def test_gemma_published_artifact_parity_topk_and_softcap():
    """Gemma readout matches the artifact formula and its soft-capped output."""
    from transformer_lens.model_bridge import TransformerBridge
    from transformer_lens.tools.analysis import JacobianLens

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerBridge.boot_transformers(GEMMA_MODEL, dtype=torch.bfloat16, device=device)
    lens = JacobianLens.from_pretrained(
        LENS_REPO,
        filename=GEMMA_LENS_FILE,
        revision=LENS_REVISION,
        model=model,
    )
    source_layer = lens.source_layers[-1]
    final_layer = model.cfg.n_layers - 1
    result = lens.readout(
        model,
        PROMPT,
        layers=[source_layer, final_layer],
        positions=[-1],
        top_k=8,
        return_full_logits=True,
    )

    assert result.lens_logits is not None
    assert result.model_logits is not None

    def assert_valid_topk(
        values: torch.Tensor, indices: torch.Tensor, logits: torch.Tensor
    ) -> None:
        expected_values = logits.topk(8, dim=-1).values
        torch.testing.assert_close(values, expected_values)
        torch.testing.assert_close(values, logits.gather(-1, indices))

    for layer in (source_layer, final_layer):
        assert_valid_topk(
            result.lens_topk_values[layer],
            result.lens_topk_indices[layer],
            result.lens_logits[layer],
        )
    assert_valid_topk(
        result.model_topk_values,
        result.model_topk_indices,
        result.model_logits,
    )

    hook_names = [
        f"blocks.{source_layer}.hook_out",
        f"blocks.{final_layer}.hook_out",
    ]
    tokens = model.to_tokens(PROMPT)
    with torch.no_grad():
        model_logits, cache = model.run_with_cache(tokens, names_filter=hook_names)

        source_residual = cache[hook_names[0]][0, -1]
        source_matrix = lens.jacobians[source_layer].to(source_residual.device)
        transported = source_residual.float() @ source_matrix.T
        source_raw_logits = model.unembed(
            model.ln_final(transported.to(model.W_U.dtype).unsqueeze(0))
        )[0]

        final_residual = cache[hook_names[1]]
        final_raw_logits = model.unembed(model.ln_final(final_residual))[0, -1]

    soft_cap = float(model.cfg.output_logits_soft_cap)
    assert soft_cap == 30.0
    expected_source_logits = (soft_cap * torch.tanh(source_raw_logits / soft_cap)).float().cpu()
    expected_final_logits = (soft_cap * torch.tanh(final_raw_logits / soft_cap)).float().cpu()

    torch.testing.assert_close(
        result.lens_logits[source_layer][0],
        expected_source_logits,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        model_logits[0, -1].float().cpu(), expected_final_logits, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(result.model_logits[0], expected_final_logits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(result.lens_logits[final_layer], result.model_logits)


def test_decompose_gpt2_activation_reconstructs_and_is_orthogonal(published_gpt2_lens, gpt2_bridge):
    """decompose on a real GPT-2 activation: up to k active nonnegative atoms, the non-J-space
    residual is orthogonal to the selected J-lens vectors, and component + residual recover the
    activation."""
    layer, k = 6, 8
    result = published_gpt2_lens.decompose(gpt2_bridge, PROMPT, layer=layer, position=-1, k=k)

    # ``k`` is an upper bound: ``support`` holds only numerically active atoms, a subset of the
    # selected set. Do not assert a closed-model active count -- just record it on failure.
    assert (
        result.support.numel() <= result.selected_support.numel() <= k
    ), f"active={result.support.numel()} selected={result.selected_support.numel()} k={k}"
    assert set(result.support.tolist()).issubset(set(result.selected_support.tolist()))
    assert result.coordinates.numel() == result.support.numel()
    assert (result.coordinates > 0).all()  # every returned coordinate is active
    assert (result.support >= 0).all() and (result.support < gpt2_bridge.cfg.d_vocab).all()

    # the decomposed activation is the model's blocks.{layer}.hook_out at the last position
    tokens = gpt2_bridge.to_tokens(PROMPT)
    hook = f"blocks.{layer}.hook_out"
    _, cache = gpt2_bridge.run_with_cache(tokens, names_filter=lambda name: name == hook)
    activation = cache[hook][0, -1, :].float()
    assert torch.allclose(
        result.j_space_component + result.non_j_space_component, activation, atol=1e-3
    )

    # the non-J-space residual is orthogonal to every selected J-lens vector (the projection
    # span is the whole selected support, not only the active atoms); cosine ~ 0
    dictionary = published_gpt2_lens.lens_vector_dictionary(gpt2_bridge, layer)
    residual = result.non_j_space_component
    for atom_id in result.selected_support.tolist():
        atom = dictionary[atom_id]
        cosine = torch.dot(residual, atom) / (residual.norm() * atom.norm())
        assert cosine.abs().item() < 1e-3


def test_coordinate_patch_gpt2_preserves_anchored_frame(published_gpt2_lens, gpt2_bridge):
    """A published Bridge lens produces a finite edit while preserving its sparse frame."""
    layer, k = 6, 8
    tokens = gpt2_bridge.to_tokens(PROMPT)
    hook = f"blocks.{layer}.hook_out"
    _, cache = gpt2_bridge.run_with_cache(tokens, names_filter=lambda name: name == hook)
    activation = cache[hook][0, -1, :].float()
    dictionary = published_gpt2_lens.lens_vector_dictionary(gpt2_bridge, layer)
    decomposition = published_gpt2_lens.decompose(gpt2_bridge, activation, layer=layer, k=k)
    source_id = int(decomposition.support[0])
    units = dictionary / dictionary.norm(dim=1, keepdim=True)
    pair_cosines = (units @ units[source_id]).abs()
    pair_cosines[source_id] = torch.inf
    target_id = int(pair_cosines.argmin().item())

    result = published_gpt2_lens.coordinate_patch(
        gpt2_bridge,
        PROMPT,
        layer,
        source_id,
        target_id,
        position=-1,
        decomposition=decomposition,
        alpha=0.5,
    )
    no_op = published_gpt2_lens.coordinate_patch(
        gpt2_bridge,
        activation,
        layer,
        source_id,
        target_id,
        decomposition=decomposition,
        alpha=0.0,
    )

    assert torch.isfinite(result.patched).all()
    torch.testing.assert_close(
        result.patched - result.reconstruction_after,
        result.residual,
        atol=1e-5,
        rtol=1e-5,
    )
    edited_positions = {
        result.support_after.tolist().index(source_id),
        result.support_after.tolist().index(target_id),
    }
    for position in range(result.coordinates_before.numel()):
        if position not in edited_positions:
            assert torch.equal(
                result.coordinates_after[position], result.coordinates_before[position]
            )
    assert torch.equal(no_op.patched, activation)


def test_occupancy_gpt2_activation_is_a_small_positive_integer(published_gpt2_lens, gpt2_bridge):
    """occupancy on a real GPT-2 activation returns a positive integer within ``[1, max_atoms]``,
    with per-step real and control captured-variance curves of the right shape. We assert shape and
    bounds -- not the paper's closed-model count (an open-weight observation, recorded on failure).
    A reduced control count keeps CI fast."""
    layer, max_atoms = 6, 12
    result = published_gpt2_lens.occupancy(
        gpt2_bridge,
        PROMPT,
        layer=layer,
        position=-1,
        max_atoms=max_atoms,
        num_control_dictionaries=8,
    )

    assert isinstance(result.occupancy, int)
    assert 1 <= result.occupancy <= max_atoms, f"occupancy={result.occupancy} max_atoms={max_atoms}"
    assert result.marginal_captured_variance.shape == (max_atoms,)
    assert result.control_captured_variance.shape == (max_atoms,)
    assert result.support.shape == (max_atoms,)
    assert (result.support >= 0).all() and (result.support < gpt2_bridge.cfg.d_vocab).all()

    # Cumulative captured variance is a projection ratio: non-decreasing and bounded to [0, 1].
    real_cumulative = result.marginal_captured_variance.cumsum(0)
    assert (result.marginal_captured_variance >= -1e-4).all()
    assert (real_cumulative >= -1e-4).all() and (real_cumulative <= 1.0 + 1e-4).all()

    # Deterministic given the seed: an identical call reproduces the count and both curves exactly.
    repeat = published_gpt2_lens.occupancy(
        gpt2_bridge,
        PROMPT,
        layer=layer,
        position=-1,
        max_atoms=max_atoms,
        num_control_dictionaries=8,
    )
    assert repeat.occupancy == result.occupancy
    assert torch.equal(repeat.marginal_captured_variance, result.marginal_captured_variance)
    assert torch.equal(repeat.control_captured_variance, result.control_captured_variance)


def test_fraction_of_variance_gpt2_is_a_small_ratio(published_gpt2_lens, gpt2_bridge):
    """fraction_of_variance over a small corpus: each layer's median and pooled ratio land in
    ``[0, 1]`` with samples recorded, consistent with the paper's "J-space is a small fraction of
    total variance" (an open-weight observation, not a numeric match to the closed-model figures).
    FIT_PROMPTS are long enough to clear the default 16-position skip."""
    layers = [3, 6]
    profile = published_gpt2_lens.fraction_of_variance(gpt2_bridge, FIT_PROMPTS, layers=layers, k=8)

    assert profile.layers == layers
    for layer in layers:
        per_position = profile.per_position[layer]
        assert per_position.numel() > 0
        # Each fraction is ||projection||^2 / ||activation||^2, a variance ratio in [0, 1].
        assert (per_position >= 0.0).all() and (per_position <= 1.0).all()
        assert 0.0 <= profile.median[layer] <= 1.0
        assert 0.0 <= profile.pooled[layer] <= 1.0

    # positions= overrides the skip_first sweep: one explicit position per prompt over the
    # two-prompt corpus exercises the override path and cross-prompt pooling (one sample each).
    pinned = published_gpt2_lens.fraction_of_variance(
        gpt2_bridge, FIT_PROMPTS, layers=layers, k=8, positions=[-1]
    )
    for layer in layers:
        assert pinned.per_position[layer].numel() == len(FIT_PROMPTS)
        assert 0.0 <= pinned.pooled[layer] <= 1.0


@pytest.mark.slow
def test_decompose_gemma_activation_is_valid():
    """Decompose a real gemma-2-2b-it activation via its published lens (slow: real download)."""
    from transformer_lens.model_bridge import TransformerBridge
    from transformer_lens.tools.analysis import JacobianLens

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerBridge.boot_transformers(GEMMA_MODEL, dtype=torch.bfloat16, device=device)
    lens = JacobianLens.from_pretrained(
        LENS_REPO, filename=GEMMA_LENS_FILE, revision=LENS_REVISION, model=model
    )
    layer = lens.source_layers[len(lens.source_layers) // 2]
    k = 16
    result = lens.decompose(model, PROMPT, layer=layer, position=-1, k=k)

    assert (
        result.support.numel() <= result.selected_support.numel() <= k
    ), f"active={result.support.numel()} selected={result.selected_support.numel()} k={k}"
    assert set(result.support.tolist()).issubset(set(result.selected_support.tolist()))
    assert (result.coordinates > 0).all()
    assert (result.support >= 0).all() and (result.support < model.cfg.d_vocab).all()

    tokens = model.to_tokens(PROMPT)
    hook = f"blocks.{layer}.hook_out"
    _, cache = model.run_with_cache(tokens, names_filter=lambda name: name == hook)
    activation = cache[hook][0, -1, :].float()
    assert torch.allclose(
        result.j_space_component + result.non_j_space_component, activation, atol=1e-2
    )
