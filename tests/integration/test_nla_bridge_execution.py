"""Real cached-model checks of private NLA extraction mechanics."""

import gc
import weakref

import pytest
import torch

from transformer_lens.tools.analysis._nla_artifacts import (
    ArtifactRef,
    NLAMetadata,
    NLAPairConfig,
)
from transformer_lens.tools.analysis._nla_contracts import (
    ActivationProvenance,
    NLAActivation,
    NLASamplingConfig,
    SourceProvenance,
)
from transformer_lens.tools.analysis._nla_target import (
    extract_activation,
    load_target,
    validate_target_structure,
)
from transformer_lens.tools.analysis._nla_verbalizer import generate_explanation


@pytest.fixture(scope="module")
def target_pair():
    from huggingface_hub import model_info

    repo = "hf-internal-testing/tiny-random-gpt2"
    revision = model_info(repo).sha
    assert revision is not None
    return NLAPairConfig(
        ArtifactRef(repo, revision),
        ArtifactRef("example/av", "a" * 40),
        ArtifactRef("example/ar", "b" * 40),
        "gpt2",
        32,
        1,
        5,
    )


@pytest.fixture(scope="module")
def target(target_pair):
    return load_target(target_pair, device="cpu")


def test_real_extraction_matches_block_cache(target, target_pair):
    tokens = torch.tensor([[1, 2, 3]])
    result = extract_activation(target, target_pair, "mechanical-fixture", tokens, position=-1)
    _, cache = target.run_with_cache(tokens, names_filter="blocks.1.hook_out", return_type=None)
    torch.testing.assert_close(result.vector, cache["blocks.1.hook_out"][0, -1])
    assert result.vector.device.type == "cpu" and not result.vector.requires_grad
    assert result.provenance.source_verified and result.provenance.position == 2
    assert result.vector.data_ptr() != cache["blocks.1.hook_out"].data_ptr()


@pytest.mark.parametrize("position", [-4, 3, True])
def test_invalid_position_does_not_run_forward(target, target_pair, position):
    with pytest.raises(ValueError, match="position"):
        extract_activation(
            target, target_pair, "mechanical-fixture", torch.tensor([1, 2, 3]), position=position
        )


def test_string_input_and_repeated_extraction(target, target_pair):
    a = extract_activation(
        target, target_pair, "mechanical-fixture", "Hello world", prepend_bos=False
    )
    b = extract_activation(
        target, target_pair, "mechanical-fixture", "Hello world", prepend_bos=False
    )
    torch.testing.assert_close(a.vector, b.vector)
    assert "prepend_bos=False" in a.provenance.tokenization_policy


def test_result_does_not_keep_target_alive(target_pair):
    model = load_target(target_pair, device="cpu")
    ref = weakref.ref(model)
    result = extract_activation(model, target_pair, "mechanical-fixture", torch.tensor([1, 2]))
    del model
    gc.collect()
    assert ref() is None
    assert result.vector.shape == (32,)


def test_processed_basis_rejected(target_pair):
    model = load_target(target_pair, device="cpu")
    model.compatibility_mode = True
    with pytest.raises(ValueError, match="raw"):
        validate_target_structure(model, target_pair)


@pytest.mark.parametrize(
    "attribute,value,message",
    [
        ("_weights_processed", True, "raw"),
        ("training", True, "eval"),
    ],
)
def test_mutated_target_guards(target_pair, attribute, value, message):
    model = load_target(target_pair, device="cpu")
    setattr(model, attribute, value)
    with pytest.raises(ValueError, match=message):
        validate_target_structure(model, target_pair)


def test_precision_guard(target_pair):
    model = load_target(target_pair, device="cpu")
    model.original_model.to(dtype=torch.float16)
    with pytest.raises(ValueError, match="float32"):
        validate_target_structure(model, target_pair)


def test_caller_hooks_preserved(target, target_pair):
    calls = []

    def observe(value, hook):
        calls.append(hook.name)
        return value

    with target.hooks(fwd_hooks=[("blocks.1.hook_out", observe)]):
        extract_activation(target, target_pair, "mechanical-fixture", torch.tensor([1, 2]))
        target(torch.tensor([[1, 2]]))
    assert calls == ["blocks.1.hook_out", "blocks.1.hook_out"]


@pytest.fixture
def actor_metadata():
    return NLAMetadata(
        "av",
        32,
        1,
        "marker",
        2,
        "{injection_char}",
        "{explanation}",
        32**0.5,
        5.0,
        (1, 2, 3),
        1,
        None,
    )


@pytest.mark.parametrize("cached", [True, False])
def test_actor_generation_hooks_and_continuation(target, actor_metadata, cached):
    observations = []
    activation = NLAActivation(
        torch.ones(32), ActivationProvenance("mechanical-fixture", SourceProvenance.UNKNOWN_VECTOR)
    )

    def observe(value, hook):
        observations.append(value.detach().clone())
        return value

    with target.hooks(fwd_hooks=[("hook_embed", observe)]):
        first = generate_explanation(
            target,
            actor_metadata,
            activation,
            NLASamplingConfig(max_new_tokens=3, do_sample=False, use_past_kv_cache=cached),
        )
        second = generate_explanation(
            target,
            actor_metadata,
            activation,
            NLASamplingConfig(max_new_tokens=3, do_sample=False, use_past_kv_cache=cached),
        )
        target(torch.tensor([[1, 2, 3]]))
    assert first.token_ids == second.token_ids
    assert 0 < len(first.token_ids) <= 3
    assert first.raw_text == target.tokenizer.decode(
        list(first.token_ids), skip_special_tokens=True
    )
    assert len(observations) == 7
    assert [x.shape[1] for x in observations[:3]] == ([3, 1, 1] if cached else [3, 4, 5])


def test_actor_exception_cleanup(target, actor_metadata):
    activation = NLAActivation(
        torch.ones(32), ActivationProvenance("mechanical-fixture", SourceProvenance.UNKNOWN_VECTOR)
    )
    calls = []

    def fail(value, hook):
        calls.append(1)
        raise RuntimeError("controlled failure")

    with target.hooks(fwd_hooks=[("blocks.1.hook_out", fail)]):
        with pytest.raises(RuntimeError, match="controlled failure"):
            generate_explanation(
                target, actor_metadata, activation, NLASamplingConfig(max_new_tokens=1)
            )
    assert calls == [1]
    assert not target.hook_dict["embed.hook_out"].fwd_hooks
    target(torch.tensor([[1, 2, 3]]))


def test_embedding_injection_logits_match_direct_hf(target):
    from transformer_lens.tools.analysis._nla_verbalizer import embedding_overwrite

    tokens = torch.tensor([[1, 2, 3]])
    vector = torch.arange(1.0, 33.0)
    replacement = vector.double() / vector.double().norm() * 5
    with torch.no_grad():
        embeddings = target.original_model.get_input_embeddings()(tokens).clone()
        embeddings[0, 1] = replacement.float()
        expected = target.original_model(inputs_embeds=embeddings).logits
        with target.hooks(
            fwd_hooks=[("hook_embed", embedding_overwrite(vector, 5.0, 1, 3, cached=True))]
        ):
            actual = target(tokens)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_actor_budget_rejected(target, actor_metadata):
    activation = NLAActivation(
        torch.ones(32), ActivationProvenance("mechanical-fixture", SourceProvenance.UNKNOWN_VECTOR)
    )
    with pytest.raises(ValueError, match="context"):
        generate_explanation(
            target, actor_metadata, activation, NLASamplingConfig(max_new_tokens=target.cfg.n_ctx)
        )


def test_non_bridge_rejected(target_pair):
    with pytest.raises(TypeError, match="TransformerBridge"):
        validate_target_structure(object(), target_pair)
