"""Download-free TransformerBridge readout geometry and tokenizer integration."""

import copy

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE, WordLevel
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

from tests.integration.model_bridge.helpers import make_tiny_pair
from tests.typecheck_errors import TYPECHECK_ERRORS
from transformer_lens.tools.analysis import RepresentationGeometry


def local_tokenizer():
    vocab = {
        "<unk>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<pad>": 3,
        "king": 4,
        "queen": 5,
        "man": 6,
        "woman": 7,
        "cat": 8,
        "cats": 9,
        "dog": 10,
        "dogs": 11,
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="<bos> $A <eos>", special_tokens=[("<bos>", 1), ("<eos>", 2)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
    )


@pytest.fixture
def tiny_bridge():
    config = GPT2Config(
        vocab_size=12,
        n_embd=4,
        n_layer=1,
        n_head=2,
        n_positions=16,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
    )
    bridge, _ = make_tiny_pair(config, "GPT2LMHeadModel", loader=GPT2LMHeadModel)
    bridge.tokenizer = local_tokenizer()
    return bridge


def test_bridge_geometry_matches_tensor_fit_without_mutating_model(tiny_bridge):
    bridge = tiny_bridge
    weights = {name: value.detach().clone() for name, value in bridge.state_dict().items()}
    config_before = copy.deepcopy(vars(bridge.cfg))
    training_before = [(name, module.training) for name, module in bridge.named_modules()]
    tokenizer_before = bridge.tokenizer.backend_tokenizer.to_str()
    pointers = {name: value.data_ptr() for name, value in bridge.named_parameters()}
    tokens = torch.tensor([[4, 6, 8]])
    with torch.no_grad():
        logits_before = bridge(tokens)
    geometry = RepresentationGeometry.from_bridge(bridge, compute_dtype=torch.float64)
    reference = RepresentationGeometry(bridge.W_U, compute_dtype=torch.float64)
    torch.testing.assert_close(geometry.covariance, reference.covariance)
    vector = torch.tensor([1.0, -2.0, 3.0, 4.0])
    torch.testing.assert_close(
        geometry.whiten_measurement(vector), reference.whiten_measurement(vector)
    )
    assert geometry.basis.source == "transformer-bridge"
    assert geometry.basis.input_location == "post-final-normalization"
    assert geometry.basis.normalization_type == "LN"
    assert geometry.basis.architecture == "GPT2LMHeadModel"
    assert geometry.diagnostics.input_shape == (4, 12)
    assert vars(bridge.cfg) == config_before
    assert training_before == [(name, module.training) for name, module in bridge.named_modules()]
    assert pointers == {name: value.data_ptr() for name, value in bridge.named_parameters()}
    assert bridge.tokenizer.backend_tokenizer.to_str() == tokenizer_before
    assert not bridge.compatibility_mode
    assert not bridge._weights_processed
    for name, value in bridge.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(bridge(tokens), logits_before, rtol=0, atol=0)


def test_post_normalization_measurement_pairing_matches_actual_logit_difference(tiny_bridge):
    geometry = RepresentationGeometry.from_bridge(tiny_bridge)
    concept = geometry.concept_direction([(4, 5)])
    with torch.no_grad():
        tiny_bridge.ln_final.original_component.weight.copy_(torch.tensor([0.5, 1.0, 2.0, 3.0]))
        tiny_bridge.ln_final.original_component.bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4]))
        tiny_bridge.unembed.original_component.bias.copy_(torch.linspace(-0.2, 0.2, 12))
    captured = []

    def observe(residual, hook):
        captured.append(residual.detach().clone())

    tokens = torch.tensor([[6, 4]])
    logits = tiny_bridge.run_with_hooks(tokens, fwd_hooks=[("unembed.hook_in", observe)])
    residual = captured[0]
    expected = logits[..., 5] - logits[..., 4] - (tiny_bridge.b_U[5] - tiny_bridge.b_U[4])
    pairing = (geometry.whiten_intervention(residual) * concept.whitened_measurement).sum(-1)
    torch.testing.assert_close(pairing, expected, atol=1e-6, rtol=1e-5)


def test_string_and_id_pairs_match_without_bos_or_eos(tiny_bridge):
    geometry = RepresentationGeometry.from_bridge(tiny_bridge)
    assert tiny_bridge.tokenizer.encode("king", add_special_tokens=True) == [1, 4, 2]
    strings = geometry.concept_direction([("king", "queen"), ("man", "woman")], label="gender")
    ids = geometry.concept_direction([(4, 5), (6, 7)])
    assert strings.pairs == ids.pairs == ((4, 5), (6, 7))
    assert strings.pair_labels == (("king", "queen"), ("man", "woman"))
    assert strings.label == "gender"
    torch.testing.assert_close(strings.raw_measurement, ids.raw_measurement)
    torch.testing.assert_close(strings.whitened_dispersion, ids.whitened_dispersion)
    mixed = geometry.concept_direction([("king", 5), (6, "woman")])
    assert mixed.pairs == ids.pairs
    assert mixed.pair_labels == strings.pair_labels
    custom = geometry.concept_direction([("king", "queen")], pair_labels=[("low", "high")])
    assert custom.pair_labels == (("low", "high"),)


@pytest.mark.parametrize("endpoint", ["", " ", "king queen", "missing"])
def test_invalid_string_endpoints_are_rejected(tiny_bridge, endpoint):
    geometry = RepresentationGeometry.from_bridge(tiny_bridge)
    with pytest.raises(ValueError, match="one token|empty|unknown"):
        geometry.concept_direction([(endpoint, "queen")])


def test_string_aliases_resolving_to_duplicate_or_self_pairs_are_rejected(tiny_bridge):
    geometry = RepresentationGeometry.from_bridge(tiny_bridge)
    with pytest.raises(ValueError, match="duplicates"):
        geometry.concept_direction([("king", "queen"), (4, 5)])
    with pytest.raises(ValueError, match="zero"):
        geometry.concept_direction([("king", 4)])


def test_tokenizer_and_weight_snapshots_survive_model_mutation(tiny_bridge):
    geometry = RepresentationGeometry.from_bridge(tiny_bridge)
    before = geometry.concept_direction([("king", "queen")])
    tiny_bridge.tokenizer.add_tokens(["king queen"])
    with torch.no_grad():
        tiny_bridge.unembed.original_component.weight.zero_()
    after = geometry.concept_direction([("king", "queen")])
    with pytest.raises(ValueError, match="one token"):
        geometry.concept_direction([("king queen", "queen")])
    assert after.pairs == before.pairs
    torch.testing.assert_close(after.raw_measurement, before.raw_measurement)


def test_tensor_construction_does_not_require_a_tokenizer(tiny_bridge):
    geometry = RepresentationGeometry(tiny_bridge.W_U)
    geometry.concept_direction([(4, 5)])
    with pytest.raises(ValueError, match="tokenizer"):
        geometry.concept_direction([("king", "queen")])


def test_bridge_without_tokenizer_still_supports_id_contrasts(tiny_bridge):
    tiny_bridge._tokenizer = None
    geometry = RepresentationGeometry.from_bridge(tiny_bridge)
    geometry.concept_direction([(4, 5)])
    with pytest.raises(ValueError, match="tokenizer"):
        geometry.concept_direction([("king", "queen")])


@pytest.mark.parametrize("flag", ["compatibility_mode", "_weights_processed"])
def test_processed_basis_is_rejected_before_readout_access(tiny_bridge, flag):
    setattr(tiny_bridge, flag, True)
    with pytest.raises(ValueError, match="raw|processed|compatibility"):
        RepresentationGeometry.from_bridge(tiny_bridge)


@pytest.mark.parametrize(
    "field, value",
    [
        ("normalization_type", "LNPre"),
        ("normalization_type", "RMS"),
        ("layer_norm_folding", True),
        ("d_model", 5),
        ("d_vocab_out", 13),
        ("output_logits_soft_cap", 2.0),
        ("output_logits_soft_cap", float("nan")),
        ("attention_dir", "bidirectional"),
    ],
)
def test_unsupported_or_inconsistent_configs_are_rejected(tiny_bridge, field, value):
    setattr(tiny_bridge.cfg, field, value)
    with pytest.raises(ValueError):
        RepresentationGeometry.from_bridge(tiny_bridge)


def test_missing_final_norm_is_rejected(tiny_bridge):
    del tiny_bridge.ln_final
    with pytest.raises(ValueError, match="ln_final"):
        RepresentationGeometry.from_bridge(tiny_bridge)


def test_non_linear_readout_is_rejected(tiny_bridge):
    tiny_bridge.unembed.set_original_component(torch.nn.Sequential(torch.nn.Linear(4, 12)))
    with pytest.raises(ValueError, match="linear"):
        RepresentationGeometry.from_bridge(tiny_bridge)


def test_custom_linear_forward_is_rejected(tiny_bridge):
    head = tiny_bridge.unembed.original_component
    head.forward = lambda hidden: torch.nn.functional.linear(hidden, head.weight, head.bias).tanh()
    with pytest.raises(ValueError, match="standard nn.Linear forward"):
        RepresentationGeometry.from_bridge(tiny_bridge)


def test_custom_post_readout_transform_is_rejected(tiny_bridge):
    tiny_bridge.adapter.apply_output_logits_transform = lambda logits: logits * 2
    with pytest.raises(ValueError, match="output transform"):
        RepresentationGeometry.from_bridge(tiny_bridge)


def test_additional_output_projection_is_rejected(tiny_bridge):
    tiny_bridge.adapter.component_mapping["project_out"] = object()
    with pytest.raises(ValueError, match="projection"):
        RepresentationGeometry.from_bridge(tiny_bridge)


def test_exposed_readout_must_match_actual_linear_head(tiny_bridge):
    actual = tiny_bridge.unembed.original_component.weight.detach().clone()
    tiny_bridge.unembed.register_parameter("_processed_W_U", torch.nn.Parameter(actual + 1.0))
    with pytest.raises(ValueError, match="actual linear readout"):
        RepresentationGeometry.from_bridge(tiny_bridge)


def test_non_finite_readout_is_not_misreported_as_an_accessor_mismatch(tiny_bridge):
    with torch.no_grad():
        tiny_bridge.unembed.original_component.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        RepresentationGeometry.from_bridge(tiny_bridge)


def test_identity_normalization_cannot_be_mislabeled_as_layer_norm(tiny_bridge):
    tiny_bridge.ln_final.set_original_component(torch.nn.Identity())
    with pytest.raises(ValueError, match="LayerNorm"):
        RepresentationGeometry.from_bridge(tiny_bridge)


def test_offloaded_meta_readout_is_rejected_without_materializing_weights(tiny_bridge):
    head = tiny_bridge.unembed.original_component
    head.weight = torch.nn.Parameter(torch.empty_like(head.weight, device="meta"))
    with pytest.raises(ValueError, match="materialized"):
        RepresentationGeometry.from_bridge(tiny_bridge)
    assert head.weight.is_meta


def test_parametrized_readout_is_rejected_without_updating_training_buffers(tiny_bridge):
    torch.nn.utils.parametrizations.spectral_norm(tiny_bridge.unembed.original_component)
    tiny_bridge.train()
    before = {name: value.clone() for name, value in tiny_bridge.state_dict().items()}
    with pytest.raises(ValueError, match="parametrizations"):
        RepresentationGeometry.from_bridge(tiny_bridge)
    for name, value in tiny_bridge.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_geometry_does_not_change_training_modes_or_execute_hooks(tiny_bridge):
    tiny_bridge.train()
    calls = []
    tiny_bridge.add_hook(
        "unembed.hook_in", lambda value, hook: calls.append(value), is_permanent=True
    )
    before = [(name, module.training) for name, module in tiny_bridge.named_modules()]
    handles = tuple(tiny_bridge.unembed.hook_in.fwd_hooks)
    RepresentationGeometry.from_bridge(tiny_bridge)
    assert not calls
    assert before == [(name, module.training) for name, module in tiny_bridge.named_modules()]
    assert handles == tuple(tiny_bridge.unembed.hook_in.fwd_hooks)


def test_string_encoding_does_not_modify_source_padding_or_truncation(tiny_bridge):
    backend = tiny_bridge.tokenizer.backend_tokenizer
    backend.enable_truncation(max_length=1)
    backend.enable_padding(length=8)
    before = backend.to_str()
    geometry = RepresentationGeometry.from_bridge(tiny_bridge)
    geometry.concept_direction([("king", "queen")])
    with pytest.raises(ValueError, match="one token"):
        geometry.concept_direction([("king queen", "woman")])
    assert backend.to_str() == before


def test_rms_normalization_bridge_uses_the_same_unfolded_readout_contract():
    config = LlamaConfig(
        vocab_size=12,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
    )
    bridge, _ = make_tiny_pair(config, "LlamaForCausalLM", loader=LlamaForCausalLM)
    geometry = RepresentationGeometry.from_bridge(bridge)
    assert geometry.basis.normalization_type == "RMS"
    torch.testing.assert_close(geometry.covariance, RepresentationGeometry(bridge.W_U).covariance)


def test_byte_level_tokenizer_preserves_leading_space_token_identity():
    pieces = [
        "<unk>",
        "<bos>",
        "<eos>",
        "<pad>",
        "k",
        "i",
        "n",
        "g",
        "q",
        "u",
        "e",
        "Ġ",
        "ki",
        "kin",
        "king",
        "qu",
        "que",
        "quee",
        "queen",
        "Ġking",
        "Ġqueen",
    ]
    merges = [
        ("k", "i"),
        ("ki", "n"),
        ("kin", "g"),
        ("q", "u"),
        ("qu", "e"),
        ("que", "e"),
        ("quee", "n"),
        ("Ġ", "king"),
        ("Ġ", "queen"),
    ]
    vocab = {piece: index for index, piece in enumerate(pieces)}
    backend = Tokenizer(BPE(vocab, merges, unk_token="<unk>"))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
    backend.decoder = ByteLevelDecoder()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
    )
    config = GPT2Config(vocab_size=len(vocab), n_embd=4, n_layer=1, n_head=2, n_positions=16)
    bridge, _ = make_tiny_pair(config, "GPT2LMHeadModel", loader=GPT2LMHeadModel)
    bridge.tokenizer = tokenizer
    geometry = RepresentationGeometry.from_bridge(bridge)
    plain = geometry.concept_direction([("king", "queen")])
    spaced = geometry.concept_direction([(" king", " queen")])
    assert plain.pairs == ((vocab["king"], vocab["queen"]),)
    assert spaced.pairs == ((vocab["Ġking"], vocab["Ġqueen"]),)
    assert spaced.pair_labels == ((" king", " queen"),)
    assert spaced.pairs != plain.pairs
    mixed = geometry.concept_direction([(" king", vocab["Ġqueen"])])
    assert mixed.pair_labels == ((" king", " queen"),)


def test_non_generation_adapter_is_rejected(tiny_bridge):
    tiny_bridge.adapter.supports_generation = False
    with pytest.raises(ValueError, match="decoder"):
        RepresentationGeometry.from_bridge(tiny_bridge)


def test_adapter_requires_a_transformer_bridge():
    with pytest.raises((TypeError,) + TYPECHECK_ERRORS):
        RepresentationGeometry.from_bridge(object())


def test_bridge_fit_options_are_forwarded(tiny_bridge):
    ids = list(range(8))
    geometry = RepresentationGeometry.from_bridge(
        tiny_bridge, token_ids=ids, ridge=0.01, rtol=1e-8, compute_dtype=torch.float64
    )
    reference = RepresentationGeometry(
        tiny_bridge.W_U, token_ids=ids, ridge=0.01, rtol=1e-8, compute_dtype=torch.float64
    )
    assert geometry.diagnostics == reference.diagnostics
    torch.testing.assert_close(geometry.regularized_covariance, reference.regularized_covariance)
    report = geometry.categorical_geometry(
        torch.stack(
            [
                geometry.concept_direction([(4, 5)]).raw_measurement,
                geometry.concept_direction([(6, 7)]).raw_measurement,
            ]
        ),
        space="measurement",
        labels=["gender-a", "gender-b"],
    )
    assert report.is_simplex()
    assert report.geometry_basis == geometry.basis
