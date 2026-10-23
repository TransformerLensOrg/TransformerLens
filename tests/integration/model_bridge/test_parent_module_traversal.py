"""Download-free parent traversal coverage across Bridge architecture shapes."""

from dataclasses import dataclass
from typing import Any, Callable

import pytest
import torch
from torch import nn
from transformers import (
    ASTConfig,
    ASTForAudioClassification,
    BartConfig,
    BartForConditionalGeneration,
    BertConfig,
    BertForMaskedLM,
    BertForNextSentencePrediction,
    BertForPreTraining,
    BloomConfig,
    BloomForCausalLM,
    Gemma4AudioConfig,
    Gemma4Config,
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
    Gemma4TextConfig,
    Gemma4VisionConfig,
    GPT2Config,
    GPT2LMHeadModel,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    HubertConfig,
    HubertForCTC,
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    MixtralConfig,
    MixtralForCausalLM,
    T5Config,
    T5ForConditionalGeneration,
    ViTConfig,
    ViTForImageClassification,
    ViTModel,
)

from tests.integration.model_bridge.helpers import make_tiny_pair
from transformer_lens.model_bridge.sources import build_bridge_from_module


@dataclass(frozen=True)
class ArchitectureCase:
    name: str
    model_type: type[nn.Module]
    config_factory: Callable[[], Any]
    architecture: str


def _bert_config() -> BertConfig:
    return BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
    )


def _vit_config() -> ViTConfig:
    return ViTConfig(
        image_size=16,
        patch_size=4,
        num_channels=3,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        num_labels=3,
    )


def _bart_config() -> BartConfig:
    return BartConfig(
        vocab_size=32,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_position_embeddings=16,
    )


def _hubert_config() -> HubertConfig:
    return HubertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        conv_dim=(8,),
        conv_stride=(2,),
        conv_kernel=(3,),
        num_conv_pos_embeddings=4,
        num_conv_pos_embedding_groups=2,
    )


def _gemma4_text_config() -> Gemma4TextConfig:
    return Gemma4TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=16,
        vocab_size_per_layer_input=32,
        hidden_size_per_layer_input=4,
    )


def _gemma4_config() -> Gemma4Config:
    return Gemma4Config(
        text_config=_gemma4_text_config(),
        vision_config=Gemma4VisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=4,
        ),
        audio_config=Gemma4AudioConfig(
            hidden_size=16,
            num_attention_heads=4,
            num_hidden_layers=1,
            output_proj_dims=16,
            subsampling_conv_channels=[8, 8],
        ),
    )


ARCHITECTURE_CASES = (
    ArchitectureCase(
        "gpt2-joint-qkv",
        GPT2LMHeadModel,
        lambda: GPT2Config(
            vocab_size=32,
            n_positions=16,
            n_ctx=16,
            n_embd=16,
            n_layer=1,
            n_head=4,
            n_inner=32,
        ),
        "GPT2LMHeadModel",
    ),
    ArchitectureCase(
        "bloom-joint-qkv",
        BloomForCausalLM,
        lambda: BloomConfig(vocab_size=32, hidden_size=16, n_layer=1, n_head=4),
        "BloomForCausalLM",
    ),
    ArchitectureCase(
        "gpt-neox-rotary",
        GPTNeoXForCausalLM,
        lambda: GPTNeoXConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            max_position_embeddings=16,
        ),
        "GPTNeoXForCausalLM",
    ),
    ArchitectureCase(
        "llama-split-qkv-rope",
        LlamaForCausalLM,
        lambda: LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=16,
        ),
        "LlamaForCausalLM",
    ),
    ArchitectureCase(
        "mistral-gqa",
        MistralForCausalLM,
        lambda: MistralConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=16,
        ),
        "MistralForCausalLM",
    ),
    ArchitectureCase("bert-mlm", BertForMaskedLM, _bert_config, "BertForMaskedLM"),
    ArchitectureCase(
        "bert-nsp",
        BertForNextSentencePrediction,
        _bert_config,
        "BertForMaskedLM",
    ),
    ArchitectureCase(
        "bert-mlm-nsp",
        BertForPreTraining,
        _bert_config,
        "BertForMaskedLM",
    ),
    ArchitectureCase(
        "t5-encoder-decoder",
        T5ForConditionalGeneration,
        lambda: T5Config(
            vocab_size=32,
            d_model=16,
            d_kv=4,
            d_ff=32,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=4,
        ),
        "T5ForConditionalGeneration",
    ),
    ArchitectureCase(
        "bart-encoder-decoder",
        BartForConditionalGeneration,
        _bart_config,
        "BartForConditionalGeneration",
    ),
    ArchitectureCase(
        "mixtral-moe",
        MixtralForCausalLM,
        lambda: MixtralConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_local_experts=2,
            num_experts_per_tok=1,
            max_position_embeddings=16,
        ),
        "MixtralForCausalLM",
    ),
    ArchitectureCase(
        "vit-vision",
        ViTForImageClassification,
        _vit_config,
        "ViTForImageClassification",
    ),
    ArchitectureCase("vit-bare-pooler", ViTModel, _vit_config, "ViTModel"),
    ArchitectureCase(
        "hubert-audio",
        HubertForCTC,
        _hubert_config,
        "HubertForCTC",
    ),
    ArchitectureCase(
        "ast-audio-classifier",
        ASTForAudioClassification,
        lambda: ASTConfig(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=32,
            patch_size=4,
            frequency_stride=4,
            time_stride=4,
            max_length=16,
            num_mel_bins=16,
        ),
        "ASTForAudioClassification",
    ),
    ArchitectureCase(
        "gemma4-per-layer-embeddings",
        Gemma4ForCausalLM,
        _gemma4_text_config,
        "Gemma4ForCausalLM",
    ),
    ArchitectureCase(
        "gemma4-audio-vision",
        Gemma4ForConditionalGeneration,
        _gemma4_config,
        "Gemma4ForConditionalGeneration",
    ),
)

ARCHITECTURE_CASE_BY_NAME = {case.name: case for case in ARCHITECTURE_CASES}


def _named_identities(named_values: Any) -> dict[int, str]:
    return {id(value): name for name, value in named_values}


def _assert_same_identities(expected: dict[int, str], actual: dict[int, str]) -> None:
    missing = [expected[identity] for identity in expected.keys() - actual.keys()]
    unexpected = [actual[identity] for identity in actual.keys() - expected.keys()]
    assert actual.keys() == expected.keys(), f"missing={missing}, unexpected={unexpected}"


@pytest.mark.parametrize("case", ARCHITECTURE_CASES, ids=lambda case: case.name)
def test_parent_and_direct_traversal_have_identical_state(case: ArchitectureCase) -> None:
    config = case.config_factory()
    model = case.model_type(config).eval()
    bridge = build_bridge_from_module(
        model,
        case.architecture,
        hf_config=config,
        dtype=torch.float32,
        device="cpu",
        model_name=f"tiny-{case.name}",
    )
    parent = nn.Module()
    parent.add_module("bridge", bridge)

    source_parameters = _named_identities(bridge.original_model.named_parameters())
    direct_parameters = _named_identities(bridge.named_parameters())
    parent_parameters = _named_identities(parent.named_parameters())
    source_buffers = _named_identities(bridge.original_model.named_buffers())
    direct_buffers = _named_identities(bridge.named_buffers())
    parent_buffers = _named_identities(parent.named_buffers())

    _assert_same_identities(source_parameters, direct_parameters)
    _assert_same_identities(direct_parameters, parent_parameters)
    _assert_same_identities(source_buffers, direct_buffers)
    _assert_same_identities(direct_buffers, parent_buffers)


def test_parent_dtype_conversion_updates_container_owned_state() -> None:
    bart_config = _bart_config()
    bridge = build_bridge_from_module(
        BartForConditionalGeneration(bart_config),
        "BartForConditionalGeneration",
        hf_config=bart_config,
        dtype=torch.float32,
        device="cpu",
        model_name="tiny-bart-container-buffer",
    )
    parent = nn.Module()
    parent.add_module("bridge", bridge)

    parent.to(torch.float64)

    assert bridge.original_model.final_logits_bias.dtype == torch.float64
    assert id(bridge.original_model.final_logits_bias) in {
        id(buffer) for buffer in parent.buffers()
    }


@pytest.mark.parametrize("through_parent", (False, True), ids=("bridge-half", "parent-to"))
def test_dtype_conversion_updates_unmapped_leaf_parameters(through_parent: bool) -> None:
    bridge, reference = make_tiny_pair(
        _gemma4_text_config(), "Gemma4ForCausalLM", loader=Gemma4ForCausalLM
    )
    if through_parent:
        parent = nn.Module()
        parent.add_module("bridge", bridge)
        parent.to(torch.float16)
    else:
        bridge.half()
    reference.half()
    tokens = torch.tensor([[2, 5, 7, 9, 11]])

    with torch.no_grad():
        logits = bridge(tokens)
        expected = reference(input_ids=tokens).logits

    assert logits.dtype == torch.float16
    assert torch.equal(logits, expected)


def test_parent_assign_load_updates_container_owned_state() -> None:
    bart_config = _bart_config()
    bridge = build_bridge_from_module(
        BartForConditionalGeneration(bart_config),
        "BartForConditionalGeneration",
        hf_config=bart_config,
        dtype=torch.float32,
        device="cpu",
        model_name="tiny-bart-container-buffer-load",
    )
    parent = nn.Module()
    parent.add_module("bridge", bridge)
    state = parent.state_dict()
    buffer_key = "bridge._container_state_owners.final_logits_bias"
    state[buffer_key] = torch.ones_like(state[buffer_key])

    parent.load_state_dict(state, strict=True, assign=True)

    assert torch.equal(bridge.original_model.final_logits_bias, torch.ones_like(state[buffer_key]))
    assert id(bridge.original_model.final_logits_bias) in {
        id(buffer) for buffer in parent.buffers()
    }


@pytest.mark.parametrize(
    ("case_name", "container_path", "state_name", "state_key"),
    (
        ("bart-encoder-decoder", "", "final_logits_bias", "final_logits_bias"),
        ("hubert-audio", "hubert", "masked_spec_embed", "hubert.masked_spec_embed"),
    ),
)
def test_direct_assign_load_stays_current_after_apply(
    case_name: str, container_path: str, state_name: str, state_key: str
) -> None:
    case = ARCHITECTURE_CASE_BY_NAME[case_name]
    config = case.config_factory()
    bridge = build_bridge_from_module(
        case.model_type(config),
        case.architecture,
        hf_config=config,
        dtype=torch.float32,
        device="cpu",
        model_name=f"tiny-{case.name}-direct-assign",
    )
    original_container = (
        bridge.original_model.get_submodule(container_path)
        if container_path
        else bridge.original_model
    )
    owner_container = (
        bridge._container_state_owners.get_submodule(container_path)
        if container_path
        else bridge._container_state_owners
    )
    replacement = torch.full_like(getattr(original_container, state_name), 7)

    bridge.load_state_dict({state_key: replacement}, strict=False, assign=True)

    assert getattr(owner_container, state_name) is getattr(original_container, state_name)
    bridge.cpu()
    assert torch.equal(getattr(original_container, state_name), replacement)


@pytest.mark.parametrize(
    ("case_name", "key_fragment", "expected_keys"),
    (
        # BERT wraps the pooler module (not its inner Linear) so hook_out carries
        # the post-tanh pooled [CLS], which nests the weights one level deeper.
        # Still two keys, so nothing is re-expanded — only renamed. ViT wraps
        # pooler.dense and keeps the flat names.
        ("bert-nsp", "pooler", {"pooler.dense.weight", "pooler.dense.bias"}),
        ("vit-bare-pooler", "pooler", {"pooler.weight", "pooler.bias"}),
        (
            "ast-audio-classifier",
            "classifier",
            {"classifier_ln.weight", "classifier_ln.bias"},
        ),
    ),
)
def test_task_head_state_dict_keys_are_not_reexpanded(
    case_name: str, key_fragment: str, expected_keys: set[str]
) -> None:
    case = ARCHITECTURE_CASE_BY_NAME[case_name]
    config = case.config_factory()
    bridge = build_bridge_from_module(
        case.model_type(config),
        case.architecture,
        hf_config=config,
        dtype=torch.float32,
        device="cpu",
        model_name=f"tiny-{case.name}-state-dict-keys",
    )

    actual_keys = {key for key in bridge.state_dict() if key_fragment in key}
    assert actual_keys == expected_keys
