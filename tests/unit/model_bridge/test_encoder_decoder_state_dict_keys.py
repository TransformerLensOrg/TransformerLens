"""Encoder-decoder state_dict() keeps every LayerNorm of both stacks.

BART-family blocks give one bridge name different HF modules per stack: ``ln2`` is
the encoder's ``final_layer_norm`` but the decoder's ``encoder_attn_layer_norm``.
Resolving the name from one map shared by both stacks, in set order, either renamed
the encoder norm to the decoder's name or collided the decoder's ``ln2`` with its
``ln3`` and dropped it, depending on the process's hash seed.
"""
import pytest
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq,
    MBartConfig,
    WhisperConfig,
)

from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
)

COMMON = dict(
    d_model=16,
    encoder_layers=2,
    decoder_layers=2,
    encoder_attention_heads=2,
    decoder_attention_heads=2,
    encoder_ffn_dim=32,
    decoder_ffn_dim=32,
    pad_token_id=1,
    bos_token_id=0,
    eos_token_id=2,
    decoder_start_token_id=2,
)


class _Tok:
    pass


def _bridge(arch, config, auto_class):
    config.architectures = [arch]
    hf_model = auto_class.from_config(config, attn_implementation="eager").eval()
    cfg = build_bridge_config_from_hf(hf_model.config, arch, "tiny", torch.float32)
    return TransformerBridge(hf_model, SUPPORTED_ARCHITECTURES[arch](cfg), tokenizer=_Tok())


@pytest.fixture(
    params=["MBartForConditionalGeneration", "WhisperForConditionalGeneration"],
    scope="module",
)
def bridge(request):
    if request.param == "MBartForConditionalGeneration":
        config = MBartConfig(vocab_size=64, max_position_embeddings=32, **COMMON)
        return _bridge(request.param, config, AutoModelForSeq2SeqLM)
    config = WhisperConfig(
        vocab_size=64, num_mel_bins=8, max_source_positions=10, max_target_positions=16, **COMMON
    )
    return _bridge(request.param, config, AutoModelForSpeechSeq2Seq)


def test_every_layer_norm_has_its_own_key(bridge):
    state_dict = bridge.state_dict()
    hf_layers = {
        "encoder_blocks": bridge.original_model.model.encoder.layers,
        "decoder_blocks": bridge.original_model.model.decoder.layers,
    }
    for stack, layers in hf_layers.items():
        norms = ["self_attn_layer_norm", "final_layer_norm"]
        if stack == "decoder_blocks":
            norms.append("encoder_attn_layer_norm")
        for i in range(len(layers)):
            for norm in norms:
                hf_norm = getattr(bridge, stack)[i]
                expected = {
                    "self_attn_layer_norm": hf_norm.ln1,
                    "encoder_attn_layer_norm": getattr(hf_norm, "ln2"),
                    "final_layer_norm": hf_norm.ln3 if stack == "decoder_blocks" else hf_norm.ln2,
                }[norm].original_component
                for param in ("weight", "bias"):
                    key = f"{stack}.{i}.{norm}.{param}"
                    assert key in state_dict, key
                    assert state_dict[key].data_ptr() == getattr(expected, param).data_ptr(), key


def test_encoder_has_no_cross_attention_norm(bridge):
    assert not [
        k for k in bridge.state_dict() if k.startswith("encoder_blocks.") and "encoder_attn" in k
    ]
