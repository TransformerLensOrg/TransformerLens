"""LN folding and writing-weight centering must stay off for post-norm decoders.

Both transforms assume the norm gain sits on a sublayer's INPUT. OLMo 2/3 apply
ln1/ln2 to the sublayer OUTPUT, so folding is the wrong algebra: on the real
allenai/Olmo-3-1025-7B it moved log-softmax by 19.73 and dropped argmax agreement
with HF to 0%. The guard existed but named only OLMo 2, so OLMo 3 folded silently.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from transformer_lens.loading_from_pretrained import get_pretrained_model_config
from transformer_lens.utilities.architectures import POST_NORM_ARCHITECTURES
from transformer_lens.weight_processing import ProcessWeights


def _tl_config(architecture: str):
    from transformer_lens import HookedTransformerConfig

    return HookedTransformerConfig(
        n_layers=0,
        d_model=8,
        n_ctx=16,
        d_head=4,
        n_heads=2,
        d_vocab=10,
        act_fn="silu",
        normalization_type="RMS",
        original_architecture=architecture,
        positional_embedding_type="rotary",
    )


POST_NORM_MODELS = [
    ("allenai/Olmo-3-1025-7B", "Olmo3ForCausalLM"),
    ("allenai/OLMo-2-0425-1B", "Olmo2ForCausalLM"),
]


def _olmo_hf_config(architecture: str) -> SimpleNamespace:
    return SimpleNamespace(
        architectures=[architecture],
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        num_hidden_layers=4,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        vocab_size=100,
        hidden_act="silu",
        rope_theta=500000.0,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        sliding_window=4096,
        initializer_range=0.02,
        tie_word_embeddings=False,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 500000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 500000.0},
        },
    )


@pytest.mark.parametrize("model_name,architecture", POST_NORM_MODELS)
@mock.patch("transformer_lens.loading_from_pretrained.AutoConfig")
def test_fold_ln_is_refused(mock_auto_config, caplog, model_name, architecture) -> None:
    mock_auto_config.from_pretrained.return_value = _olmo_hf_config(architecture)
    with caplog.at_level("WARNING"):
        get_pretrained_model_config(model_name, fold_ln=True)
    assert any(
        "fold_ln=True is incompatible" in record.getMessage() for record in caplog.records
    ), [r.getMessage() for r in caplog.records]


@mock.patch("transformer_lens.loading_from_pretrained.AutoConfig")
def test_pre_norm_architecture_still_folds(mock_auto_config, caplog) -> None:
    """Negative control: the guard must not disable folding for everyone."""
    mock_auto_config.from_pretrained.return_value = SimpleNamespace(
        architectures=["LlamaForCausalLM"],
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        num_hidden_layers=2,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        vocab_size=100,
        hidden_act="silu",
        rope_theta=10000.0,
    )
    with caplog.at_level("WARNING"):
        get_pretrained_model_config("01-ai/Yi-6B", fold_ln=True)
    assert not any("fold_ln=True is incompatible" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("architecture", sorted(POST_NORM_ARCHITECTURES))
def test_embeddings_are_not_centered(architecture) -> None:
    """The first attention's input is un-normed, so centering W_E shifts a residual
    stream nothing re-normalizes."""
    torch.manual_seed(0)
    embedding = torch.randn(10, 8)
    state = {"embed.W_E": embedding.clone()}
    cfg = _tl_config(architecture)
    out = ProcessWeights.center_writing_weights(state, cfg)
    torch.testing.assert_close(out["embed.W_E"], embedding)


def test_embeddings_are_centered_for_pre_norm() -> None:
    """Engagement check: centering is a no-op above only because of the guard."""
    torch.manual_seed(0)
    embedding = torch.randn(10, 8)
    state = {"embed.W_E": embedding.clone()}
    cfg = _tl_config("LlamaForCausalLM")
    out = ProcessWeights.center_writing_weights(state, cfg)
    assert not torch.allclose(out["embed.W_E"], embedding)
    torch.testing.assert_close(out["embed.W_E"].mean(-1), torch.zeros(10), atol=1e-6, rtol=0)
