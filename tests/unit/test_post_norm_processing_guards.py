"""Post-norm decoders must not have their embedding writing-weights centered.

Centering assumes the first attention's input is normalized; OLMo 2/3 apply
ln1/ln2 to the sublayer OUTPUT, leaving the residual stream un-normed there, so
centering shifts a stream nothing re-normalizes. On the real allenai/Olmo-3-1025-7B
this moved log-softmax by 19.73 and dropped argmax agreement with HF to 0%. The
guard once named only OLMo 2, so OLMo 3 was silently mis-centered.

Re-anchored on ``ProcessWeights`` directly (the surviving weight-processing path)
after the legacy ``get_pretrained_model_config`` wrapper was removed at 4.0; the
``POST_NORM_ARCHITECTURES`` membership it keys on is unchanged.
"""

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.utilities.architectures import POST_NORM_ARCHITECTURES
from transformer_lens.weight_processing import ProcessWeights

POST_NORM_MODELS = [
    "Olmo3ForCausalLM",
    "Olmo2ForCausalLM",
]


def _cfg(architecture: str) -> TransformerBridgeConfig:
    cfg = TransformerBridgeConfig(
        n_layers=2,
        d_model=8,
        n_ctx=16,
        d_head=4,
        n_heads=2,
        d_vocab=10,
        act_fn="silu",
        normalization_type="RMS",
        positional_embedding_type="rotary",
    )
    cfg.original_architecture = architecture
    return cfg


@pytest.mark.parametrize("architecture", POST_NORM_MODELS)
def test_post_norm_architectures_are_registered(architecture):
    """Both OLMo generations are in the guard set — OLMo 3 being absent is the
    exact regression this file guards against."""
    assert architecture in POST_NORM_ARCHITECTURES


@pytest.mark.parametrize("architecture", POST_NORM_MODELS)
def test_center_writing_weights_skips_post_norm(architecture, capsys):
    """center_writing_weights leaves a post-norm model's embedding untouched."""
    import torch

    cfg = _cfg(architecture)
    W_E = torch.randn(cfg.d_vocab, cfg.d_model)
    state_dict = {"embed.W_E": W_E.clone()}

    result = ProcessWeights.center_writing_weights(state_dict, cfg, adapter=None)

    assert torch.equal(result["embed.W_E"], W_E), "post-norm embedding was centered"
    assert f"Not centering embedding weights for {architecture}" in capsys.readouterr().out


def test_pre_norm_architecture_is_still_centered(capsys):
    """Negative control: the guard must not disable centering for everyone."""
    import torch

    cfg = _cfg("LlamaForCausalLM")
    assert "LlamaForCausalLM" not in POST_NORM_ARCHITECTURES
    W_E = torch.randn(cfg.d_vocab, cfg.d_model)
    state_dict = {"embed.W_E": W_E.clone()}

    result = ProcessWeights.center_writing_weights(state_dict, cfg, adapter=None)

    assert not torch.equal(result["embed.W_E"], W_E), "pre-norm embedding was not centered"
