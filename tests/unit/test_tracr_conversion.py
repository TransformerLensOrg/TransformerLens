from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from transformer_lens.utilities.tracr import (
    infer_tracr_output_label,
    make_tracr_categorical_unembed,
    make_tracr_transformer_bridge_config,
    make_tracr_transformer_bridge_state_dict,
)


class _CategoricalEncoder:
    def __init__(self, encoding_map):
        self.encoding_map = encoding_map


def _fake_tracr_model() -> SimpleNamespace:
    d_model = 8
    d_vocab = 5
    d_vocab_out = 3
    n_ctx = 4
    d_head = 2
    n_heads = 2
    d_mlp = 6
    n_layers = 1

    params = {
        "token_embed": {"embeddings": np.arange(d_vocab * d_model).reshape(d_vocab, d_model)},
        "pos_embed": {"embeddings": np.arange(n_ctx * d_model).reshape(n_ctx, d_model)},
    }
    for layer in range(n_layers):
        prefix = f"transformer/layer_{layer}"
        params.update(
            {
                f"{prefix}/attn/key": {
                    "w": np.arange(d_model * n_heads * d_head).reshape(d_model, n_heads * d_head),
                    "b": np.arange(n_heads * d_head),
                },
                f"{prefix}/attn/query": {
                    "w": np.arange(d_model * n_heads * d_head).reshape(d_model, n_heads * d_head)
                    + 100,
                    "b": np.arange(n_heads * d_head) + 100,
                },
                f"{prefix}/attn/value": {
                    "w": np.arange(d_model * n_heads * d_head).reshape(d_model, n_heads * d_head)
                    + 200,
                    "b": np.arange(n_heads * d_head) + 200,
                },
                f"{prefix}/attn/linear": {
                    "w": np.arange(n_heads * d_head * d_model).reshape(n_heads * d_head, d_model)
                    + 300,
                    "b": np.arange(d_model) + 300,
                },
                f"{prefix}/mlp/linear_1": {
                    "w": np.arange(d_model * d_mlp).reshape(d_model, d_mlp) + 400,
                    "b": np.arange(d_mlp) + 400,
                },
                f"{prefix}/mlp/linear_2": {
                    "w": np.arange(d_mlp * d_model).reshape(d_mlp, d_model) + 500,
                    "b": np.arange(d_model) + 500,
                },
            }
        )

    return SimpleNamespace(
        params=params,
        residual_labels=[
            "aggregate_1:1",
            "aggregate_1:2",
            "aggregate_1:3",
            "indices:0",
            "reverse:1",
            "reverse:2",
            "reverse:3",
            "tokens:BOS",
        ],
        output_encoder=_CategoricalEncoder({1: 0, 2: 1, 3: 2}),
        model_config=SimpleNamespace(
            num_heads=n_heads,
            num_layers=n_layers,
            key_size=d_head,
            mlp_hidden_size=d_mlp,
            layer_norm=False,
            causal=False,
        ),
    )


def test_categorical_unembed_uses_named_output_basis():
    model = _fake_tracr_model()

    unembed = make_tracr_categorical_unembed(model, output_label="reverse")

    assert unembed.shape == (8, 3)
    np.testing.assert_array_equal(unembed[:4], np.zeros((4, 3)))
    np.testing.assert_array_equal(unembed[4:7], np.eye(3))


def test_infer_output_label_rejects_ambiguous_prefixes():
    model = _fake_tracr_model()

    with pytest.raises(ValueError, match="Pass output_label explicitly"):
        infer_tracr_output_label(model)


def test_bridge_config_matches_tracr_metadata():
    cfg = make_tracr_transformer_bridge_config(_fake_tracr_model())

    assert cfg.d_model == 8
    assert cfg.d_vocab == 5
    assert cfg.d_vocab_out == 3
    assert cfg.d_head == 2
    assert cfg.n_heads == 2
    assert cfg.d_mlp == 6
    assert cfg.normalization_type is None
    assert cfg.attention_dir == "bidirectional"


def test_bridge_state_dict_transposes_tracr_weights_and_reconstructs_unembed():
    model = _fake_tracr_model()

    state_dict = make_tracr_transformer_bridge_state_dict(
        model, output_label="reverse", dtype=torch.float64
    )

    assert state_dict["tok_embed.weight"].shape == (5, 8)
    assert state_dict["pos.weight"].shape == (4, 8)
    assert state_dict["head.weight"].shape == (3, 8)
    assert state_dict["layers.0.attn.q.weight"].shape == (4, 8)
    assert state_dict["layers.0.mlp.fc_in.weight"].shape == (6, 8)
    assert state_dict["layers.0.mlp.fc_out.weight"].shape == (8, 6)
    assert state_dict["layers.0.attn.q.weight"].dtype == torch.float64

    np.testing.assert_array_equal(
        state_dict["layers.0.attn.q.weight"].numpy(),
        model.params["transformer/layer_0/attn/query"]["w"].T,
    )
    np.testing.assert_array_equal(
        state_dict["layers.0.mlp.fc_out.weight"].numpy(),
        model.params["transformer/layer_0/mlp/linear_2"]["w"].T,
    )
    np.testing.assert_array_equal(
        state_dict["head.weight"].numpy(),
        np.array(
            [
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ]
        ),
    )


def test_bridge_state_dict_rejects_layer_norm_tracr_models():
    model = _fake_tracr_model()
    model.model_config.layer_norm = True

    with pytest.raises(NotImplementedError, match="layer_norm=True"):
        make_tracr_transformer_bridge_state_dict(model, output_label="reverse")
