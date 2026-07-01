"""Utilities for loading Tracr-assembled models into TransformerBridge.

These helpers are intentionally duck-typed so importing TransformerLens does not
pull in Tracr, JAX, or Haiku. Pass in a Tracr ``AssembledTransformerModel`` from
``tracr.compiler.compiling.compile_rasp_to_model``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from transformer_lens.config.transformer_bridge_config import TransformerBridgeConfig


def infer_tracr_output_label(model: Any) -> str:
    """Infer the RASP output label used in ``model.residual_labels``.

    Tracr stores residual basis labels as strings like ``"reverse:3"`` while
    its categorical output encoder stores only values and output-column ids.
    The output label is the unique residual-label prefix whose value set exactly
    matches the output encoder's categorical value set.
    """
    encoding_map = _categorical_output_encoding_map(model)
    output_values = {str(value) for value in encoding_map}
    values_by_label: dict[str, set[str]] = {}

    for residual_label in _residual_labels(model):
        label, separator, value = residual_label.rpartition(":")
        if not separator:
            continue
        values_by_label.setdefault(label, set()).add(value)

    candidates = [label for label, values in values_by_label.items() if values == output_values]
    if len(candidates) != 1:
        raise ValueError(
            "Could not infer Tracr output label from residual labels. Pass "
            f"output_label explicitly. Candidates: {candidates!r}."
        )
    return candidates[0]


def make_tracr_categorical_unembed(
    model: Any,
    output_label: str | None = None,
    *,
    dtype: np.dtype = np.dtype("float32"),
) -> np.ndarray:
    """Return Tracr's categorical unembed matrix in TL format.

    The returned matrix has shape ``[d_model, d_vocab_out]`` and projects the
    residual stream onto the output-basis coordinates selected by Tracr's
    categorical output encoder. This avoids assuming those coordinates are the
    first residual dimensions.
    """
    encoding_map = _categorical_output_encoding_map(model)
    output_label = output_label if output_label is not None else infer_tracr_output_label(model)
    label_to_residual_index = {label: index for index, label in enumerate(_residual_labels(model))}
    unembed = np.zeros((len(label_to_residual_index), len(encoding_map)), dtype=dtype)

    for output_value, output_index in encoding_map.items():
        residual_label = f"{output_label}:{output_value}"
        if residual_label not in label_to_residual_index:
            raise ValueError(
                f"Could not find output basis label {residual_label!r} in "
                "model.residual_labels. Pass the final RASP expression label "
                "as output_label."
            )
        unembed[label_to_residual_index[residual_label], int(output_index)] = 1

    return unembed


def make_tracr_transformer_bridge_config(model: Any) -> TransformerBridgeConfig:
    """Build a ``TransformerBridgeConfig`` matching a categorical Tracr model."""
    model_config = model.model_config
    d_model = _param_array(model, "token_embed", "embeddings").shape[1]
    d_vocab = _param_array(model, "token_embed", "embeddings").shape[0]
    n_ctx = _param_array(model, "pos_embed", "embeddings").shape[0]

    return TransformerBridgeConfig(
        n_layers=model_config.num_layers,
        d_model=d_model,
        d_head=model_config.key_size,
        n_ctx=n_ctx,
        d_vocab=d_vocab,
        d_vocab_out=len(_categorical_output_encoding_map(model)),
        d_mlp=model_config.mlp_hidden_size,
        n_heads=model_config.num_heads,
        act_fn="relu",
        attention_dir="causal" if model_config.causal else "bidirectional",
        normalization_type="LN" if model_config.layer_norm else None,
    )


def make_tracr_transformer_bridge_state_dict(
    model: Any,
    output_label: str | None = None,
    *,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Build a ``TransformerBridge.boot_native`` state dict from Tracr weights.

    The state-dict keys use the native PyTorch module names accepted by
    ``TransformerBridge.load_state_dict`` after ``boot_native``.
    """
    if model.model_config.layer_norm:
        raise NotImplementedError(
            "Tracr layer_norm=True models are not supported by this converter yet."
        )

    n_layers = model.model_config.num_layers
    state_dict: dict[str, torch.Tensor] = {
        "tok_embed.weight": _tensor(model, "token_embed", "embeddings", dtype=dtype),
        "pos.weight": _tensor(model, "pos_embed", "embeddings", dtype=dtype),
        "head.weight": torch.tensor(
            make_tracr_categorical_unembed(model, output_label).T,
            dtype=dtype,
        ),
    }

    for layer in range(n_layers):
        prefix = f"transformer/layer_{layer}"
        state_dict.update(
            {
                f"layers.{layer}.attn.k.weight": _tensor(
                    model, f"{prefix}/attn/key", "w", dtype=dtype
                ).T,
                f"layers.{layer}.attn.k.bias": _tensor(
                    model, f"{prefix}/attn/key", "b", dtype=dtype
                ),
                f"layers.{layer}.attn.q.weight": _tensor(
                    model, f"{prefix}/attn/query", "w", dtype=dtype
                ).T,
                f"layers.{layer}.attn.q.bias": _tensor(
                    model, f"{prefix}/attn/query", "b", dtype=dtype
                ),
                f"layers.{layer}.attn.v.weight": _tensor(
                    model, f"{prefix}/attn/value", "w", dtype=dtype
                ).T,
                f"layers.{layer}.attn.v.bias": _tensor(
                    model, f"{prefix}/attn/value", "b", dtype=dtype
                ),
                f"layers.{layer}.attn.o.weight": _tensor(
                    model, f"{prefix}/attn/linear", "w", dtype=dtype
                ).T,
                f"layers.{layer}.attn.o.bias": _tensor(
                    model, f"{prefix}/attn/linear", "b", dtype=dtype
                ),
                f"layers.{layer}.mlp.fc_in.weight": _tensor(
                    model, f"{prefix}/mlp/linear_1", "w", dtype=dtype
                ).T,
                f"layers.{layer}.mlp.fc_in.bias": _tensor(
                    model, f"{prefix}/mlp/linear_1", "b", dtype=dtype
                ),
                f"layers.{layer}.mlp.fc_out.weight": _tensor(
                    model, f"{prefix}/mlp/linear_2", "w", dtype=dtype
                ).T,
                f"layers.{layer}.mlp.fc_out.bias": _tensor(
                    model, f"{prefix}/mlp/linear_2", "b", dtype=dtype
                ),
            }
        )

    return state_dict


def _categorical_output_encoding_map(model: Any) -> Mapping[Any, int]:
    output_encoder = getattr(model, "output_encoder", None)
    encoding_map = getattr(output_encoder, "encoding_map", None)
    if not isinstance(encoding_map, Mapping):
        raise NotImplementedError(
            "Only categorical Tracr outputs are supported; expected "
            "model.output_encoder.encoding_map."
        )
    return encoding_map


def _residual_labels(model: Any) -> list[str]:
    residual_labels = getattr(model, "residual_labels", None)
    if residual_labels is None:
        raise ValueError("Expected Tracr model to expose residual_labels.")
    return list(residual_labels)


def _param_array(model: Any, module_name: str, param_name: str) -> np.ndarray:
    return np.asarray(model.params[module_name][param_name])


def _tensor(
    model: Any,
    module_name: str,
    param_name: str,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.tensor(_param_array(model, module_name, param_name), dtype=dtype)
