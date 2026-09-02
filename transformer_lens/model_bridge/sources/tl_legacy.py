"""Loader for legacy TransformerLens-format HF repos.

NeelNanda/*, ArthurConmy/*, and Baidicoot/* repos predate the HF model format:
no ``model_type`` in config.json (AutoConfig refuses them), weights stored as
``*.pth`` state dicts in HookedTransformer property format (or older layouts
converted below), and training checkpoints as ``checkpoints/*_<label>.pth``
files rather than revisions. This module derives a TransformerBridgeConfig from
the repo's TL-style config.json, fetches and normalizes the state dict, and
loads it into a ``boot_native`` bridge via ``convert_tl_checkpoint``.

The two legacy layout converters are ports of the HookedTransformer loaders
(pretrained/weight_conversions/{neel_solu_old,mingpt}.py), rehomed here so this
path survives the 4.0 deletion of the legacy loading stack.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Union

import einops
import torch

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.utilities.tl_checkpoint_conversion import convert_tl_checkpoint

TL_LEGACY_PREFIXES = ("NeelNanda/", "ArthurConmy/", "Baidicoot/")


def _fetch_json(repo_id: str, filename: str) -> dict:
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(repo_id, filename)) as f:
        return json.load(f)


def derive_tl_legacy_config(repo_id: str) -> TransformerBridgeConfig:
    """TransformerBridgeConfig from a legacy TL repo's config.json."""
    cfg_json = _fetch_json(repo_id, "config.json")
    architecture = cfg_json.get(
        "architecture", "neel" if "_old" not in repo_id else "neel-solu-old"
    )
    normalization = cfg_json.get("normalization", cfg_json.get("normalization_type"))
    if cfg_json.get("shortformer_pos", False):
        raise NotImplementedError(
            f"{repo_id} uses shortformer positional embeddings, which the native "
            "bridge does not implement."
        )
    cfg = TransformerBridgeConfig(
        d_model=cfg_json["d_model"],
        n_layers=cfg_json["n_layers"],
        d_mlp=cfg_json["d_mlp"],
        d_head=cfg_json["d_head"],
        n_heads=cfg_json["n_heads"],
        n_ctx=cfg_json["n_ctx"],
        d_vocab=cfg_json["d_vocab"],
        act_fn=cfg_json["act_fn"],
        attn_only=cfg_json["attn_only"],
        final_rms=cfg_json.get("final_rms", False),
        normalization_type=normalization,
        positional_embedding_type="standard",
        tokenizer_name=cfg_json.get("tokenizer_name"),
        architecture="TransformerLensNative",
    )
    cfg.original_architecture = architecture  # type: ignore[attr-defined]
    return cfg


def _convert_neel_solu_old_weights(state_dict: dict, cfg: TransformerBridgeConfig) -> dict:
    """Old-layout SoLU repos ('*_old'): left-facing weights below 8L, and 8L's
    W_pos alone left-facing. Port of the HookedTransformer converter."""
    reverse_pos = cfg.n_layers <= 8
    reverse_weights = cfg.n_layers <= 6
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("norm", "ln")
        if k.startswith("ln."):
            k = k.replace("ln.", "ln_final.")
        new_state_dict[k] = v
    if reverse_pos:
        new_state_dict["pos_embed.W_pos"] = new_state_dict["pos_embed.W_pos"].T
    if reverse_weights:
        for k, v in new_state_dict.items():
            if "W_" in k and "W_pos" not in k:
                new_state_dict[k] = v.transpose(-2, -1)
    return new_state_dict


def _convert_mingpt_weights(old_state_dict: dict, cfg: TransformerBridgeConfig) -> dict:
    """minGPT layout (Baidicoot/Othello-GPT): unconcatenated QKV. Port of the
    HookedTransformer converter."""
    state_dict = {
        "embed.W_E": old_state_dict["tok_emb.weight"],
        "pos_embed.W_pos": old_state_dict["pos_emb"].squeeze(),
    }
    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = old_state_dict[f"blocks.{l}.ln1.weight"]
        state_dict[f"blocks.{l}.ln1.b"] = old_state_dict[f"blocks.{l}.ln1.bias"]
        for name, hf in (("Q", "query"), ("K", "key"), ("V", "value")):
            w = einops.rearrange(
                old_state_dict[f"blocks.{l}.attn.{hf}.weight"], "(i h) m->i m h", i=cfg.n_heads
            )
            b = einops.rearrange(
                old_state_dict[f"blocks.{l}.attn.{hf}.bias"], "(i h)->i h", i=cfg.n_heads
            )
            state_dict[f"blocks.{l}.attn.W_{name}"] = w
            state_dict[f"blocks.{l}.attn.b_{name}"] = b
        state_dict[f"blocks.{l}.attn.W_O"] = einops.rearrange(
            old_state_dict[f"blocks.{l}.attn.proj.weight"], "m (i h)->i h m", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.b_O"] = old_state_dict[f"blocks.{l}.attn.proj.bias"]
        state_dict[f"blocks.{l}.ln2.w"] = old_state_dict[f"blocks.{l}.ln2.weight"]
        state_dict[f"blocks.{l}.ln2.b"] = old_state_dict[f"blocks.{l}.ln2.bias"]
        state_dict[f"blocks.{l}.mlp.W_in"] = old_state_dict[f"blocks.{l}.mlp.0.weight"].T
        state_dict[f"blocks.{l}.mlp.b_in"] = old_state_dict[f"blocks.{l}.mlp.0.bias"]
        state_dict[f"blocks.{l}.mlp.W_out"] = old_state_dict[f"blocks.{l}.mlp.2.weight"].T
        state_dict[f"blocks.{l}.mlp.b_out"] = old_state_dict[f"blocks.{l}.mlp.2.bias"]
    state_dict["ln_final.w"] = old_state_dict["ln_f.weight"]
    state_dict["ln_final.b"] = old_state_dict["ln_f.bias"]
    state_dict["unembed.W_U"] = old_state_dict["head.weight"].T
    return state_dict


def fetch_tl_legacy_state_dict(
    repo_id: str,
    cfg: TransformerBridgeConfig,
    checkpoint_value: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Download and normalize a legacy repo's state dict to TL property format."""
    from huggingface_hub import HfApi, hf_hub_download

    repo_files = HfApi().list_repo_files(repo_id)
    suffix = f"{checkpoint_value}.pth" if checkpoint_value is not None else "final.pth"
    matches = [f for f in repo_files if f.endswith(suffix)]
    if not matches:
        raise FileNotFoundError(f"No '*{suffix}' file in {repo_id}; files: {repo_files[:8]}...")
    state_dict = torch.load(
        hf_hub_download(repo_id, matches[0]), map_location="cpu", weights_only=True
    )
    state_dict = {k: v.to(dtype) for k, v in state_dict.items()}

    original_architecture = getattr(cfg, "original_architecture", None)
    if original_architecture == "neel-solu-old":
        state_dict = _convert_neel_solu_old_weights(state_dict, cfg)
    elif original_architecture == "mingpt":
        state_dict = _convert_mingpt_weights(state_dict, cfg)
    return state_dict


def boot(
    model_name: str,
    checkpoint_index: Optional[int] = None,
    checkpoint_value: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    tokenizer: Optional[Any] = None,
) -> TransformerBridge:
    """Build a bridge for a legacy TransformerLens-format HF repo.

    ``checkpoint_index`` / ``checkpoint_value`` select a training checkpoint
    (``checkpoints/*_<label>.pth``); by default the final weights load. The
    resolved values are stamped on ``cfg.checkpoint_index`` /
    ``cfg.checkpoint_value``, mirroring the legacy loader.
    """
    if not model_name.startswith(TL_LEGACY_PREFIXES):
        raise ValueError(
            f"{model_name!r} is not a known legacy TransformerLens repo family "
            f"{TL_LEGACY_PREFIXES}. Use TransformerBridge.boot_transformers for "
            "HuggingFace-format models."
        )
    cfg = derive_tl_legacy_config(model_name)

    if checkpoint_index is not None or checkpoint_value is not None:
        from transformer_lens.tools.model_registry.checkpoints import (
            get_checkpoint_labels,
        )

        labels, _ = get_checkpoint_labels(model_name)
        if checkpoint_value is None:
            assert checkpoint_index is not None
            # Negative indices count from the end (checkpoint_index=-1 is the
            # final checkpoint), matching the legacy loader's list indexing.
            if not -len(labels) <= checkpoint_index < len(labels):
                raise ValueError(
                    f"checkpoint_index={checkpoint_index} out of range "
                    f"[-{len(labels)}, {len(labels)}) for {model_name!r}."
                )
            checkpoint_value = labels[checkpoint_index]
        elif checkpoint_value not in labels:
            raise ValueError(
                f"checkpoint_value={checkpoint_value} not in available checkpoints for "
                f"{model_name!r} ({len(labels)} labels, {labels[0]}..{labels[-1]})."
            )
        cfg.checkpoint_index = labels.index(checkpoint_value)  # type: ignore[attr-defined]
        cfg.checkpoint_value = checkpoint_value  # type: ignore[attr-defined]
    else:
        cfg.checkpoint_index = None  # type: ignore[attr-defined]
        cfg.checkpoint_value = None  # type: ignore[attr-defined]

    if tokenizer is None and cfg.tokenizer_name is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)

    from transformer_lens.model_bridge.sources.native import boot as _boot_native

    bridge = _boot_native(
        cfg, tokenizer=tokenizer, device=device, dtype=dtype, model_name=model_name
    )
    legacy_sd = fetch_tl_legacy_state_dict(
        model_name, cfg, checkpoint_value=checkpoint_value, dtype=dtype
    )
    converted = convert_tl_checkpoint(legacy_sd, cfg)
    # Several legacy repos ship no unembed bias; the legacy loader zero-filled
    # missing params, so mirror that for exactly this key rather than failing
    # a strict load or silently accepting arbitrary gaps.
    if "unembed.bias" not in converted:
        converted["unembed.bias"] = torch.zeros(cfg.d_vocab, dtype=dtype)
    result = bridge.load_state_dict(converted, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    return bridge


setattr(TransformerBridge, "boot_tl_legacy", staticmethod(boot))
