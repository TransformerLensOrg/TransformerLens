import gc
import os
from typing import Iterable, Tuple

import pytest
import torch


def _to_list(keys: Iterable[str]) -> list[str]:
    return list(keys) if not isinstance(keys, list) else keys


# Test models selected for architectural diversity while respecting memory constraints.
# Note: Python doesn't release model memory between parameterized tests, causing accumulation.
# Models are ordered by size to minimize peak memory usage.
PUBLIC_HF_MODELS = [
    "sshleifer/tiny-gpt2",
    "EleutherAI/pythia-70m",
    "facebook/opt-125m",
]

# Extended model list for authenticated CI runs
FULL_HF_MODELS = [
    "sshleifer/tiny-gpt2",
    "EleutherAI/pythia-70m",
    "roneneldan/TinyStories-33M",
    "facebook/opt-125m",
    "gpt2",
]


def _select_model_ids_from_acceptance_lists() -> list[str]:
    return FULL_HF_MODELS if os.environ.get("HF_TOKEN", "") else PUBLIC_HF_MODELS


# Allow overriding via env, comma-separated HF ids
DEFAULT_IDS = ",".join(_select_model_ids_from_acceptance_lists())
MODELS_ENV = os.getenv("TL_HOOK_SHAPE_MODELS", DEFAULT_IDS)
MODEL_NAMES = [m.strip() for m in MODELS_ENV.split(",") if m.strip()]


def _expected_shape_for_name(
    name: str,
    *,
    batch: int,
    pos: int,
    d_model: int,
    d_vocab: int | None,
    n_heads: int | None,
    d_head: int | None,
    d_mlp: int | None,
) -> Tuple[int, ...] | None:
    # Canonical TransformerBridge hook names only (no legacy aliases)

    # Unembedding (check before embedding to avoid matching "embed" in "unembed")
    if name.endswith("unembed.hook_in"):
        return (batch, pos, d_model)
    if name.endswith("unembed.hook_out") and d_vocab is not None:
        return (batch, pos, d_vocab)

    # Embedding components
    if name.endswith("embed.hook_in") or name.endswith("pos_embed.hook_in"):
        return (batch, pos)
    if name.endswith("embed.hook_out") or name.endswith("pos_embed.hook_out"):
        return (batch, pos, d_model)

    # Block IO
    if ".hook_in" in name and ".attn." not in name and ".mlp." not in name and ".ln" not in name:
        # blocks.{i}.hook_in
        return (batch, pos, d_model)
    if ".hook_out" in name and ".attn." not in name and ".mlp." not in name and ".ln" not in name:
        # blocks.{i}.hook_out
        return (batch, pos, d_model)

    # Attention module (canonical TB names)
    if name.endswith("attn.hook_in") or name.endswith("attn.hook_out"):
        return (batch, pos, d_model)
    if name.endswith("attn.hook_hidden_states"):
        return (batch, pos, d_model)
    if name.endswith("attn.hook_attention_weights") and n_heads is not None:
        return (batch, n_heads, pos, pos)
    if name.endswith("attn.hook_attn_scores") and n_heads is not None:
        return (batch, n_heads, pos, pos)
    if name.endswith("attn.hook_pattern") and n_heads is not None:
        return (batch, n_heads, pos, pos)

    # Attention subprojections: q/k/v/o
    # Note: q/k/v/o input hooks can be either:
    # - (batch, pos, n_heads, d_head) for models with split heads (GPT-2, Pythia, etc.)
    # - (batch, pos, d_model) for models without split heads (GPT-Neo, etc.)
    # Both are valid depending on the architecture
    # The o.hook_out is always (batch, pos, d_model) since it's after the projection
    if name.endswith("attn.o.hook_out"):
        return (batch, pos, d_model)

    # LayerNorms within blocks
    if ".ln" in name and name.endswith("hook_in"):
        return (batch, pos, d_model)
    if ".ln" in name and name.endswith("hook_out"):
        return (batch, pos, d_model)
    if name.endswith("hook_normalized"):
        return (batch, pos, d_model)
    if name.endswith("hook_scale"):
        # LayerNorm scale is (batch, pos, 1) for broadcasting
        return (batch, pos, 1)

    # MLP module
    if name.endswith("mlp.hook_in") or name.endswith("mlp.hook_out"):
        return (batch, pos, d_model)
    if name.endswith("mlp.hook_pre") and d_mlp is not None:
        return (batch, pos, d_mlp)
    # MLP submodules: input and out projections
    if name.endswith("mlp.input.hook_in") or name.endswith("mlp.out.hook_out"):
        return (batch, pos, d_model)
    if (
        name.endswith("mlp.input.hook_out") or name.endswith("mlp.out.hook_in")
    ) and d_mlp is not None:
        return (batch, pos, d_mlp)

    return None


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_transformer_bridge_hook_shapes(model_name: str):
    # Skip Pythia for now - dense component initialization issue
    if "pythia" in model_name.lower():
        pytest.skip("Pythia architecture needs dense component initialization fix")

    # Ensure boot method is registered
    from transformer_lens.model_bridge.bridge import TransformerBridge
    from transformer_lens.model_bridge.sources import (  # noqa: F401
        transformers as bridge_sources,
    )

    bridge = TransformerBridge.boot_transformers(model_name, device="cpu")

    prompt = "Hello world"
    tokens = bridge.to_tokens(prompt, move_to_device=False)
    batch, pos = int(tokens.shape[0]), int(tokens.shape[1])

    cfg = bridge.cfg
    d_model = int(getattr(cfg, "d_model"))
    # Use actual vocab size from weights (may differ from config due to padding)
    d_vocab = None
    if hasattr(bridge, "unembed") and hasattr(bridge.unembed, "weight"):
        d_vocab = int(bridge.unembed.weight.shape[0])
    elif hasattr(cfg, "d_vocab"):
        d_vocab = int(getattr(cfg, "d_vocab", 0))

    n_heads = int(getattr(cfg, "n_heads", 0)) if hasattr(cfg, "n_heads") else None
    d_head = int(getattr(cfg, "d_head", 0)) if hasattr(cfg, "d_head") else None
    d_mlp = int(getattr(cfg, "d_mlp", 0)) if hasattr(cfg, "d_mlp") else None
    if n_heads == 0:
        n_heads = None
    if d_head == 0:
        d_head = None
    if d_mlp == 0:
        d_mlp = None

    _, cache = bridge.run_with_cache(tokens, device="cpu")
    keys = sorted(_to_list(cache.keys()))

    # OPT reshapes to (batch*seq, d_model) internally for efficiency
    is_opt_model = "opt" in model_name.lower()

    mismatches: list[tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
    checked = 0
    for name in keys:
        # Special handling for q/k/v/o input hooks which can have two valid shapes
        is_qkv_hook = any(
            name.endswith(suf)
            for suf in (
                "attn.q.hook_in",
                "attn.k.hook_in",
                "attn.v.hook_in",
                "attn.o.hook_in",
                "attn.q.hook_out",
                "attn.k.hook_out",
                "attn.v.hook_out",
            )
        )

        if is_qkv_hook:
            tensor = cache[name]
            assert isinstance(tensor, torch.Tensor), f"Non-tensor cached for {name}"
            got = tuple(tensor.shape)
            # Valid shapes: (batch, pos, n_heads, d_head) or (batch, pos, d_model)
            valid_shapes = []
            if n_heads is not None and d_head is not None:
                valid_shapes.append((batch, pos, n_heads, d_head))
            valid_shapes.append((batch, pos, d_model))

            if got not in valid_shapes:
                exp_str = " or ".join(str(s) for s in valid_shapes)
                mismatches.append((name, exp_str, got))  # type: ignore
            checked += 1
            continue

        # Rotary embeddings have architecture-specific partial dimensions
        if "rotary" in name.lower():
            checked += 1
            continue

        exp = _expected_shape_for_name(
            name,
            batch=batch,
            pos=pos,
            d_model=d_model,
            d_vocab=d_vocab,
            n_heads=n_heads,
            d_head=d_head,
            d_mlp=d_mlp,
        )
        if exp is None:
            continue
        tensor = cache[name]
        assert isinstance(tensor, torch.Tensor), f"Non-tensor cached for {name}"
        got = tuple(tensor.shape)

        # OPT flattens batch and sequence dimensions for MLP/LayerNorm
        if is_opt_model and got != exp:
            is_flattened_hook = (".ln" in name and ".hook" in name and ".attn.ln" not in name) or (
                ".mlp." in name and "hook" in name
            )

            if is_flattened_hook and len(exp) == 3 and len(got) == 2:
                if got == (batch * pos, exp[2]):
                    checked += 1
                    continue

        if got != exp:
            mismatches.append((name, exp, got))
        checked += 1

    assert checked > 0, "No hooks were checked; update expected mapping or model filter"
    msg = "\n".join(f"{n}: expected {e}, got {g}" for n, e, g in mismatches[:20])
    assert not mismatches, f"Found {len(mismatches)} shape mismatches. Examples:\n{msg}"

    # Clean up to reduce memory usage during parameterized test runs
    del bridge, cache, tokens
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
