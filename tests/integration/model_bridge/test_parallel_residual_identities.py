"""Residual-stream decomposition identity for parallel-residual architectures.

Companion to ``test_residual_decomposition_identities.py``, which covers the
sequential wiring and scopes these architectures out. A parallel block has no
``hook_resid_mid`` — attention and MLP both read ``resid_pre`` and both write
into ``resid_post``, so the identity ``ParallelBlockBridge`` documents is:

    resid_post == resid_pre + attn_out + mlp_out

#1639 showed how quietly an alias can end up pointing at a residual-added state
instead of an additive contribution. That identity held for every
``ParallelBlockBridge`` adapter but nothing asserted it.

GPTNeoX, StableLM and Falcon pick their block class from the config
(``use_parallel_residual`` / ``parallel_attn``), so each is exercised in both
wirings — the sequential cases are the regression guard for a NeoX adapter that
hardcoded the parallel block and dropped ``hook_resid_mid`` on checkpoints that
genuinely have one.

Fixtures are seeded tinies built from local HF configs, so no hub access is
needed and both wirings are reachable from one checkpoint shape. Bridge-vs-HF
logit parity is deliberately not asserted here — that contract belongs to the
per-adapter tests (e.g. ``test_cohere_adapter.py``).
"""

import copy
from typing import Any

import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    CodeGenConfig,
    CohereConfig,
    FalconConfig,
    GPTJConfig,
    GPTNeoXConfig,
    PhiConfig,
    StableLmConfig,
)

from transformer_lens.model_bridge.sources import build_bridge_from_module

_SHARED = dict(vocab_size=128, pad_token_id=0, bos_token_id=1, eos_token_id=2)
# Multi-head (no GQA); GPTNeoX splits a fused QKV and needs n_kv == n_heads.
_MHA = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    max_position_embeddings=128,
    **_SHARED,
)
_GQA = dict(_MHA, num_key_value_heads=2)
_GPTJ = dict(n_embd=64, n_layer=2, n_head=4, n_positions=128, rotary_dim=16, **_SHARED)
_FALCON = dict(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    max_position_embeddings=128,
    new_decoder_architecture=False,
    **_SHARED,
)

PARALLEL_CASES = [
    pytest.param("GPTJForCausalLM", GPTJConfig, _GPTJ, id="gptj"),
    pytest.param("CodeGenForCausalLM", CodeGenConfig, _GPTJ, id="codegen"),
    pytest.param("PhiForCausalLM", PhiConfig, _GQA, id="phi"),
    pytest.param("CohereForCausalLM", CohereConfig, _GQA, id="cohere"),
    pytest.param(
        "GPTNeoXForCausalLM", GPTNeoXConfig, dict(_MHA, use_parallel_residual=True), id="neox"
    ),
    pytest.param(
        "StableLmForCausalLM", StableLmConfig, dict(_GQA, use_parallel_residual=True), id="stablelm"
    ),
    pytest.param("FalconForCausalLM", FalconConfig, dict(_FALCON, parallel_attn=True), id="falcon"),
]

SEQUENTIAL_CASES = [
    pytest.param(
        "GPTNeoXForCausalLM", GPTNeoXConfig, dict(_MHA, use_parallel_residual=False), id="neox"
    ),
    pytest.param(
        "StableLmForCausalLM",
        StableLmConfig,
        dict(_GQA, use_parallel_residual=False),
        id="stablelm",
    ),
    pytest.param(
        "FalconForCausalLM", FalconConfig, dict(_FALCON, parallel_attn=False), id="falcon"
    ),
]

# Built once per (architecture, wiring); the tests only read from them.
_CACHE: dict[tuple[str, str], tuple[Any, dict[str, torch.Tensor]]] = {}


def _run(hf_architecture: str, config_cls: Any, config_kwargs: dict[str, Any]):
    """Boot a seeded tiny bridge for this architecture and cache one forward."""
    key = (hf_architecture, repr(sorted(config_kwargs.items())))
    if key not in _CACHE:
        config = config_cls(**config_kwargs)
        config._attn_implementation = "eager"
        # Fork the RNG so seeding here stays deterministic on a cache miss
        # without leaking state into later unseeded tests on this worker.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            hf = AutoModelForCausalLM.from_config(config).eval()
            bridge = build_bridge_from_module(
                hf, hf_architecture, hf_config=copy.deepcopy(config), tokenizer=None, device="cpu"
            ).eval()

            torch.manual_seed(0)
            tokens = torch.randint(3, config_kwargs["vocab_size"], (1, 10))
            with torch.no_grad():
                logits, cache = bridge.run_with_cache(tokens)
        assert torch.isfinite(logits).all(), f"{hf_architecture} produced non-finite logits"
        _CACHE[key] = (bridge, cache)
    return _CACHE[key]


@pytest.mark.parametrize("hf_architecture,config_cls,config_kwargs", PARALLEL_CASES)
def test_parallel_block_omits_resid_mid(
    hf_architecture: str, config_cls: Any, config_kwargs: dict[str, Any]
) -> None:
    """A parallel block has no post-attention residual, so the alias must be absent."""
    bridge, cache = _run(hf_architecture, config_cls, config_kwargs)

    assert type(bridge.blocks[0]).__name__ == "ParallelBlockBridge"
    assert bridge.cfg.parallel_attn_mlp is True
    for layer in range(bridge.cfg.n_layers):
        assert f"blocks.{layer}.hook_resid_mid" not in cache


@pytest.mark.parametrize("hf_architecture,config_cls,config_kwargs", PARALLEL_CASES)
def test_parallel_residual_decomposition(
    hf_architecture: str, config_cls: Any, config_kwargs: dict[str, Any]
) -> None:
    """resid_post == resid_pre + attn_out + mlp_out on every layer."""
    bridge, cache = _run(hf_architecture, config_cls, config_kwargs)

    for layer in range(bridge.cfg.n_layers):
        torch.testing.assert_close(
            cache[f"blocks.{layer}.hook_resid_post"],
            cache[f"blocks.{layer}.hook_resid_pre"]
            + cache[f"blocks.{layer}.hook_attn_out"]
            + cache[f"blocks.{layer}.hook_mlp_out"],
            msg=lambda got: f"{hf_architecture} layer {layer} parallel identity broken:\n{got}",
        )


@pytest.mark.parametrize("hf_architecture,config_cls,config_kwargs", SEQUENTIAL_CASES)
def test_sequential_variant_exposes_resid_mid(
    hf_architecture: str, config_cls: Any, config_kwargs: dict[str, Any]
) -> None:
    """The sequential variant of a switchable family keeps its post-attention residual."""
    bridge, cache = _run(hf_architecture, config_cls, config_kwargs)

    assert type(bridge.blocks[0]).__name__ == "BlockBridge"
    assert bridge.cfg.parallel_attn_mlp is False
    for layer in range(bridge.cfg.n_layers):
        assert f"blocks.{layer}.hook_resid_mid" in cache


@pytest.mark.parametrize("use_parallel_residual", [True, False])
def test_hf_use_parallel_residual_reaches_the_bridge_config(use_parallel_residual: bool) -> None:
    """The adapter reads HF's own flag, so the passthrough must carry it — the
    wiring must not rest on two defaults happening to agree.
    """
    bridge, _ = _run(
        "GPTNeoXForCausalLM", GPTNeoXConfig, dict(_MHA, use_parallel_residual=use_parallel_residual)
    )

    assert bridge.cfg.use_parallel_residual is use_parallel_residual


@pytest.mark.parametrize("hf_architecture,config_cls,config_kwargs", SEQUENTIAL_CASES)
def test_sequential_variant_decomposition(
    hf_architecture: str, config_cls: Any, config_kwargs: dict[str, Any]
) -> None:
    """Both HookedTransformer identities hold once resid_mid is exposed."""
    bridge, cache = _run(hf_architecture, config_cls, config_kwargs)

    for layer in range(bridge.cfg.n_layers):
        torch.testing.assert_close(
            cache[f"blocks.{layer}.hook_resid_mid"],
            cache[f"blocks.{layer}.hook_resid_pre"] + cache[f"blocks.{layer}.hook_attn_out"],
            msg=lambda got: f"{hf_architecture} layer {layer} attn identity broken:\n{got}",
        )
        torch.testing.assert_close(
            cache[f"blocks.{layer}.hook_resid_post"],
            cache[f"blocks.{layer}.hook_resid_mid"] + cache[f"blocks.{layer}.hook_mlp_out"],
            msg=lambda got: f"{hf_architecture} layer {layer} mlp identity broken:\n{got}",
        )
