"""Keep docs/source/content/model_structure.md honest about hook names and alias targets."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import (
    Gemma3ForCausalLM,
    Gemma3TextConfig,
    GPT2Config,
    GPT2LMHeadModel,
)

from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.bridge_core import build_alias_to_canonical_map
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)

DOC = Path(__file__).resolve().parents[3] / "docs" / "source" / "content" / "model_structure.md"
FENCED_BLOCK = re.compile(r"```.*?```", re.S)
# No newlines inside a token: a stray fence would otherwise flip open/close parity.
BACKTICKED = re.compile(r"`([^`\n]+)`")
# Fully-qualified hook names only; bare short names in prose (`hook_q`) are not resolvable.
QUALIFIED_HOOK = re.compile(
    r"^(blocks\.\{i\}\.[a-z0-9_.]*hook_[a-z0-9_]+"
    r"|(embed|pos_embed|unembed|ln_final|rotary_emb)\.[a-z0-9_.]*hook_[a-z0-9_]+"
    r"|hook_embed|hook_pos_embed|hook_unembed)$"
)


def _randomize_norm_gains(model: nn.Module) -> None:
    """Fresh HF norms have unit gain, which would make hook_normalized equal hook_out."""
    for module in model.modules():
        if isinstance(module, nn.LayerNorm) or "Norm" in type(module).__name__:
            weight = getattr(module, "weight", None)
            if isinstance(weight, torch.Tensor):
                weight.data.normal_(mean=1.0, std=0.5)
            bias = getattr(module, "bias", None)
            if isinstance(bias, torch.Tensor):
                bias.data.normal_(std=0.5)


def _tiny_gpt2() -> TransformerBridge:
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=64, n_positions=16, n_embd=32, n_layer=2, n_head=2)
    model = GPT2LMHeadModel(cfg).eval()
    _randomize_norm_gains(model)
    return build_bridge_from_module(model, "GPT2LMHeadModel", hf_config=cfg, device="cpu")


def _tiny_gemma3() -> TransformerBridge:
    torch.manual_seed(0)
    cfg = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=16,
    )
    model = Gemma3ForCausalLM(cfg).eval()
    _randomize_norm_gains(model)
    return build_bridge_from_module(model, "Gemma3ForCausalLM", hf_config=cfg, device="cpu")


@pytest.fixture(scope="module")
def bridges() -> dict[str, TransformerBridge]:
    return {"gpt2": _tiny_gpt2(), "gemma3": _tiny_gemma3()}


@pytest.fixture(scope="module")
def doc_text() -> str:
    return DOC.read_text()


def _alias_table_rows(doc_text: str) -> list[tuple[str, set[str]]]:
    """(legacy name, accepted canonical targets) per row of the legacy alias table."""
    section = doc_text.split("### Legacy alias table", 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in section.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        legacy = BACKTICKED.findall(cells[0])
        canonical = BACKTICKED.findall(cells[1])
        assert len(legacy) == 1 and len(canonical) == 1, line
        note_hooks = {n for n in BACKTICKED.findall(cells[2]) if "hook_" in n}
        # A "not X" note names a hook the alias must NOT resolve to.
        if "not " in cells[2]:
            note_hooks = set()
        rows.append((legacy[0], {canonical[0]} | note_hooks))
    assert len(rows) >= 15
    return rows


def _cache(bridge: TransformerBridge, seed: int = 1):
    torch.manual_seed(seed)
    tokens = torch.randint(0, 64, (1, 5))
    with torch.no_grad():
        _, cache = bridge.run_with_cache(tokens)
    return cache


def test_alias_table_rows_resolve_on_both_bridges(bridges, doc_text):
    for name, bridge in bridges.items():
        alias_map = build_alias_to_canonical_map(bridge.hook_dict)
        for legacy, accepted in _alias_table_rows(doc_text):
            legacy_0 = legacy.replace("{i}", "0")
            accepted_0 = {a.replace("{i}", "0") for a in accepted}
            assert legacy_0 in alias_map, f"{name}: {legacy} is not an alias"
            assert (
                alias_map[legacy_0] in accepted_0
            ), f"{name}: {legacy} resolves to {alias_map[legacy_0]}, doc says {sorted(accepted)}"


def test_every_qualified_hook_name_on_the_page_exists(bridges, doc_text):
    known: set[str] = set()
    for bridge in bridges.values():
        known |= set(bridge.hook_dict.keys())
    missing = sorted(
        {
            token
            for token in BACKTICKED.findall(FENCED_BLOCK.sub("", doc_text))
            if QUALIFIED_HOOK.match(token) and token.replace("{i}", "0") not in known
        }
    )
    assert not missing, f"documented hooks that no tiny bridge exposes: {missing}"


@pytest.mark.parametrize("name", ["gpt2", "gemma3"])
def test_norm_hooks_are_four_distinct_tensors(bridges, name):
    bridge = bridges[name]
    cache = _cache(bridge)
    ln2 = bridge.blocks[0].ln2
    x = cache["blocks.0.ln2.hook_in"]
    scale = cache["blocks.0.ln2.hook_scale"]
    normalized = cache["blocks.0.ln2.hook_normalized"]
    out = cache["blocks.0.ln2.hook_out"]
    assert scale.shape == (*x.shape[:-1], 1)
    centered = x if ln2.uses_rms_norm else x - x.mean(-1, keepdim=True)
    torch.testing.assert_close(normalized, centered / scale, rtol=1e-5, atol=1e-6)
    with torch.no_grad():
        torch.testing.assert_close(out, ln2.original_component(x), rtol=1e-5, atol=1e-6)
    assert not torch.allclose(normalized, out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("name", ["gpt2", "gemma3"])
def test_hook_mlp_in_is_the_branch_entry_not_the_mlp_input(bridges, name):
    bridge = bridges[name]
    assert "blocks.0.hook_mlp_in" not in _cache(bridge)
    bridge.set_use_hook_mlp_in(True)
    try:
        cache = _cache(bridge)
    finally:
        bridge.set_use_hook_mlp_in(False)
    mlp_in = cache["blocks.0.hook_mlp_in"]
    torch.testing.assert_close(mlp_in, cache["blocks.0.ln2.hook_in"])
    torch.testing.assert_close(mlp_in, cache["blocks.0.hook_resid_mid"])
    assert not torch.allclose(mlp_in, cache["blocks.0.mlp.hook_in"], rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("name", ["gpt2", "gemma3"])
def test_flag_gated_attention_hooks_fire_per_head_pre_norm(bridges, name):
    bridge = bridges[name]
    n_heads = bridge.cfg.n_heads
    d_model = bridge.cfg.d_model
    base = _cache(bridge)
    for key in ("attn.hook_attn_in", "attn.hook_q_input", "attn.hook_result", "hook_attn_in"):
        assert f"blocks.0.{key}" not in base
    bridge.set_use_attn_in(True)
    bridge.set_use_attn_result(True)
    try:
        cache = _cache(bridge)
    finally:
        bridge.set_use_attn_in(False)
        bridge.set_use_attn_result(False)
    attn_in = cache["blocks.0.attn.hook_attn_in"]
    assert attn_in.shape == (1, 5, n_heads, d_model)
    torch.testing.assert_close(cache["blocks.0.hook_attn_in"], attn_in)
    for head in range(n_heads):
        torch.testing.assert_close(attn_in[:, :, head, :], cache["blocks.0.hook_in"])
    assert not torch.allclose(attn_in[:, :, 0, :], cache["blocks.0.attn.hook_in"], atol=1e-3)
    assert cache["blocks.0.attn.hook_result"].shape == (1, 5, n_heads, d_model)
