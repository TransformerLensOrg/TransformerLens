import copy
from types import SimpleNamespace
from typing import Any, Callable, Dict

import pytest
import torch
import torch.nn as nn
import transformers

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)

N_LAYERS = 3
D_MODEL = 16
TOKENS = torch.tensor([[3, 4, 5, 6, 7]])


def _bare_bridge(**block_lists: nn.Module) -> TransformerBridge:
    """A TransformerBridge with only the given block lists registered (no HF model)."""
    bridge = TransformerBridge.__new__(TransformerBridge)
    nn.Module.__init__(bridge)
    bridge.cfg = TransformerBridgeConfig(
        d_model=8,
        d_head=4,
        n_layers=1,
        n_ctx=16,
        d_vocab=32,
        d_mlp=16,
        n_heads=2,
        architecture="RavenForCausalLM",
    )
    for name, module in block_lists.items():
        bridge.add_module(name, module)
    return bridge


def test_stop_at_layer_raises_without_blocks_stack() -> None:
    """Raven-style prelude/core_block/coda lists must not silently ignore stop_at_layer."""
    bridge = _bare_bridge(
        prelude=nn.ModuleList([nn.Identity()]),
        core_block=nn.ModuleList([nn.Identity()]),
        coda=nn.ModuleList([nn.Identity()]),
    )
    with pytest.raises(NotImplementedError, match="stop_at_layer requires a 'blocks' stack"):
        bridge.forward(torch.zeros(1, 3, dtype=torch.long), stop_at_layer=0)


def test_blocks_guard_ignores_unregistered_blocks_attribute() -> None:
    """A wrapped HF model exposing `.blocks` must not satisfy the guard via __getattr__."""
    bridge = _bare_bridge()
    bridge.__dict__["original_model"] = SimpleNamespace(blocks=[object()])
    assert hasattr(bridge, "blocks")  # the trap: __getattr__ falls through to the HF model
    assert not bridge._has_registered_blocks()
    with pytest.raises(NotImplementedError, match="stop_at_layer requires a 'blocks' stack"):
        bridge.forward(torch.zeros(1, 3, dtype=torch.long), stop_at_layer=0)
    with pytest.raises(NotImplementedError, match="start_at_layer requires a 'blocks' stack"):
        bridge.forward(torch.zeros(1, 3, 8), start_at_layer=0)


def _tiny_bridge(architecture: str, cfg: Any, model_cls: type) -> TransformerBridge:
    cfg._attn_implementation = "eager"
    cfg.architectures = [architecture]
    torch.manual_seed(0)
    model = model_cls(cfg).eval()
    bridge = build_bridge_from_module(
        model, architecture, hf_config=copy.deepcopy(cfg), tokenizer=None, device="cpu"
    )
    return bridge.eval()


_BUILDERS: Dict[str, Callable[[], TransformerBridge]] = {
    "gpt2": lambda: _tiny_bridge(
        "GPT2LMHeadModel",
        transformers.GPT2Config(
            vocab_size=64, n_embd=D_MODEL, n_layer=N_LAYERS, n_head=2, n_positions=32
        ),
        transformers.GPT2LMHeadModel,
    ),
    "mamba": lambda: _tiny_bridge(
        "MambaForCausalLM",
        transformers.MambaConfig(
            vocab_size=64,
            hidden_size=D_MODEL,
            state_size=8,
            num_hidden_layers=N_LAYERS,
            expand=2,
            time_step_rank=4,
            conv_kernel=4,
        ),
        transformers.MambaForCausalLM,
    ),
    "bloom": lambda: _tiny_bridge(
        "BloomForCausalLM",
        transformers.BloomConfig(vocab_size=64, hidden_size=D_MODEL, n_layer=N_LAYERS, n_head=2),
        transformers.BloomForCausalLM,
    ),
    "bert": lambda: _tiny_bridge(
        "BertForMaskedLM",
        transformers.BertConfig(
            vocab_size=64,
            hidden_size=D_MODEL,
            num_hidden_layers=N_LAYERS,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
        ),
        transformers.BertForMaskedLM,
    ),
    "roberta": lambda: _tiny_bridge(
        "RobertaForMaskedLM",
        transformers.RobertaConfig(
            vocab_size=64,
            hidden_size=D_MODEL,
            num_hidden_layers=N_LAYERS,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
        ),
        transformers.RobertaForMaskedLM,
    ),
    "t5": lambda: _tiny_bridge(
        "T5ForConditionalGeneration",
        transformers.T5Config(
            vocab_size=64, d_model=D_MODEL, d_kv=8, d_ff=32, num_layers=2, num_heads=2
        ),
        transformers.T5ForConditionalGeneration,
    ),
}


@pytest.fixture(scope="module")
def tiny_bridge(request: pytest.FixtureRequest) -> TransformerBridge:
    return _BUILDERS[request.param]()


def _stop_with(bridge: TransformerBridge, api: str, k: int) -> torch.Tensor:
    if api == "forward":
        return bridge(TOKENS, stop_at_layer=k)
    if api == "run_with_hooks":
        return bridge.run_with_hooks(TOKENS, stop_at_layer=k)
    out, _ = bridge.run_with_cache(TOKENS, stop_at_layer=k)
    return out


@pytest.mark.parametrize("tiny_bridge", ["t5"], indirect=True)
def test_run_with_cache_raises_without_blocks_stack(tiny_bridge: TransformerBridge) -> None:
    """Encoder/decoder stacks must not let run_with_cache ignore stop_at_layer and return logits."""
    with pytest.raises(NotImplementedError, match="stop_at_layer requires a 'blocks' stack"):
        tiny_bridge.run_with_cache(TOKENS, stop_at_layer=1, decoder_input=TOKENS)
    leaked = [name for name, hp in tiny_bridge.hook_dict.items() if hp.fwd_hooks or hp.bwd_hooks]
    assert leaked == []
    assert getattr(tiny_bridge, "context_level", 0) == 0


@pytest.mark.parametrize("k", [0, 1, -1])
@pytest.mark.parametrize("api", ["forward", "run_with_hooks", "run_with_cache"])
@pytest.mark.parametrize("tiny_bridge", ["bert", "roberta"], indirect=True)
def test_stop_at_layer_stops_on_encoder_layer_names(
    tiny_bridge: TransformerBridge, api: str, k: int
) -> None:
    """``*.encoder.layer.N`` blocks stop at the residual entering block k, as on decoder stacks."""
    hook_name = f"blocks.{k % N_LAYERS}.hook_in"
    with torch.no_grad():
        _, cache = tiny_bridge.run_with_cache(TOKENS, names_filter=hook_name)
        out = _stop_with(tiny_bridge, api, k)
    assert out.shape == (1, TOKENS.shape[1], D_MODEL)
    torch.testing.assert_close(out, cache[hook_name])


@pytest.mark.parametrize("tiny_bridge", ["bert", "roberta"], indirect=True)
def test_input_to_embed_on_bert_returns_residual(tiny_bridge: TransformerBridge) -> None:
    """input_to_embed must return resid_pre[0], not MLM logits."""
    with torch.no_grad():
        residual, _, _, _ = tiny_bridge.input_to_embed(TOKENS)
        _, cache = tiny_bridge.run_with_cache(TOKENS, names_filter="blocks.0.hook_in")
    assert residual.shape == (1, TOKENS.shape[1], D_MODEL)
    torch.testing.assert_close(residual, cache["blocks.0.hook_in"])


@pytest.mark.parametrize("start", [1, 2])
@pytest.mark.parametrize("tiny_bridge", ["bert", "roberta", "bloom", "mamba"], indirect=True)
def test_start_at_layer_resumes_on_every_block_class(
    tiny_bridge: TransformerBridge, start: int
) -> None:
    """Resuming from a cached ``hook_in`` reproduces the full pass on encoder-layer names,
    Bloom's alibi block and the opaque SSM block, each of which runs its own forward."""
    with torch.no_grad():
        full, cache = tiny_bridge.run_with_cache(TOKENS)
        resumed = tiny_bridge(cache[f"blocks.{start}.hook_in"], start_at_layer=start)
    torch.testing.assert_close(resumed, full, atol=1e-5, rtol=0)


@pytest.mark.parametrize("k", [0, 1, N_LAYERS, -1])
@pytest.mark.parametrize("tiny_bridge", ["gpt2", "mamba"], indirect=True)
def test_run_with_cache_stop_parity_gpt2_and_mamba(tiny_bridge: TransformerBridge, k: int) -> None:
    """run_with_cache's stop returns the stopped residual, caches nothing past it, fires nothing later."""
    stop = k % N_LAYERS if k < 0 else k
    expected_name = (
        f"blocks.{stop}.hook_in" if stop < N_LAYERS else f"blocks.{N_LAYERS - 1}.hook_out"
    )
    fired: list = []

    def record(tensor: torch.Tensor, hook: Any) -> None:
        fired.append(hook.name)

    def block_idx(name: str) -> int:
        return int(name.split(".")[1]) if name.startswith("blocks.") else -1

    with torch.no_grad():
        _, full_cache = tiny_bridge.run_with_cache(TOKENS)
        # Component boundaries only: gated hooks (hook_result, ...) are off and would warn.
        with tiny_bridge.hooks(
            fwd_hooks=[(lambda n: n.endswith((".hook_in", ".hook_out")), record)]
        ):
            out, cache = tiny_bridge.run_with_cache(TOKENS, stop_at_layer=k)

    torch.testing.assert_close(out, full_cache[expected_name])
    assert [name for name in cache.keys() if block_idx(name) >= stop] == []
    late = [
        name
        for name in fired
        if (block_idx(name) >= stop and name != f"blocks.{stop}.hook_in")
        or name.startswith(("unembed.", "ln_final."))
    ]
    assert late == []
    assert expected_name in fired
