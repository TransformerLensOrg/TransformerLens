"""run_with_hooks and hooks() warn once per run about forward hooks that never fired.

A hook can attach to a hook point the forward pass never reaches; the caller then reads
or edits nothing and nothing says so. Each test pairs the silence it checks with a dead
hook that must be named, so none can pass by the warning simply not existing.
"""

import copy
import warnings
from typing import Any

import pytest
import torch
from transformers import AutoModelForCausalLM, CodeGenConfig, LlamaConfig

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.sources import build_bridge_from_module

TOKENS = torch.tensor([[1, 17, 42, 99, 7, 63, 5, 80]])
DEAD = "blocks.0.hook_off_path"


def _bridge(architecture: str, config: Any) -> Any:
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        hf_model = AutoModelForCausalLM.from_config(config).eval()
    return build_bridge_from_module(
        hf_model, architecture, hf_config=copy.deepcopy(config), device="cpu"
    ).eval()


@pytest.fixture
def bridge() -> Any:
    """A three-layer Llama with one registered hook point no forward pass reaches."""
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = _bridge("LlamaForCausalLM", config)
    model.blocks[0].hook_off_path = HookPoint()
    model.clear_hook_registry()
    model._initialize_hook_registry()
    assert DEAD in model.hook_dict
    return model


def _keep(tensor: torch.Tensor, *, hook: Any) -> None:
    """A hook that takes `hook` by keyword only, as HookPoint passes it."""
    return None


def _dead_hook_warnings(records: Any) -> list[str]:
    return [str(r.message) for r in records if "never fired" in str(r.message)]


def test_run_with_hooks_names_the_dead_hook_and_not_the_one_that_fired(bridge: Any) -> None:
    fired: list[str] = []
    with pytest.warns(UserWarning, match="never fired") as records:
        bridge.run_with_hooks(
            TOKENS,
            fwd_hooks=[
                ("blocks.0.hook_resid_post", lambda tensor, hook: fired.append(hook.name)),
                (DEAD, _keep),
            ],
        )
    messages = _dead_hook_warnings(records)
    assert len(messages) == 1, "one warning per run"
    assert DEAD in messages[0] and "run_with_hooks" in messages[0]
    assert "hook_resid_post" not in messages[0]
    assert fired, "the tracked hook still runs the caller's function"


def test_hooks_context_names_the_dead_hook_on_exit(bridge: Any) -> None:
    with pytest.warns(UserWarning, match="never fired") as records:
        with bridge.hooks(fwd_hooks=[("blocks.1.hook_resid_post", _keep), (DEAD, _keep)]):
            bridge(TOKENS)
            bridge(TOKENS)
    messages = _dead_hook_warnings(records)
    assert len(messages) == 1 and DEAD in messages[0] and "hooks:" in messages[0]
    assert "hook_resid_post" not in messages[0]


def test_a_stopped_run_warns_only_for_hooks_before_the_stop(bridge: Any) -> None:
    with pytest.warns(UserWarning, match="never fired") as records:
        bridge.run_with_hooks(
            TOKENS,
            stop_at_layer=1,
            fwd_hooks=[
                (DEAD, _keep),
                ("blocks.2.hook_resid_post", _keep),
                ("ln_final.hook_out", _keep),
            ],
        )
    (message,) = _dead_hook_warnings(records)
    assert DEAD in message
    assert "blocks.2" not in message and "ln_final" not in message

    with pytest.warns(UserWarning, match="never fired") as records:
        with bridge.hooks(fwd_hooks=[(DEAD, _keep), ("blocks.2.hook_resid_post", _keep)]):
            bridge(TOKENS, stop_at_layer=1)
    (message,) = _dead_hook_warnings(records)
    assert DEAD in message and "blocks.2" not in message


def test_a_filter_is_dead_only_when_none_of_its_matches_fired(bridge: Any) -> None:
    with pytest.warns(UserWarning, match="never fired") as records:
        bridge.run_with_hooks(
            TOKENS,
            fwd_hooks=[
                (lambda name: name.startswith("blocks.0.hook_"), _keep),
                (lambda name: name == DEAD, _keep),
            ],
        )
    (message,) = _dead_hook_warnings(records)
    assert "<name filter #1>" in message and "<name filter #0>" not in message


def test_cache_callbacks_backward_hooks_and_unrun_contexts_are_not_reported(bridge: Any) -> None:
    _, cache_hooks, _ = bridge.get_caching_hooks()
    assert any(name == DEAD or callable(name) for name, _ in cache_hooks)
    with pytest.warns(UserWarning, match="never fired") as records:
        bridge.run_with_hooks(
            TOKENS, fwd_hooks=[*cache_hooks, (DEAD, _keep)], bwd_hooks=[(DEAD, _keep)]
        )
    (message,) = _dead_hook_warnings(records)
    assert message.count("blocks.0.hook_off_path") == 1 and "1 forward hook" in message

    with warnings.catch_warnings(record=True) as quiet:
        warnings.simplefilter("always")
        # No forward pass ran inside the context, so there is nothing to judge.
        with bridge.hooks(fwd_hooks=[(DEAD, _keep)]):
            pass
        with pytest.raises(RuntimeError, match="boom"):
            with bridge.hooks(fwd_hooks=[(DEAD, _keep)]):
                bridge(TOKENS)
                raise RuntimeError("boom")
    assert not _dead_hook_warnings(quiet)


def test_codegen_split_qkv_hook_fires_and_only_the_dead_hook_is_reported() -> None:
    """CodeGen with `use_split_qkv_input` feeds `hook_q_input`; only a hook no forward pass reaches is named."""
    settings: dict[str, Any] = dict(
        vocab_size=128,
        n_embd=64,
        n_layer=2,
        n_head=4,
        n_positions=128,
        rotary_dim=16,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config = CodeGenConfig(**settings)
    codegen = _bridge("CodeGenForCausalLM", config)
    codegen.set_use_split_qkv_input(True)
    codegen.blocks[0].hook_off_path = HookPoint()
    codegen.clear_hook_registry()
    codegen._initialize_hook_registry()
    assert DEAD in codegen.hook_dict

    fired: list[str] = []

    def _record(tensor: torch.Tensor, *, hook: Any) -> None:
        fired.append(hook.name)

    with pytest.warns(UserWarning, match="never fired") as records:
        codegen.run_with_hooks(
            TOKENS, fwd_hooks=[("blocks.0.hook_q_input", _record), (DEAD, _keep)]
        )
    (message,) = _dead_hook_warnings(records)
    assert DEAD in message and "blocks.0.hook_q_input" not in message
    assert "blocks.0.hook_q_input" in fired
