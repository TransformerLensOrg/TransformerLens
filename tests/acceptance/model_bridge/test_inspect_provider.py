"""Acceptance gate for the Inspect driver: gpt2 through our HF provider must match
``boot_transformers`` to fp tolerance (same backend + dtype + eager attention), with
exact next-token argmax, and interventions must take effect.

Gated on ``inspect_ai`` being installed (the ``inspect`` extra); runs on CPU.
"""
from __future__ import annotations

import pytest
import torch

try:
    import inspect_ai  # noqa: F401

    INSPECT_AVAILABLE = True
except ImportError:
    INSPECT_AVAILABLE = False

pytestmark = [
    pytest.mark.inspect,
    pytest.mark.skipif(
        not INSPECT_AVAILABLE, reason="inspect_ai not installed (pip install '.[inspect]')"
    ),
]

MODEL = "gpt2"
PROMPT = "The quick brown fox"
# A spread across boundary kinds and layers — all must match boot_transformers to fp
# tolerance. TransformerBridge-native names (the bridge cache carries these too).
PARITY_HOOKS = [
    "blocks.0.hook_in",  # resid_pre
    "blocks.0.attn.hook_out",  # attn_out
    "blocks.0.ln2.hook_in",  # resid_mid
    "blocks.0.mlp.hook_out",  # mlp_out
    "blocks.0.hook_out",  # resid_post
    "blocks.6.attn.hook_out",
    "blocks.11.hook_out",
    # Head-split kinds gpt2 serves despite its fused c_attn (q/k/v are gated): z reads
    # the c_proj (Conv1D) input, pattern rides output_attentions under eager.
    "blocks.0.attn.hook_z",
    "blocks.0.attn.hook_pattern",
]

# Boundary kind -> TransformerBridge-native hook suffix.
KIND_SUFFIX = {
    "resid_pre": "hook_in",
    "resid_mid": "ln2.hook_in",
    "resid_post": "hook_out",
    "attn_out": "attn.hook_out",
    "mlp_out": "mlp.hook_out",
}

# Token-free tiny-random checkpoints, one per structural-check code path: standard
# sequential (nothing gated), parallel-residual (resid_mid gated by the causal probe),
# post-norm (resid_mid gated by the linear-identity probe). Pins cross-family behavior
# so a detector/load-path regression on non-gpt2 archs fails CI instead of shipping silently.
STRUCTURAL_FAMILIES = [
    ("hf-internal-testing/tiny-random-LlamaForCausalLM", frozenset()),
    ("hf-internal-testing/tiny-random-GPTJForCausalLM", frozenset({"resid_mid"})),
    ("hf-internal-testing/tiny-random-Gemma2ForCausalLM", frozenset({"resid_mid"})),
]


@pytest.fixture(scope="module")
def hf_bridge():
    from transformer_lens.model_bridge.transformer_bridge import TransformerBridge

    return TransformerBridge.boot_transformers(MODEL)


@pytest.fixture(scope="module")
def inspect_bridge():
    from transformer_lens.model_bridge.remote_bridge import RemoteBridge

    bridge = RemoteBridge.boot_inspect(MODEL)
    yield bridge
    bridge.close()


@pytest.fixture(scope="module")
def tokens(hf_bridge):
    # Shared token ids so the only variable under test is the backend path, not
    # tokenization (to_tokens BOS parity is covered by test_string_input_parity).
    return hf_bridge.to_tokens(PROMPT)


class TestInspectParity:
    def test_argmax_matches_boot_transformers(self, hf_bridge, inspect_bridge, tokens):
        hf_logits = hf_bridge.forward(tokens)
        i_logits = inspect_bridge.forward(tokens)
        assert int(hf_logits[0, -1].argmax()) == int(i_logits[0, -1].argmax())

    def test_string_input_parity(self, hf_bridge, inspect_bridge):
        # Same STRING (not shared tokens): exercises RemoteBridge.to_tokens BOS handling.
        # A bare encode (no BOS) would diverge from boot_transformers here.
        hf_argmax = int(hf_bridge.forward(PROMPT)[0, -1].argmax())
        i_argmax = int(inspect_bridge.forward(PROMPT)[0, -1].argmax())
        assert hf_argmax == i_argmax

    def test_hooks_match(self, hf_bridge, inspect_bridge, tokens):
        _, hf_cache = hf_bridge.run_with_cache(tokens)
        _, i_cache = inspect_bridge.run_with_cache(tokens)
        for hook in PARITY_HOOKS:
            a, b = hf_cache[hook].float(), i_cache[hook].float()
            assert a.shape == b.shape
            assert torch.allclose(
                a, b, atol=1e-3, rtol=1e-3
            ), f"{hook} diverges: max {(a - b).abs().max().item():.2e}"

    def test_loss_matches(self, hf_bridge, inspect_bridge, tokens):
        # Full-sequence logits ⇒ loss is computable and matches boot_transformers.
        hf_loss = hf_bridge.forward(tokens, return_type="loss")
        i_loss = inspect_bridge.forward(tokens, return_type="loss")
        assert torch.allclose(hf_loss, i_loss, atol=1e-3)


# Each intervenable boundary kind at layer 0, with whether suppress flips the top token
# (mlp_out shifts the logits but not the argmax at layer 0). resid_pre exercises the
# forward_pre_hook path; the others the forward_hook path.
INTERVENE_HOOKS = [
    ("blocks.0.hook_in", True),  # resid_pre
    ("blocks.0.attn.hook_out", True),  # attn_out
    ("blocks.0.mlp.hook_out", False),  # mlp_out
    ("blocks.0.hook_out", True),  # resid_post
]


class TestInspectInterventions:
    @pytest.mark.parametrize("hook,flips_argmax", INTERVENE_HOOKS)
    def test_suppress_applies_at_each_boundary(self, inspect_bridge, tokens, hook, flips_argmax):
        clean_logits = inspect_bridge.forward(tokens)
        clean_argmax = int(clean_logits[0, -1].argmax())

        supp_logits, supp_cache = inspect_bridge.run_with_cache(
            tokens, intervene={hook: {"op": "suppress"}}
        )
        assert supp_cache[hook].abs().max().item() == 0.0  # capture reflects the intervention
        assert not torch.allclose(supp_logits, clean_logits)  # and it propagated to the logits
        if flips_argmax:
            assert int(supp_logits[0, -1].argmax()) != clean_argmax

    def test_intervention_reverts(self, inspect_bridge, tokens):
        clean_argmax = int(inspect_bridge.forward(tokens)[0, -1].argmax())
        inspect_bridge.forward(tokens, intervene={"blocks.0.hook_out": {"op": "suppress"}})
        assert int(inspect_bridge.forward(tokens)[0, -1].argmax()) == clean_argmax  # not sticky


class TestPreHookKwargs:
    """The resid_pre pre-hook runs with_kwargs=True so it modifies the right tensor
    whether hidden_states arrives positionally (args[0]) or as a kwarg."""

    CALL = "test_call"

    def _hook(self):
        from transformer_lens.model_bridge.sources.inspect.transformers_provider import (
            _pre_hook,
        )

        return _pre_hook(
            layer=0, want_capture=True, spec={"op": "suppress"}, raw={}, call_id=self.CALL
        )

    def _scope(self):
        from transformer_lens.model_bridge.sources.inspect.transformers_provider import (
            _current_call_id,
        )

        return _current_call_id.set(self.CALL)

    def test_positional_hidden_states(self):
        from transformer_lens.model_bridge.sources.inspect.transformers_provider import (
            _current_call_id,
        )

        hidden = torch.ones(1, 2, 4)
        token = self._scope()
        try:
            new_args, new_kwargs = self._hook()(None, (hidden, "mask"), {})
        finally:
            _current_call_id.reset(token)
        assert torch.allclose(new_args[0], torch.zeros_like(hidden)) and new_args[1] == "mask"
        assert new_kwargs == {}

    def test_kwarg_hidden_states(self):
        from transformer_lens.model_bridge.sources.inspect.transformers_provider import (
            _current_call_id,
        )

        hidden = torch.ones(1, 2, 4)
        token = self._scope()
        try:
            new_args, new_kwargs = self._hook()(
                None, (), {"hidden_states": hidden, "use_cache": True}
            )
        finally:
            _current_call_id.reset(token)
        assert new_args == ()
        assert torch.allclose(new_kwargs["hidden_states"], torch.zeros_like(hidden))  # suppressed
        assert new_kwargs["use_cache"] is True  # other kwargs preserved


class TestStructuralProbe:
    """The boot-time structural self-check: resid_mid is offered only when attn feeds mlp.
    Toy modules (no download) exercise the causal probe for sequential vs parallel blocks."""

    def _toy_model(self, parallel: bool = False, resid_scale: float = 1.0):
        import torch.nn as nn

        class Sub(nn.Module):
            def __init__(self, k):
                super().__init__()
                self.lin = nn.Linear(4, 4)
                self.k = k

            def forward(self, x):
                return self.lin(x) * self.k

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = Sub(0.5)
                self.mlp = Sub(0.1)

            def forward(self, x):
                a = self.self_attn(x)
                m = self.mlp(x) if parallel else self.mlp(x + a)  # parallel reads block input
                # resid_scale != 1 mimics post-norm/residual-multiplier archs (Gemma2/OLMo2/
                # Granite): outputs don't add linearly, so resid_pre + attn_out is wrong.
                return x + resid_scale * a + resid_scale * m

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(16, 4)
                self.layers = nn.ModuleList([Block()])

            def forward(self, ids):
                x = self.embed(ids)
                for b in self.layers:
                    x = b(x)
                return x

        torch.manual_seed(0)
        return Model()

    def test_sequential_offers_resid_mid(self):
        from transformer_lens.model_bridge.sources.inspect.transformers_provider import (
            _detect_capabilities,
        )

        m = self._toy_model()  # standard sequential, linear residual
        kinds, note = _detect_capabilities(m, m.layers, (None, None, None))
        assert "resid_mid" in kinds
        # No boundary kind gated. (Head kinds ARE gated — the toy has no head geometry —
        # so the note mentions only those.)
        assert "resid_mid" not in note and "attn_out" not in note and "mlp_out" not in note

    def test_parallel_gates_resid_mid(self):
        from transformer_lens.model_bridge.sources.inspect.transformers_provider import (
            _detect_capabilities,
        )

        m = self._toy_model(parallel=True)  # mlp reads block input, not attn output
        kinds, note = _detect_capabilities(m, m.layers, (None, None, None))
        assert "resid_mid" not in kinds
        assert {"resid_pre", "resid_post", "attn_out", "mlp_out"} <= kinds  # rest still served
        assert "resid_mid" in note  # note explains the gate

    def test_nonlinear_residual_gates_resid_mid(self):
        from transformer_lens.model_bridge.sources.inspect.transformers_provider import (
            _detect_capabilities,
        )

        # Sequential (attn feeds mlp) but outputs are scaled before the residual add, so
        # resid_post != resid_pre + attn_out + mlp_out — resid_mid must still be gated.
        m = self._toy_model(resid_scale=2.0)
        kinds, note = _detect_capabilities(m, m.layers, (None, None, None))
        assert "resid_mid" not in kinds
        assert {"resid_pre", "resid_post", "attn_out", "mlp_out"} <= kinds
        assert "resid_mid" in note

    def test_probe_leaves_global_rng_untouched(self):
        # The probe's attn perturbation must use a local generator — booting a model
        # should never reset the caller's torch RNG.
        from transformer_lens.model_bridge.sources.inspect.transformers_provider import (
            _detect_capabilities,
        )

        m = self._toy_model()
        before = torch.get_rng_state()
        _detect_capabilities(m, m.layers, (None, None, None))
        assert torch.equal(torch.get_rng_state(), before)


class TestStructuralCheckAcrossFamilies:
    """Real tiny-random models, one per detector code path, run in CI (no token, seconds).
    Locks both the gating decision and offered-boundary parity vs boot_transformers, so a
    structural-check or load-path regression on non-gpt2 architectures fails here."""

    @pytest.mark.parametrize("model_id,expected_gated", STRUCTURAL_FAMILIES)
    def test_gating_and_parity(self, model_id, expected_gated):
        from transformer_lens.model_bridge.remote_bridge import RemoteBridge
        from transformer_lens.model_bridge.transformer_bridge import TransformerBridge

        # Matched fp32 + eager attention (boot_inspect's defaults) so the only variable
        # is the boundary mapping.
        hf = TransformerBridge.boot_transformers(model_id, device="cpu", dtype=torch.float32)
        inspect = RemoteBridge.boot_inspect(model_id, dtype=torch.float32)
        try:
            supported = inspect._driver.supported_hook_points
            offered = {k for k, suf in KIND_SUFFIX.items() if f"blocks.0.{suf}" in supported}
            gated = frozenset(set(KIND_SUFFIX) - offered)
            assert gated == expected_gated, f"{model_id}: gated {sorted(gated)}"
            # gated kinds are reported non-fireable, not silently dropped
            for kind in expected_gated:
                assert f"blocks.0.{KIND_SUFFIX[kind]}" in inspect._driver.non_fireable_hook_points

            n_layers = int(hf.cfg.n_layers)
            toks = hf.to_tokens(PROMPT)
            _, hf_cache = hf.run_with_cache(toks)
            _, i_cache = inspect.run_with_cache(toks)
            assert int(hf.forward(toks)[0, -1].argmax()) == int(
                inspect.forward(toks)[0, -1].argmax()
            )
            for layer in sorted({0, n_layers - 1}):
                for kind in offered:
                    hk = f"blocks.{layer}.{KIND_SUFFIX[kind]}"
                    a, b = hf_cache[hk].float(), i_cache[hk].float()
                    assert torch.allclose(
                        a, b, atol=1e-3, rtol=1e-3
                    ), f"{model_id} {hk} diverges: {(a - b).abs().max().item():.2e}"
        finally:
            inspect.close()


HEAD_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
HEAD_HOOKS = [
    "blocks.0.attn.hook_q",
    "blocks.0.attn.hook_k",
    "blocks.0.attn.hook_v",
    "blocks.0.attn.hook_z",
    "blocks.0.attn.hook_pattern",
    "blocks.1.attn.hook_z",
]


class TestHeadSplitHooks:
    """Head-split q/k/v/z/pattern on a separate-projection arch (tiny-random Llama):
    capture parity vs boot_transformers, interventions, and the fused-qkv gating path
    (via the module-scoped gpt2 fixtures)."""

    @pytest.fixture(scope="class")
    def head_pair(self):
        from transformer_lens.model_bridge.remote_bridge import RemoteBridge
        from transformer_lens.model_bridge.transformer_bridge import TransformerBridge

        hf = TransformerBridge.boot_transformers(HEAD_MODEL, device="cpu", dtype=torch.float32)
        inspect = RemoteBridge.boot_inspect(HEAD_MODEL, dtype=torch.float32)
        yield hf, inspect
        inspect.close()

    def test_all_head_hooks_served(self, head_pair):
        _, inspect = head_pair
        for hook in HEAD_HOOKS:
            assert hook in inspect._driver.supported_hook_points, f"{hook} not served"

    def test_head_capture_parity(self, head_pair):
        hf, inspect = head_pair
        toks = hf.to_tokens(PROMPT)
        _, hf_cache = hf.run_with_cache(toks)
        _, i_cache = inspect.run_with_cache(toks)
        for hook in HEAD_HOOKS:
            a, b = hf_cache[hook].float(), i_cache[hook].float()
            assert a.shape == b.shape, f"{hook}: {tuple(b.shape)} vs bridge {tuple(a.shape)}"
            assert torch.allclose(
                a, b, atol=1e-4, rtol=1e-4
            ), f"{hook} diverges: max {(a - b).abs().max().item():.2e}"

    def test_suppress_v_zeroes_capture_and_moves_logits(self, head_pair):
        _, inspect = head_pair
        toks = inspect.to_tokens(PROMPT)
        base = inspect.forward(toks)
        logits, cache = inspect.run_with_cache(
            toks, intervene={"blocks.0.attn.hook_v": {"op": "suppress"}}
        )
        assert cache["blocks.0.attn.hook_v"].abs().max().item() == 0.0
        assert not torch.allclose(base, logits)

    def test_per_position_q_patch_is_position_scoped(self, head_pair):
        _, inspect = head_pair
        toks = inspect.to_tokens(PROMPT)
        _, base_cache = inspect.run_with_cache(toks)
        _, cache = inspect.run_with_cache(
            toks, intervene={"blocks.0.attn.hook_q": {"op": "add", "value": 5.0, "pos": 1}}
        )
        q_base, q_new = base_cache["blocks.0.attn.hook_q"][0], cache["blocks.0.attn.hook_q"][0]
        others = [p for p in range(q_base.shape[0]) if p != 1]
        assert torch.allclose(q_new[1], q_base[1] + 5.0, atol=1e-5)
        assert torch.equal(q_new[others], q_base[others])

    def test_pattern_intervention_rejected(self, head_pair):
        _, inspect = head_pair
        with pytest.raises(ValueError, match="capture-only"):
            inspect.forward(
                inspect.to_tokens(PROMPT),
                intervene={"blocks.0.attn.hook_pattern": {"op": "suppress"}},
            )

    def test_gpt2_fused_qkv_gated_but_z_pattern_served(self, inspect_bridge):
        supported = inspect_bridge._driver.supported_hook_points
        nonfireable = inspect_bridge._driver.non_fireable_hook_points
        for hook in ("blocks.0.attn.hook_q", "blocks.0.attn.hook_k", "blocks.0.attn.hook_v"):
            assert hook not in supported and hook in nonfireable
        assert "blocks.0.attn.hook_z" in supported
        assert "blocks.0.attn.hook_pattern" in supported

    def test_gptneo_wrapper_attention_serves_head_hooks(self):
        """GPT-Neo wraps the real attention (standard q_proj/out_proj) at attn.attention —
        the projection-host descent must find it, and captures must match the bridge."""
        from transformer_lens.model_bridge.remote_bridge import RemoteBridge
        from transformer_lens.model_bridge.transformer_bridge import TransformerBridge

        mid = "hf-internal-testing/tiny-random-GPTNeoForCausalLM"
        hf = TransformerBridge.boot_transformers(mid, device="cpu", dtype=torch.float32)
        inspect = RemoteBridge.boot_inspect(mid, dtype=torch.float32)
        try:
            for kind in ("q", "k", "v", "z", "pattern"):
                assert f"blocks.0.attn.hook_{kind}" in inspect._driver.supported_hook_points
            toks = hf.to_tokens(PROMPT)
            _, hf_cache = hf.run_with_cache(toks)
            _, i_cache = inspect.run_with_cache(toks)
            for hook in ("blocks.0.attn.hook_q", "blocks.0.attn.hook_z"):
                a, b = hf_cache[hook].float(), i_cache[hook].float()
                assert a.shape == b.shape
                assert torch.allclose(a, b, atol=1e-4, rtol=1e-4)
        finally:
            inspect.close()

    def test_provider_direct_interventions_validated(self, inspect_bridge, tokens):
        """The provider's documented extra_args interface must reject gated and
        capture-only intervention kinds instead of silently no-op'ing (the driver path
        already rejects; this covers callers that speak to the provider directly)."""
        import asyncio

        from inspect_ai.model import GenerateConfig

        api = inspect_bridge._driver._model.api

        def call(interventions):
            cfg = GenerateConfig(
                extra_body={
                    "extra_args": {
                        "input_ids": tokens[0].tolist(),
                        "capture": [],
                        "interventions": interventions,
                    }
                }
            )
            return asyncio.run(api.generate("", [], None, cfg))

        with pytest.raises(ValueError, match="gated"):  # q gated on gpt2's fused c_attn
            call({"0:q": {"op": "suppress"}})
        with pytest.raises(ValueError, match="capture-only"):
            call({"0:resid_mid": {"op": "suppress"}})


class TestRebootSemantics:
    """inspect_ai memoizes get_model by name; boot_inspect passes memoize=False so each
    boot honors its own args and independently owns (and frees) its model."""

    def test_reboot_is_fresh_and_honors_dtype(self):
        from transformer_lens.model_bridge.remote_bridge import RemoteBridge

        mid = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        b1 = RemoteBridge.boot_inspect(mid, dtype=torch.float32)
        b2 = RemoteBridge.boot_inspect(mid, dtype=torch.float16)
        try:
            # Distinct provider objects — not inspect_ai's memoized singleton.
            assert b1._driver._model is not b2._driver._model
            # Each provider loaded at the dtype its boot requested (not a stale cache).
            assert next(b1._driver._model.api._hf.parameters()).dtype == torch.float32
            assert next(b2._driver._model.api._hf.parameters()).dtype == torch.float16
        finally:
            b1.close()
            b2.close()


class TestEvalNativeGeneration:
    """Beyond TL-driven capture, the provider works as a normal Inspect model: real
    multi-token generation from chat input, with logprobs + usage, so an Inspect eval runs."""

    def _api(self):
        from inspect_ai.model import get_model

        from transformer_lens.model_bridge.sources.inspect import (  # noqa: F401 register
            transformers_provider,
        )

        return get_model("tl_bridge/gpt2", memoize=False).api

    def test_multi_token_generation_with_logprobs(self):
        import asyncio

        from inspect_ai.model import ChatMessageUser, GenerateConfig

        out = asyncio.run(
            self._api().generate(
                [ChatMessageUser(content="The capital of France is")],
                None,
                None,
                GenerateConfig(max_tokens=5, logprobs=True, top_logprobs=3),
            )
        )
        choice = out.choices[0]
        assert choice.message.text  # non-empty completion
        assert choice.stop_reason in ("max_tokens", "stop")
        assert len(choice.logprobs.content) == 5  # one logprob per generated token
        assert len(choice.logprobs.content[0].top_logprobs) == 3
        assert out.usage.output_tokens == 5
        assert out.usage.total_tokens == out.usage.input_tokens + 5

    def test_eval_runs_end_to_end(self, tmp_path):
        from inspect_ai import Task
        from inspect_ai import eval as inspect_eval
        from inspect_ai.dataset import Sample
        from inspect_ai.scorer import includes
        from inspect_ai.solver import generate

        from transformer_lens.model_bridge.sources.inspect import (  # noqa: F401 register
            transformers_provider,
        )

        task = Task(
            dataset=[Sample(input="The capital of France is", target="Paris")],
            solver=generate(),
            scorer=includes(),
        )
        logs = inspect_eval(
            task, model="tl_bridge/gpt2", max_tokens=8, log_dir=str(tmp_path), display="none"
        )
        assert logs[0].status == "success"
        assert logs[0].samples and logs[0].samples[0].output.completion


class TestCaptureInEval:
    """The capture_activations solver: full activations to a per-sample side artifact + a
    reduction in the store, surfaced by samples_df to correlate features with scores."""

    def test_artifact_store_and_samples_df(self, tmp_path):
        import json
        import pathlib

        import numpy as np
        from inspect_ai import Task
        from inspect_ai import eval as inspect_eval
        from inspect_ai.analysis import SampleSummary, samples_df
        from inspect_ai.dataset import Sample
        from inspect_ai.solver import generate

        from transformer_lens.model_bridge.sources.inspect import (  # noqa: F401 register
            transformers_provider,
        )
        from transformer_lens.model_bridge.sources.inspect.eval import (
            activations_column,
            capture_activations,
        )

        acts, logs = str(tmp_path / "acts"), str(tmp_path / "logs")
        task = Task(
            dataset=[Sample(input="The capital of France is", target="Paris")],
            solver=[capture_activations(["blocks.6.hook_out"], output_dir=acts), generate()],
        )
        log = inspect_eval(
            task, model="tl_bridge/gpt2", max_tokens=4, log_dir=logs, display="none"
        )[0]
        assert log.status == "success"

        # full activations side-artifact, keyed by hook name
        npz = list(pathlib.Path(acts).glob("*.npz"))
        assert len(npz) == 1
        assert np.load(npz[0])["blocks.6.hook_out"].shape[-1] == 768

        # reduction in the store, queryable via samples_df
        df = samples_df(logs, columns=[*SampleSummary, activations_column()])
        reduction = df["tl_activations"].iloc[0]
        reduction = json.loads(reduction) if isinstance(reduction, str) else reduction
        assert reduction["blocks.6.hook_out"]["shape"][-1] == 768

    def test_rejects_unknown_hook(self):
        from transformer_lens.model_bridge.sources.inspect.eval import (
            capture_activations,
        )

        with pytest.raises(ValueError, match="not a fireable hook"):
            capture_activations(["blocks.0.not_a_hook"])


class TestAgenticToolCapture:
    """Honor tools (render into the template or raise clearly), best-effort tool-call
    parsing, and per-turn activation capture across a rollout."""

    def test_tool_call_parsing(self):
        from transformer_lens.model_bridge.sources.inspect.transformers_provider import (
            _parse_tool_calls,
        )

        block = _parse_tool_calls(
            'ok <tool_call>{"name": "add", "arguments": {"a": 1}}</tool_call>'
        )
        assert block and block[0].function == "add" and block[0].arguments == {"a": 1}
        bare = _parse_tool_calls('{"name": "f", "arguments": {}}')
        assert bare and bare[0].function == "f"
        assert _parse_tool_calls("no tool calls in this text") is None

    def test_tools_without_chat_template_raise(self):
        import asyncio

        from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model
        from inspect_ai.tool import ToolInfo, ToolParams

        from transformer_lens.model_bridge.sources.inspect import (  # noqa: F401 register
            transformers_provider,
        )

        api = get_model("tl_bridge/gpt2", memoize=False).api
        tool = ToolInfo(name="add", description="add two ints", parameters=ToolParams())
        with pytest.raises(NotImplementedError, match="tool-aware chat template"):
            asyncio.run(
                api.generate(
                    [ChatMessageUser(content="2+2?")], [tool], None, GenerateConfig(max_tokens=3)
                )
            )

    def test_per_turn_capture_collected_from_transcript(self, tmp_path):
        from inspect_ai import Task
        from inspect_ai import eval as inspect_eval
        from inspect_ai.dataset import Sample
        from inspect_ai.solver import generate

        from transformer_lens.model_bridge.sources.inspect import (  # noqa: F401 register
            transformers_provider,
        )
        from transformer_lens.model_bridge.sources.inspect.eval import turn_activations

        task = Task(
            dataset=[Sample(input="The capital of France is", target="Paris")], solver=generate()
        )
        log = inspect_eval(
            task,
            model="tl_bridge/gpt2",
            model_args={"capture": ["blocks.6.hook_out"]},
            max_tokens=4,
            log_dir=str(tmp_path),
            display="none",
        )[0]
        assert log.status == "success"
        # each model turn's activations are recoverable from the transcript
        turns = turn_activations(log.samples[0])
        assert len(turns) >= 1
        assert turns[0]["blocks.6.hook_out"].shape[-1] == 768


class TestEvalPathStructuralGating:
    """Both eval entry points (``model_args={"capture": [...]}`` and the solver's
    ``extra_args`` route through ``_generate_capture``) must enforce the same structural
    gate as the driver path — pinned on GPT-J, where resid_mid is gated."""

    PARALLEL = "hf-internal-testing/tiny-random-GPTJForCausalLM"
    GATED_HOOK = "blocks.0.ln2.hook_in"  # resid_mid

    def test_model_args_capture_rejects_gated_kind(self):
        from inspect_ai.model import get_model

        from transformer_lens.model_bridge.sources.inspect import (  # noqa: F401 register
            transformers_provider,
        )

        with pytest.raises(ValueError, match="resid_mid"):
            get_model(f"tl_bridge/{self.PARALLEL}", memoize=False, capture=[self.GATED_HOOK])

    def test_extra_args_capture_rejects_gated_kind(self):
        import asyncio

        from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model

        from transformer_lens.model_bridge.sources.inspect import (  # noqa: F401 register
            transformers_provider,
        )

        api = get_model(f"tl_bridge/{self.PARALLEL}", memoize=False).api
        with pytest.raises(ValueError, match="resid_mid"):
            asyncio.run(
                api.generate(
                    [ChatMessageUser(content="hi")],
                    None,
                    None,
                    GenerateConfig(extra_body={"extra_args": {"capture": ["0:resid_mid"]}}),
                )
            )

    def test_model_args_unresolvable_hook_raises(self):
        # model_args["capture"] must validate like the solver does (was silently dropping).
        from inspect_ai.model import get_model

        from transformer_lens.model_bridge.sources.inspect import (  # noqa: F401 register
            transformers_provider,
        )

        with pytest.raises(ValueError, match="not a fireable hook"):
            get_model("tl_bridge/gpt2", memoize=False, capture=["blocks.0.not_a_hook"])


class TestProviderReviewFixes:
    """Targeted regressions for the post-merge review: validation parity, output_dir
    absolute-resolve, ContextVar hook isolation (concurrency-safe), and per-turn capture
    without the extra prompt forward."""

    def test_solver_output_dir_resolved_absolute_at_construction(self, tmp_path, monkeypatch):
        # A multi-eval run from different CWDs must not scatter artifacts.

        from transformer_lens.model_bridge.sources.inspect.eval import (
            capture_activations,
        )

        monkeypatch.chdir(tmp_path)
        # The solver factory captures output_dir in its closure; resolve at construction.
        cap = capture_activations(["blocks.0.hook_out"], output_dir="tl_acts")
        # Move CWD; the path the solver writes to must still resolve to tmp_path/tl_acts.
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        # Closure inspection: the wrapped solve fn closes over the absolute output_dir.
        absolute = str(tmp_path / "tl_acts")
        assert any(absolute == c.cell_contents for c in (cap.__closure__ or ()))

    def test_hooks_isolated_by_contextvar(self):
        # The hook only fires for the call whose call_id matches the contextvar — so two
        # concurrent inspect_eval samples (each in their own contextvar copy) don't write
        # into each other's raw dicts.
        from transformer_lens.model_bridge.sources.inspect.transformers_provider import (
            _current_call_id,
            _out_hook,
        )

        raw: dict = {}
        hook = _out_hook(
            layer=0, kind="resid_post", want_capture=True, spec=None, raw=raw, call_id="MINE"
        )

        # Default contextvar ("") doesn't match -> hook is a no-op.
        hook(None, None, torch.ones(1, 3, 4))
        assert raw == {}

        # Set the contextvar to a DIFFERENT call's id -> still skip (concurrent peer).
        token = _current_call_id.set("OTHER")
        try:
            hook(None, None, torch.ones(1, 3, 4))
            assert raw == {}
        finally:
            _current_call_id.reset(token)

        # Set the contextvar to our id -> hook fires.
        token = _current_call_id.set("MINE")
        try:
            hook(None, None, torch.ones(1, 3, 4))
            assert (0, "resid_post") in raw

            # First-write-wins: a second call with different data doesn't overwrite (so
            # generate's decode-step forwards don't clobber the prompt-forward capture).
            first = raw[(0, "resid_post")].copy()
            hook(None, None, torch.zeros(1, 3, 4))
            assert (raw[(0, "resid_post")] == first).all()
        finally:
            _current_call_id.reset(token)
