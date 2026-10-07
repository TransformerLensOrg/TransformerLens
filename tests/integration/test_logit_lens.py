"""Integration tests for the logit lens: exactness against real models.

The contract under test is that the final entry of the accumulated stack,
read through the real final norm, ``W_U``, ``b_U`` and the adapter's output
transform, reproduces the bridge's own logits. gpt2 (cached fixture),
Pythia-14m and Qwen2.5-0.5B cover LayerNorm with bias, parallel residuals, and
RMSNorm with GQA; a softcapped Gemma-2 and a hybrid NemotronH are built from
tiny configs in memory so the softcap and SSM paths run in regular CI.
"""

import pytest
import torch

from transformer_lens.tools.analysis import logit_lens, logit_readout

PROMPT = "The Eiffel Tower is in the city of"
ANSWER = " Paris"
TOL = dict(atol=1e-4, rtol=1e-4)


def _exact_at_final_entry(model, tokens):
    with torch.no_grad():
        ref = model(tokens)
    result = logit_lens(model, tokens, positions=None)
    torch.testing.assert_close(result.values[-1], ref, **TOL)
    return ref, result


class TestGPT2Raw:
    def test_final_entry_reproduces_logits(self, gpt2_bridge):
        tokens = gpt2_bridge.to_tokens(PROMPT)
        ref, result = _exact_at_final_entry(gpt2_bridge, tokens)
        assert result.readout.applied_ln
        # Teeth: the final norm is load-bearing on gpt2.
        raw = logit_lens(gpt2_bridge, tokens, apply_ln=False)
        assert (raw.values[-1] - ref[:, -1]).abs().max() > 1.0

    def test_readout_matches_logits_and_chunking(self, gpt2_bridge):
        tokens = gpt2_bridge.to_tokens(PROMPT)
        with torch.no_grad():
            ref = gpt2_bridge(tokens)
        _, cache = gpt2_bridge.run_with_cache(tokens)
        resid = cache["blocks.11.hook_resid_post"]
        full = logit_readout(gpt2_bridge, resid, chunk_size=64)
        small = logit_readout(gpt2_bridge, resid, chunk_size=3)
        torch.testing.assert_close(full.values, ref, **TOL)
        torch.testing.assert_close(small.values, full.values, atol=1e-5, rtol=1e-5)

    def test_string_targets_and_top_tokens(self, gpt2_bridge):
        result = logit_lens(gpt2_bridge, PROMPT, targets=ANSWER)
        with torch.no_grad():
            ref = gpt2_bridge(PROMPT)[0, -1]
        answer = gpt2_bridge.to_single_token(ANSWER)
        trajectory = result.rank_trajectory(ANSWER)
        assert trajectory.shape == (13, 1)
        assert trajectory[-1, 0].item() == (ref > ref[answer]).sum().item()
        top = logit_lens(gpt2_bridge, PROMPT, top_k=3)
        decoded = top.top_tokens(gpt2_bridge.tokenizer)
        assert decoded["final_post"][0][0] == gpt2_bridge.tokenizer.decode([ref.argmax().item()])

    def test_parity_with_the_documented_recipe(self, gpt2_bridge):
        """``accumulated_resid(apply_ln=True) @ W_U + b_U`` is what the lens computes."""
        _, cache = gpt2_bridge.run_with_cache(PROMPT)
        stack = cache.accumulated_resid(apply_ln=True, pos_slice=-1)
        recipe = stack @ gpt2_bridge.W_U + gpt2_bridge.b_U
        result = logit_lens(cache)
        torch.testing.assert_close(result.values, recipe, **TOL)


class TestGPT2Compat:
    def test_final_entry_reproduces_processed_logits(self, gpt2_bridge_compat):
        tokens = gpt2_bridge_compat.to_tokens(PROMPT)
        ref, result = _exact_at_final_entry(gpt2_bridge_compat, tokens)
        # fold_ln moves the final-norm bias into b_U, so the bias path is exercised here.
        assert gpt2_bridge_compat.b_U.abs().max() > 0
        log_probs = logit_lens(gpt2_bridge_compat, tokens, positions=None, return_type="log_probs")
        torch.testing.assert_close(log_probs.values[-1], torch.log_softmax(ref, -1), **TOL)

    def test_centering_shifts_raw_logits_by_a_constant(self, gpt2_bridge, gpt2_bridge_compat):
        """center_unembed changes logits by a per-position constant; log-probs agree."""
        tokens = gpt2_bridge.to_tokens(PROMPT)
        raw = logit_lens(gpt2_bridge, tokens).values[-1]
        compat = logit_lens(gpt2_bridge_compat, tokens).values[-1]
        shift = compat - raw
        assert shift.std(dim=-1).max() < 1e-3
        torch.testing.assert_close(
            torch.log_softmax(compat, -1), torch.log_softmax(raw, -1), atol=1e-3, rtol=1e-3
        )


@pytest.fixture(scope="module")
def pythia():
    from transformer_lens.model_bridge import TransformerBridge

    return TransformerBridge.boot_transformers("EleutherAI/pythia-14m", device="cpu")


def test_pythia_parallel_residual_is_exact(pythia):
    tokens = pythia.to_tokens(PROMPT)
    _exact_at_final_entry(pythia, tokens)


@pytest.fixture(scope="module")
def tiny_gemma2():
    from transformers import Gemma2Config, Gemma2ForCausalLM

    from transformer_lens.model_bridge.sources import build_bridge_from_module

    torch.manual_seed(0)
    config = Gemma2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        sliding_window=16,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    hf = Gemma2ForCausalLM(config).eval()
    with torch.no_grad():
        hf.lm_head.weight.mul_(1000)  # push logits well past the softcap
    return build_bridge_from_module(
        hf, "Gemma2ForCausalLM", hf_config=config, dtype=torch.float32, device="cpu"
    )


def test_gemma2_softcap_is_applied(tiny_gemma2):
    tokens = torch.tensor([[1, 7, 11, 3]])
    ref, result = _exact_at_final_entry(tiny_gemma2, tokens)
    assert result.readout.applied_output_transform
    assert ref.abs().max() <= 30.0
    # Teeth: without the adapter transform the same path does not reproduce the model.
    _, cache = tiny_gemma2.run_with_cache(tokens)
    normed = tiny_gemma2.ln_final(cache["blocks.1.hook_resid_post"])
    uncapped = normed @ tiny_gemma2.W_U + tiny_gemma2.b_U
    assert (uncapped - ref).abs().max() > 1.0


def _tiny_cohere():
    from transformers import CohereConfig, CohereForCausalLM

    from transformer_lens.model_bridge.sources import build_bridge_from_module

    torch.manual_seed(0)
    config = CohereConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        logit_scale=0.25,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    return build_bridge_from_module(
        CohereForCausalLM(config).eval(),
        "CohereForCausalLM",
        hf_config=config,
        dtype=torch.float32,
        device="cpu",
    )


@pytest.mark.parametrize("compat", [False, True])
def test_cohere_logit_scale_is_applied_exactly_once(compat):
    """Compatibility mode folds logit_scale into W_U; the transform must not re-apply it."""
    model = _tiny_cohere()
    if compat:
        model.enable_compatibility_mode()
        assert model.adapter._logit_scale_already_folded
    tokens = torch.tensor([[1, 7, 11, 3]])
    with torch.no_grad():
        hf = model.original_model(tokens).logits
    ref, result = _exact_at_final_entry(model, tokens)
    torch.testing.assert_close(ref, hf, **TOL)
    assert result.readout.applied_output_transform
    # Teeth: scaling once more is what the double-application bug produced.
    assert (result.values[-1] * 0.25 - ref).abs().max() > 1e-2


@pytest.fixture(scope="module")
def tiny_nemotron_h():
    from transformers import AutoModelForCausalLM
    from transformers.models.nemotron_h import NemotronHConfig

    from transformer_lens.model_bridge.sources import build_bridge_from_module

    torch.manual_seed(0)
    cfg = NemotronHConfig(
        vocab_size=256,
        hidden_size=64,
        layers_block_type=["mamba", "attention", "mamba", "mlp"],
        num_attention_heads=4,
        num_key_value_heads=2,
        ssm_state_size=16,
        mamba_num_heads=4,
        mamba_head_dim=16,
        n_groups=2,
        conv_kernel=4,
        expand=2,
        intermediate_size=128,
        chunk_size=8,
    )
    cfg.architectures = ["NemotronHForCausalLM"]
    hf = AutoModelForCausalLM.from_config(cfg).to(torch.float32).eval()
    return build_bridge_from_module(
        hf, "NemotronHForCausalLM", hf_config=cfg, dtype=torch.float32, device="cpu"
    )


def test_hybrid_stack_reads_without_resid_mid(tiny_nemotron_h):
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    _, result = _exact_at_final_entry(tiny_nemotron_h, tokens)
    assert result.labels == ["0_pre", "1_pre", "2_pre", "3_pre", "final_post"]
    with pytest.raises(ValueError, match="incl_mid=True"):
        logit_lens(tiny_nemotron_h, tokens, incl_mid=True)


@pytest.mark.slow
def test_qwen2_5_rmsnorm_gqa_is_exact():
    from transformer_lens.model_bridge import TransformerBridge

    model = TransformerBridge.boot_transformers(
        "Qwen/Qwen2.5-0.5B", device="cpu", dtype=torch.float32
    )
    tokens = model.to_tokens(PROMPT)
    with torch.no_grad():
        ref = model(tokens)
    result = logit_lens(model, tokens, targets=ANSWER)
    torch.testing.assert_close(logit_lens(model, tokens).values[-1], ref[:, -1], **TOL)
    assert result.rank_trajectory(ANSWER).shape[0] == model.cfg.n_layers + 1
