"""Offline mask parity through the real Inspect driver and HF provider."""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    OPTConfig,
    OPTForCausalLM,
    PreTrainedTokenizerFast,
)

pytest.importorskip("inspect_ai")
pytestmark = pytest.mark.inspect


@pytest.fixture(scope="module", params=["gpt2", "llama", "opt"])
def inspect_models(tmp_path_factory, request):
    from transformer_lens.model_bridge.remote_bridge import RemoteBridge

    path = tmp_path_factory.mktemp("inspect_mask_model")
    torch.manual_seed(2)
    if request.param == "gpt2":
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=32,
                n_embd=16,
                n_layer=1,
                n_head=2,
                n_positions=32,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
                attn_implementation="eager",
            )
        )
    elif request.param == "llama":
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=32,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
                attn_implementation="eager",
            )
        )
    else:
        model = OPTForCausalLM(
            OPTConfig(
                vocab_size=32,
                hidden_size=16,
                ffn_dim=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                word_embed_proj_dim=16,
                max_position_embeddings=32,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
                attn_implementation="eager",
            )
        )
    model.eval()
    model.save_pretrained(path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(32)}, unk_token="3")),
        pad_token="0",
        bos_token="1",
        eos_token="2",
        unk_token="3",
    )
    tokenizer.save_pretrained(path)
    bridge = RemoteBridge.boot_inspect(str(path), device="cpu")
    yield bridge, model
    bridge.close()


@pytest.mark.parametrize("left_padding", [True, False])
def test_masked_logits_cache_and_loss(inspect_models, left_padding):
    bridge, model = inspect_models
    tokens = torch.tensor([[0, 0, 7, 8, 9] if left_padding else [7, 8, 9, 0, 0]])
    mask = torch.tensor([[0, 0, 1, 1, 1] if left_padding else [1, 1, 1, 0, 0]])
    kwargs = {"attention_mask": mask}
    if left_padding and model.config.model_type != "opt":
        kwargs["position_ids"] = (mask.cumsum(-1) - 1).clamp_min(0)
    with torch.no_grad():
        expected = model(tokens, **kwargs).logits
    actual, cache = bridge.run_with_cache(
        tokens, attention_mask=mask, names_filter=["blocks.0.hook_out"]
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    alone, alone_cache = bridge.run_with_cache(
        torch.tensor([[7, 8, 9]]), names_filter=["blocks.0.hook_out"]
    )
    valid = mask[0].bool()
    torch.testing.assert_close(actual[:, valid], alone, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        cache["blocks.0.hook_out"][:, valid], alone_cache["blocks.0.hook_out"], rtol=1e-5, atol=1e-6
    )
    loss = bridge.forward(tokens, attention_mask=mask, return_type="loss")
    alone_loss = bridge.forward(torch.tensor([[7, 8, 9]]), return_type="loss")
    torch.testing.assert_close(loss, alone_loss)
    labeled_loss = bridge.forward(tokens, labels=tokens, attention_mask=mask, return_type="loss")
    torch.testing.assert_close(labeled_loss, alone_loss)


def test_all_ones_mask_preserves_unmasked_forward(inspect_models):
    bridge, _ = inspect_models
    tokens = torch.tensor([[7, 8, 9]])
    torch.testing.assert_close(
        bridge.forward(tokens, attention_mask=torch.ones_like(tokens)),
        bridge.forward(tokens),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("mask", [[[1, 1]], [[1, 0.5, 1]], [[0, 0, 0]], [[[1, 1, 1]]]])
def test_invalid_masks_fail_at_public_and_provider_boundaries(inspect_models, mask):
    from inspect_ai.model import GenerateConfig

    bridge, _ = inspect_models
    with pytest.raises(ValueError, match="attention_mask"):
        bridge.forward(torch.tensor([[7, 8, 9]]), attention_mask=torch.tensor(mask))
    with pytest.raises(ValueError, match="attention_mask"):
        bridge._driver._model.api._generate_capture(
            [], {"input_ids": [7, 8, 9], "attention_mask": mask}, GenerateConfig()
        )


@pytest.mark.parametrize("dtype", [torch.bool, torch.float32, torch.bfloat16])
def test_binary_mask_dtypes(inspect_models, dtype):
    bridge, _ = inspect_models
    tokens = torch.tensor([[0, 7, 8]])
    mask = torch.tensor([[0, 1, 1]])
    torch.testing.assert_close(
        bridge.forward(tokens, attention_mask=mask.to(dtype)),
        bridge.forward(tokens, attention_mask=mask),
        rtol=0,
        atol=0,
    )


def test_masked_capture_and_intervention(inspect_models):
    bridge, _ = inspect_models
    tokens = torch.tensor([[0, 0, 7, 8, 9]])
    mask = torch.tensor([[0, 0, 1, 1, 1]])
    names = ["blocks.0.hook_out", "blocks.0.attn.hook_pattern"]
    intervention = {"blocks.0.attn.hook_out": {"op": "scale", "factor": 0.5}}
    padded, cache = bridge.run_with_cache(
        tokens, attention_mask=mask, names_filter=names, intervene=intervention
    )
    alone, alone_cache = bridge.run_with_cache(
        torch.tensor([[7, 8, 9]]), names_filter=names, intervene=intervention
    )
    torch.testing.assert_close(padded[:, 2:], alone, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(cache[names[0]][:, 2:], alone_cache[names[0]], rtol=1e-5, atol=1e-6)
    pattern = cache[names[1]]
    torch.testing.assert_close(pattern[:, :, 2:, 2:], alone_cache[names[1]], rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(pattern[:, :, 2:, :2]) == 0


def test_provider_completion_uses_last_attended_token(inspect_models):
    from inspect_ai.model import GenerateConfig

    bridge, model = inspect_models
    api = bridge._driver._model.api
    with torch.no_grad():
        logits = model(torch.tensor([[7, 8, 9]])).logits[0, -1]
    next_id = int(logits.argmax())
    output = api._generate_capture(
        [],
        {"input_ids": [7, 8, 9, 0, 0], "attention_mask": [1, 1, 1, 0, 0]},
        GenerateConfig(logprobs=True, top_logprobs=3),
    )
    assert output.completion == api._tokenizer.decode([next_id])
    entry = output.choices[0].logprobs.content[0]
    assert entry.logprob == pytest.approx(float(logits.log_softmax(-1)[next_id]), abs=1e-6)
