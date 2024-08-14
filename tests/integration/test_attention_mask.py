import torch

from transformer_lens import utils
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def test_attention_mask():
    # Verify the attention mask attends properly, including for low attention scores
    cfg = HookedTransformerConfig(
        d_head=1,
        d_model=12,
        d_vocab=2,
        n_ctx=5,
        n_layers=1,
        attn_only=True,
        attention_dir="causal",
    )
    model = HookedTransformer(cfg)
    input_length = 5
    input = torch.ones((1, input_length), dtype=torch.int64)
    layer = 0
    low_attn_score = 1e-6
    ones_input_matrix = torch.ones((input_length, input_length))
    masked = torch.triu(ones_input_matrix, diagonal=1).bool()

    def attn_scores_hook(attn_scores, hook):
        assert torch.all(
            attn_scores[:, :, masked] == float("-inf")
        ), "Attention scores excluded by the mask are not being set to -inf"

        # Set low attention scores that are attended to by the mask
        attn_scores[:, :, ~masked] = low_attn_score

        return attn_scores

    def attn_hook(attn, hook):
        assert torch.all(attn[:, :, masked] == 0), "Attention pattern attends outside the mask"

        return attn

    fwd_hooks = [
        (utils.get_act_name("attn_scores", layer), attn_scores_hook),
        (utils.get_act_name("attn", layer), attn_hook),
    ]

    model.run_with_hooks(input, fwd_hooks=fwd_hooks)


def test_masked_tokens():
    """Test that masking tokens works as expected."""
    MODEL = "solu-1l"
    prompts = [
        "Hello world!",
        "The quick brown fox jumps over the lazy dog.",
    ]
    model = HookedTransformer.from_pretrained(MODEL)
    tokens = model.to_tokens(prompts)

    # Part 1: If the mask is all ones, the output should be the same as if there was no mask.
    full_mask = torch.ones_like(tokens)
    no_mask_out = model(tokens)
    full_mask_out = model(tokens, attention_mask=full_mask)
    assert torch.allclose(no_mask_out, full_mask_out), "Full mask should be equivalent to no mask"

    # Part 2: If the mask has a column of zeros, the output should be the same as if that token
    # position was removed from the input.
    remove_tok_idx = 2
    edited_tokens = torch.cat([tokens[:, :remove_tok_idx], tokens[:, remove_tok_idx + 1 :]], dim=1)
    edited_mask = full_mask.clone()
    edited_mask[:, remove_tok_idx] = 0
    edited_no_mask_out = model(edited_tokens)
    edited_mask_out = model(tokens, attention_mask=edited_mask)
    edited_mask_out = torch.cat(
        [edited_mask_out[:, :remove_tok_idx], edited_mask_out[:, remove_tok_idx + 1 :]], dim=1
    )
    assert torch.allclose(
        edited_no_mask_out, edited_mask_out, atol=1e-4
    ), "Edited mask should be equivalent to no mask"
