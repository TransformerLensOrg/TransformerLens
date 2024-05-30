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
