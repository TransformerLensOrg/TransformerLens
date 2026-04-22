import torch

from transformer_lens.HookedTransformer import HookedTransformer


def test_cross_entropy_attention_mask():
    """Check that adding a bunch of masked tokens to the input does not change the loss."""
    MODEL = "solu-1l"
    model = HookedTransformer.from_pretrained(MODEL)

    # Step 1: Get the default loss on a prompt
    prompt = ["The quick brown fox jumps over the lazy dog."]
    default_tokens = model.to_tokens(prompt)
    default_attention_mask = torch.ones_like(default_tokens)
    default_loss = model(default_tokens, return_type="loss")
    ones_mask_loss = model(
        default_tokens, attention_mask=default_attention_mask, return_type="loss"
    )
    assert torch.allclose(default_loss, ones_mask_loss, atol=1e-6)

    # Step 2: Get the loss when we add some extra tokens to the input and set their attention mask
    # to zero
    extra_prompt = ["Lorem ipsum dolor sit amet, consectetur adipiscing elit."]
    extra_tokens = model.to_tokens(extra_prompt)
    extra_zeros_attention_mask = torch.zeros_like(extra_tokens)

    combined_tokens = torch.cat([default_tokens, extra_tokens], dim=1)
    combined_attention_mask = torch.cat([default_attention_mask, extra_zeros_attention_mask], dim=1)
    combined_masked_loss = model(
        combined_tokens, attention_mask=combined_attention_mask, return_type="loss"
    )
    assert torch.allclose(default_loss, combined_masked_loss)
