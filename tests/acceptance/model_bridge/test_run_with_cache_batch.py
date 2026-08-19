"""Tests that batched run_with_cache and run_with_hooks produce correct results.

The default right-padded layout places each row's last real token at a
different batch position. These tests compare each row at that position rather
than assuming that every real sequence ends at index ``-1``.
"""

import torch


def test_run_with_cache_batch_matches_individual(gpt2_bridge):
    """Batched run_with_cache logits at the last real token should match per-prompt runs."""
    prompts = [
        "Hello, my dog is cute",
        "This is a much longer text. Hello, my cat is cute",
    ]
    last_token_positions = [gpt2_bridge.to_tokens(prompt).shape[1] - 1 for prompt in prompts]

    # Individual runs
    individual_logits = []
    for p in prompts:
        logits, _ = gpt2_bridge.run_with_cache(p)
        individual_logits.append(logits[0, -1, :])

    # Batched run
    batched_logits, _ = gpt2_bridge.run_with_cache(prompts)
    for i in range(len(prompts)):
        batched_last = batched_logits[i, last_token_positions[i], :]
        assert torch.allclose(
            individual_logits[i], batched_last, atol=1e-4
        ), f"Prompt {i} logit mismatch between individual and batched run_with_cache"


def test_run_with_hooks_batch_matches_individual(gpt2_bridge):
    """Batched run_with_hooks should produce the same hook values as per-prompt runs
    (for the last real token position of each sequence)."""
    prompts = [
        "Hello, my dog is cute",
        "This is a much longer text. Hello, my cat is cute",
    ]
    last_token_positions = [gpt2_bridge.to_tokens(prompt).shape[1] - 1 for prompt in prompts]

    # Capture resid_post at last layer for last token
    captured_individual = []

    def capture_individual(tensor, hook):
        # Last token's residual
        captured_individual.append(tensor[0, -1, :].detach().clone())

    for p in prompts:
        gpt2_bridge.run_with_hooks(
            p,
            fwd_hooks=[("blocks.11.hook_resid_post", capture_individual)],
        )

    # Batched run
    captured_batched = []

    def capture_batched(tensor, hook):
        for i in range(tensor.shape[0]):
            captured_batched.append(tensor[i, last_token_positions[i], :].detach().clone())

    gpt2_bridge.run_with_hooks(
        prompts,
        fwd_hooks=[("blocks.11.hook_resid_post", capture_batched)],
    )

    assert len(captured_individual) == len(captured_batched) == len(prompts)
    for i in range(len(prompts)):
        assert torch.allclose(
            captured_individual[i], captured_batched[i], atol=1e-4
        ), f"Prompt {i} hook value mismatch between individual and batched run_with_hooks"
