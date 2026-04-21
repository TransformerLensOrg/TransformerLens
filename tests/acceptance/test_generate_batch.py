"""Tests that batched HookedTransformer generation matches individual generation."""


def test_ht_generate_batch_matches_individual(gpt2_model):
    """Batched generate() should match one-by-one generate() for left-padded inputs."""
    prompts = ["Hello, my dog is cute", "This is a much longer text. Hello, my cat is cute"]
    individual_outputs = [
        gpt2_model.generate(p, verbose=False, do_sample=False) for p in prompts
    ]

    batched_outputs = gpt2_model.generate(prompts, verbose=False, do_sample=False)
    for i, prompt in enumerate(prompts):
        assert individual_outputs[i] == batched_outputs[i], (
            f"Prompt {i} mismatch:\n  individual: {individual_outputs[i]}\n  batched: {batched_outputs[i]}"
        )


def test_ht_generate_batch_without_kv_cache(gpt2_model):
    """Same test with use_past_kv_cache=False."""
    prompts = ["Hello, my dog is cute", "This is a much longer text. Hello, my cat is cute"]
    individual_outputs = [
        gpt2_model.generate(p, verbose=False, do_sample=False, use_past_kv_cache=False)
        for p in prompts
    ]

    batched_outputs = gpt2_model.generate(
        prompts, verbose=False, do_sample=False, use_past_kv_cache=False
    )
    for i, prompt in enumerate(prompts):
        assert individual_outputs[i] == batched_outputs[i], (
            f"Prompt {i} mismatch:\n  individual: {individual_outputs[i]}\n  batched: {batched_outputs[i]}"
        )
