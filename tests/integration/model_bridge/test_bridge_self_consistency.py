"""Bridge self-consistency tests (formerly bridge-vs-HookedTransformer comparison).

The numeric parity content of the old file lives in golden-anchored tests now
(test_weight_processing.py, compatibility/test_bridge_cache_behavior.py,
test_bridge_wiring_integration.py). What remains here are the properties that
were only expressible as cross-model checks and are really self-consistency
contracts: deterministic generation and batch-vs-individual equivalence.
"""

import pytest
import torch


@pytest.mark.slow
def test_generation_deterministic_and_input_form_invariant(distilgpt2_bridge_compat):
    """Greedy generation is reproducible and identical for token vs string input."""
    bridge = distilgpt2_bridge_compat
    prompt = "The future of AI"
    tokens = bridge.to_tokens(prompt)

    with torch.no_grad():
        from_tokens_1 = bridge.generate(tokens, max_new_tokens=5, temperature=0.0, do_sample=False)
        from_tokens_2 = bridge.generate(tokens, max_new_tokens=5, temperature=0.0, do_sample=False)

    assert torch.equal(from_tokens_1, from_tokens_2), "Greedy generation is not deterministic"

    with torch.no_grad():
        from_string = bridge.generate(prompt, max_new_tokens=5, temperature=0.0, do_sample=False)

    # The token path decodes the prepended BOS marker; the string path omits it.
    decoded = bridge.to_string(from_tokens_1[0])
    bos = bridge.tokenizer.bos_token or ""
    decoded = decoded.removeprefix(bos)
    assert decoded == from_string, (
        f"Token-input and string-input generation diverge:\n"
        f"  tokens: {decoded!r}\n  string: {from_string!r}"
    )


def test_batch_processing_matches_individual(distilgpt2_bridge_compat):
    """Right-padded batch logits must match each text's individual forward pass."""
    bridge = distilgpt2_bridge_compat
    texts = ["First text", "Second text", "Third text for batch"]

    tokens_list = [bridge.to_tokens(text)[0] for text in texts]
    max_len = max(len(tokens) for tokens in tokens_list)

    padded_tokens = []
    for tokens in tokens_list:
        if len(tokens) < max_len:
            padding = torch.full(
                (max_len - len(tokens),), bridge.tokenizer.pad_token_id or 0, dtype=tokens.dtype
            )
            tokens = torch.cat([tokens, padding])
        padded_tokens.append(tokens)
    batch_tokens = torch.stack(padded_tokens)

    with torch.no_grad():
        batch_logits = bridge(batch_tokens, return_type="logits")

    assert batch_logits.shape[0] == len(texts), "Batch size should match input"

    # Each row's logits over its true (unpadded) positions must match the
    # individual forward pass — padding must not leak into earlier positions.
    for i, tokens in enumerate(tokens_list):
        with torch.no_grad():
            single_logits = bridge(tokens.unsqueeze(0), return_type="logits")
        row_diff = (batch_logits[i, : len(tokens)] - single_logits[0]).abs().max()
        assert row_diff < 1e-3, f"Row {i} batch/individual divergence: {row_diff:.6f}"
