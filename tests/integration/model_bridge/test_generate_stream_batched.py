"""Batched ``generate_stream(return_type="str")`` decodes every row of the batch.

The streaming decoder used to decode ``tokens[0]`` only, so a batched call
yielded the first sequence's text for the whole batch and the other rows were
never observable. See ``generate()``, which decodes each row and unwraps a
single-row batch — ``generate_stream`` now matches that contract.
"""

PROMPTS = ["The capital of France is", "My favourite colour is"]


def test_batched_stream_yields_one_string_per_row(distilgpt2_bridge):
    chunks = list(
        distilgpt2_bridge.generate_stream(
            PROMPTS,
            max_new_tokens=4,
            max_tokens_per_yield=2,
            do_sample=False,
            verbose=False,
            return_type="str",
        )
    )

    assert chunks, "expected at least one yield"
    for chunk in chunks:
        assert isinstance(chunk, list)
        assert len(chunk) == len(PROMPTS)
        assert all(isinstance(text, str) for text in chunk)

    # The first yield carries the input tokens, so each row must echo its own prompt.
    for prompt, streamed in zip(PROMPTS, chunks[0]):
        assert prompt in streamed


def test_single_prompt_stream_stays_a_bare_string(distilgpt2_bridge):
    """A one-element batch keeps the scalar contract shared with generate()."""
    chunks = list(
        distilgpt2_bridge.generate_stream(
            [PROMPTS[0]],
            max_new_tokens=2,
            do_sample=False,
            verbose=False,
            return_type="str",
        )
    )

    assert chunks
    assert all(isinstance(chunk, str) for chunk in chunks)
