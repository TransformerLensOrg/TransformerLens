"""attribute_utils.

This module contains utility functions related to exploratory analysis
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from rich import print as rprint


def test_prompt(
    prompt: str,
    answer: Union[str, list[str]],
    model,  # Can't give type hint due to circular imports
    prepend_space_to_answer: bool = True,
    print_details: bool = True,
    prepend_bos: Optional[bool] = None,
    top_k: int = 10,
) -> None:
    """Test if the Model Can Give the Correct Answer to a Prompt.

    Intended for exploratory analysis. Prints out the performance on the answer (rank, logit, prob),
    as well as the top k tokens. Works for multi-token prompts and multi-token answers.

    Warning:

    This will print the results (it does not return them).

    Examples:

    >>> from transformer_lens import HookedTransformer, utils
    >>> model = HookedTransformer.from_pretrained("tiny-stories-1M")
    Loaded pretrained model tiny-stories-1M into HookedTransformer

    >>> prompt = "Why did the elephant cross the"
    >>> answer = "road"
    >>> utils.test_prompt(prompt, answer, model)
    Tokenized prompt: ['<|endoftext|>', 'Why', ' did', ' the', ' elephant', ' cross', ' the']
    Tokenized answer: [' road']
    Performance on answer token:
    Rank: 2        Logit: 14.24 Prob:  3.51% Token: | road|
    Top 0th token. Logit: 14.51 Prob:  4.59% Token: | ground|
    Top 1th token. Logit: 14.41 Prob:  4.18% Token: | tree|
    Top 2th token. Logit: 14.24 Prob:  3.51% Token: | road|
    Top 3th token. Logit: 14.22 Prob:  3.45% Token: | car|
    Top 4th token. Logit: 13.92 Prob:  2.55% Token: | river|
    Top 5th token. Logit: 13.79 Prob:  2.25% Token: | street|
    Top 6th token. Logit: 13.77 Prob:  2.21% Token: | k|
    Top 7th token. Logit: 13.75 Prob:  2.16% Token: | hill|
    Top 8th token. Logit: 13.64 Prob:  1.92% Token: | swing|
    Top 9th token. Logit: 13.46 Prob:  1.61% Token: | park|
    Ranks of the answer tokens: [(' road', 2)]

    Args:
        prompt:
            The prompt string, e.g. "Why did the elephant cross the".
        answer:
            The answer, e.g. "road". Note that if you set prepend_space_to_answer to False, you need
            to think about if you have a space before the answer here (as e.g. in this example the
            answer may really be " road" if the prompt ends without a trailing space). If this is a
            list of strings, then we only look at the next-token completion, and we compare them all
            as possible model answers.
        model:
            The model.
        prepend_space_to_answer:
            Whether or not to prepend a space to the answer. Note this will only ever prepend a
            space if the answer doesn't already start with one.
        print_details:
            Print the prompt (as a string but broken up by token), answer and top k tokens (all
            with logit, rank and probability).
        prepend_bos:
            Overrides self.cfg.default_prepend_bos if set. Whether to prepend
            the BOS token to the input (applicable when input is a string). Models generally learn
            to use the BOS token as a resting place for attention heads (i.e. a way for them to be
            "turned off"). This therefore often improves performance slightly.
        top_k:
            Top k tokens to print details of (when print_details is set to True).

    Returns:
        None (just prints the results directly).
    """
    answers = [answer] if isinstance(answer, str) else answer
    n_answers = len(answers)
    using_multiple_answers = n_answers > 1
    if prepend_space_to_answer:
        answers = [answer if answer.startswith(" ") else " " + answer for answer in answers]
    # GPT-2 often treats the first token weirdly, so lets give it a resting position
    prompt_tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    answer_tokens = model.to_tokens(answers, prepend_bos=False)
    # If we have multiple answers, we're only allowed a single token generation
    if using_multiple_answers:
        answer_tokens = answer_tokens[:, :1]
    # Deal with case where answers is a list of strings
    prompt_tokens = prompt_tokens.repeat(answer_tokens.shape[0], 1)
    tokens = torch.cat((prompt_tokens, answer_tokens), dim=1)
    prompt_str_tokens = model.to_str_tokens(prompt, prepend_bos=prepend_bos)
    answer_str_tokens_list = [model.to_str_tokens(answer, prepend_bos=False) for answer in answers]
    prompt_length = len(prompt_str_tokens)
    answer_length = 1 if using_multiple_answers else len(answer_str_tokens_list[0])

    if print_details:
        print("Tokenized prompt:", prompt_str_tokens)
        if using_multiple_answers:
            print("Tokenized answers:", answer_str_tokens_list)
        else:
            print("Tokenized answer:", answer_str_tokens_list[0])
    logits = model(tokens)
    probs = logits.softmax(dim=-1)
    answer_ranks = []

    for index in range(prompt_length, prompt_length + answer_length):
        # Get answer tokens for this sequence position
        answer_tokens = tokens[:, index]
        answer_str_tokens = [a[index - prompt_length] for a in answer_str_tokens_list]
        # Offset by 1 because models predict the NEXT token
        token_probs = probs[:, index - 1]
        sorted_token_probs, sorted_token_positions = token_probs.sort(descending=True)
        answer_token_ranks = sorted_token_positions.argsort(-1)[
            range(n_answers), answer_tokens.cpu()
        ].tolist()
        answer_ranks.append(
            [
                (answer_str_token, answer_token_rank)
                for answer_str_token, answer_token_rank in zip(
                    answer_str_tokens, answer_token_ranks
                )
            ]
        )
        if print_details:
            # String formatting syntax - the first number gives the number of characters to pad to, the second number gives the number of decimal places.
            # rprint gives rich text printing
            rprint(
                f"Performance on answer token{'s' if n_answers > 1 else ''}:\n"
                + "\n".join(
                    [
                        f"[b]Rank: {answer_token_ranks[i]: <8} Logit: {logits[i, index-1, answer_tokens[i]].item():5.2f} Prob: {token_probs[i, answer_tokens[i]].item():6.2%} Token: |{answer_str_tokens[i]}|[/b]"
                        for i in range(n_answers)
                    ]
                )
            )
            for i in range(top_k):
                print(
                    f"Top {i}th token. Logit: {logits[0, index-1, sorted_token_positions[0, i]].item():5.2f} Prob: {sorted_token_probs[0, i].item():6.2%} Token: |{model.to_string(sorted_token_positions[0, i])}|"
                )
    # If n_answers = 1 then unwrap answer ranks, so printed output matches original version of function
    if not using_multiple_answers:
        single_answer_ranks = [r[0] for r in answer_ranks]
        rprint(f"[b]Ranks of the answer tokens:[/b] {single_answer_ranks}")
    else:
        rprint(f"[b]Ranks of the answer tokens:[/b] {answer_ranks}")


try:
    import pytest

    # Note: Docstring won't be tested with PyTest (it's ignored), as it thinks this is a regular unit
    # test (because its name is prefixed `test_`).
    pytest.mark.skip(test_prompt)
except ModuleNotFoundError:
    pass  # disregard if pytest not in env
