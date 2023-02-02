import torch
from fancy_einsum import einsum
from transformer_lens import HookedTransformer, ActivationCache
from torchtyping import TensorType as TT

# TODO: replace with actual unit tests
@torch.set_grad_enabled(False)
def test_logit_attrs_matches_reference_code():

    # Load gpt2-small
    model = HookedTransformer.from_pretrained(
        "solu-2l",
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
    )

    # Create IOI prompts
    prompt_format = [
        "When John and Mary went to the shops,{} gave the bag to",
        "When Tom and James went to the park,{} gave the ball to",
        "When Dan and Sid went to the shops,{} gave an apple to",
        "After Martin and Amy went to the park,{} gave a drink to",
    ]
    names = [
        (" Mary", " John"),
        (" Tom", " James"),
        (" Dan", " Sid"),
        (" Martin", " Amy"),
    ]
    # List of prompts
    prompts = []
    # List of answers, in the format (correct, incorrect)
    answers = []
    # List of the token (ie an integer) corresponding to each answer, in the format (correct_token, incorrect_token)
    answer_tokens = []
    for i in range(len(prompt_format)):
        for j in range(2):
            answers.append((names[i][j], names[i][1 - j]))
            answer_tokens.append(
                (
                    model.to_single_token(answers[-1][0]),
                    model.to_single_token(answers[-1][1]),
                )
            )
            # Insert the *incorrect* answer to the prompt, making the correct answer the indirect object.
            prompts.append(prompt_format[i].format(answers[-1][1]))
    answer_tokens = torch.tensor(answer_tokens)

    tokens = model.to_tokens(prompts, prepend_bos=True)

    # Run the model and cache all activations
    original_logits, cache = model.run_with_cache(tokens)

    # Reference notebook code
    answer_residual_directions = model.tokens_to_residual_directions(answer_tokens)
    logit_diff_directions = answer_residual_directions[:, 0] - answer_residual_directions[:, 1]

    def residual_stack_to_logit_diff(residual_stack: TT["components", "batch", "d_model"], cache: ActivationCache) -> float:
        scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer = -1, pos_slice=-1)
        return einsum("... batch d_model, batch d_model -> ...", scaled_residual_stack, logit_diff_directions)/len(prompts)

    # Get accumulated resid
    accumulated_residual, labels = cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)

    # Get reference logit diffs
    ref_ave_logit_diffs = residual_stack_to_logit_diff(accumulated_residual, cache)

    # Get our logit diffs
    logit_diffs = cache.logit_attrs(accumulated_residual, pos_slice=-1, tokens=answer_tokens[:,0], incorrect_tokens=answer_tokens[:,1])
    ave_logit_diffs = logit_diffs.mean(dim=-1)

    assert (ref_ave_logit_diffs - ave_logit_diffs).abs().sum() < 4e-5
