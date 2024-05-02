import torch
from fancy_einsum import einsum

from transformer_lens import HookedTransformer

# Create IOI prompts
ioi_prompt_formats = [
    "When John and Mary went to the shops,{} gave the bag to",
    "When Tom and James went to the park,{} gave the ball to",
    "When Dan and Sid went to the shops,{} gave an apple to",
    "After Martin and Amy went to the park,{} gave a drink to",
]
ioi_names = [
    (" Mary", " John"),
    (" Tom", " James"),
    (" Dan", " Sid"),
    (" Martin", " Amy"),
]


def get_ioi_tokens_and_answer_tokens(model):
    # List of prompts
    prompts = []
    # List of answers, in the format (correct, incorrect)
    answers = []
    # List of the token (ie an integer) corresponding to each answer, in the format (correct_token, incorrect_token)
    answer_tokens = []
    for i in range(len(ioi_prompt_formats)):
        for j in range(2):
            answers.append((ioi_names[i][j], ioi_names[i][1 - j]))
            answer_tokens.append(
                (
                    model.to_single_token(answers[-1][0]),
                    model.to_single_token(answers[-1][1]),
                )
            )
            # Insert the *incorrect* answer to the prompt, making the correct answer the indirect object.
            prompts.append(ioi_prompt_formats[i].format(answers[-1][1]))
    answer_tokens = torch.tensor(answer_tokens)

    tokens = model.to_tokens(prompts, prepend_bos=True)

    return tokens, answer_tokens


def load_model(name):
    return HookedTransformer.from_pretrained(
        name,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
    )


@torch.no_grad
def test_logit_attrs_matches_reference_code():
    # Load solu-2l
    model = load_model("solu-2l")

    tokens, answer_tokens = get_ioi_tokens_and_answer_tokens(model)

    # Run the model and cache all activations
    _, cache = model.run_with_cache(tokens)

    # Get accumulated resid
    accumulated_residual = cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1)

    # Get ref ave logit diffs (cribbed notebook code)
    answer_residual_directions = model.tokens_to_residual_directions(answer_tokens)
    logit_diff_directions = answer_residual_directions[:, 0] - answer_residual_directions[:, 1]
    scaled_residual_stack = cache.apply_ln_to_stack(accumulated_residual, layer=-1, pos_slice=-1)
    ref_ave_logit_diffs = einsum(
        "... batch d_model, batch d_model -> ...",
        scaled_residual_stack,
        logit_diff_directions,
    ) / len(tokens)

    # Get our ave logit diffs
    logit_diffs = cache.logit_attrs(
        accumulated_residual,
        pos_slice=-1,
        tokens=answer_tokens[:, 0],
        incorrect_tokens=answer_tokens[:, 1],
    )
    ave_logit_diffs = logit_diffs.mean(dim=-1)

    assert torch.isclose(ref_ave_logit_diffs, ave_logit_diffs, atol=1e-7).all()


@torch.no_grad
def test_logit_attrs_works_for_all_input_shapes():
    # Load solu-2l
    model = load_model("solu-2l")

    tokens, answer_tokens = get_ioi_tokens_and_answer_tokens(model)

    # Run the model and cache all activations
    _, cache = model.run_with_cache(tokens)

    # Get accumulated resid
    accumulated_residual = cache.accumulated_resid(
        layer=-1, incl_mid=True, pos_slice=-1, return_labels=False
    )

    # Get ref logit diffs (cribbed notebook code)
    answer_residual_directions = model.tokens_to_residual_directions(answer_tokens)
    logit_diff_directions = answer_residual_directions[:, 0] - answer_residual_directions[:, 1]
    scaled_residual_stack = cache.apply_ln_to_stack(accumulated_residual, layer=-1, pos_slice=-1)
    ref_logit_diffs = einsum(
        "... d_model, ... d_model -> ...", scaled_residual_stack, logit_diff_directions
    )

    # All tokens
    logit_diffs = cache.logit_attrs(
        accumulated_residual,
        pos_slice=-1,
        tokens=answer_tokens[:, 0],
        incorrect_tokens=answer_tokens[:, 1],
    )
    assert torch.isclose(ref_logit_diffs, logit_diffs).all()

    # Single token
    batch = -1
    logit_diffs = cache.logit_attrs(
        accumulated_residual,
        batch_slice=batch,
        pos_slice=-1,
        tokens=answer_tokens[batch, 0],
        incorrect_tokens=answer_tokens[batch, 1],
    )
    assert torch.isclose(ref_logit_diffs[:, batch], logit_diffs).all()

    # Single token (int)
    batch = -1
    logit_diffs = cache.logit_attrs(
        accumulated_residual,
        batch_slice=batch,
        pos_slice=-1,
        tokens=int(answer_tokens[batch, 0]),
        incorrect_tokens=int(answer_tokens[batch, 1]),
    )
    assert torch.isclose(ref_logit_diffs[:, batch], logit_diffs).all()

    # Single token (str)
    batch = -1
    logit_diffs = cache.logit_attrs(
        accumulated_residual,
        batch_slice=batch,
        pos_slice=-1,
        tokens=model.to_string(answer_tokens[batch, 0]),
        incorrect_tokens=model.to_string(answer_tokens[batch, 1]),
    )
    assert torch.isclose(ref_logit_diffs[:, batch], logit_diffs).all()

    # Single token and residual stack without batch dim
    batch = -1
    logit_diffs = cache.logit_attrs(
        accumulated_residual[:, batch, :],
        has_batch_dim=False,
        batch_slice=batch,
        pos_slice=-1,
        tokens=answer_tokens[batch, 0],
        incorrect_tokens=answer_tokens[batch, 1],
    )
    assert torch.isclose(ref_logit_diffs[:, batch], logit_diffs).all()

    # Array slice of tokens
    batch = [2, 5, 7]
    logit_diffs = cache.logit_attrs(
        accumulated_residual,
        batch_slice=batch,
        pos_slice=-1,
        tokens=answer_tokens[batch, 0],
        incorrect_tokens=answer_tokens[batch, 1],
    )
    assert torch.isclose(ref_logit_diffs[:, batch], logit_diffs).all()


@torch.no_grad
def test_accumulated_resid_with_apply_ln():
    # Load solu-2l
    model = load_model("solu-2l")

    tokens, _ = get_ioi_tokens_and_answer_tokens(model)

    # Run the model and cache all activations
    _, cache = model.run_with_cache(tokens)

    # Get accumulated resid and apply ln seperately (cribbed notebook code)
    accumulated_residual = cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1)
    ref_scaled_residual_stack = cache.apply_ln_to_stack(
        accumulated_residual, layer=-1, pos_slice=-1
    )

    # Get scaled_residual_stack using apply_ln parameter
    scaled_residual_stack = cache.accumulated_resid(
        layer=-1, incl_mid=True, pos_slice=-1, apply_ln=True
    )

    assert torch.isclose(ref_scaled_residual_stack, scaled_residual_stack, atol=1e-7).all()


@torch.no_grad
def test_decompose_resid_with_apply_ln():
    # Load solu-2l
    model = load_model("solu-2l")

    tokens, _ = get_ioi_tokens_and_answer_tokens(model)

    # Run the model and cache all activations
    _, cache = model.run_with_cache(tokens)

    # Get decomposed resid and apply ln seperately (cribbed notebook code)
    per_layer_residual = cache.decompose_resid(layer=-1, pos_slice=-1)
    ref_scaled_residual_stack = cache.apply_ln_to_stack(per_layer_residual, layer=-1, pos_slice=-1)

    # Get scaled_residual_stack using apply_ln parameter
    scaled_residual_stack = cache.decompose_resid(layer=-1, pos_slice=-1, apply_ln=True)

    assert torch.isclose(ref_scaled_residual_stack, scaled_residual_stack, atol=1e-7).all()


@torch.no_grad
def test_stack_head_results_with_apply_ln():
    # Load solu-2l
    model = load_model("solu-2l")

    tokens, _ = get_ioi_tokens_and_answer_tokens(model)

    # Run the model and cache all activations
    _, cache = model.run_with_cache(tokens)

    # Get per head resid stack and apply ln seperately (cribbed notebook code)
    per_head_residual = cache.stack_head_results(layer=-1, pos_slice=-1)
    ref_scaled_residual_stack = cache.apply_ln_to_stack(per_head_residual, layer=-1, pos_slice=-1)

    # Get scaled_residual_stack using apply_ln parameter
    scaled_residual_stack = cache.stack_head_results(layer=-1, pos_slice=-1, apply_ln=True)

    assert torch.isclose(ref_scaled_residual_stack, scaled_residual_stack, atol=1e-7).all()


@torch.no_grad
def test_stack_neuron_results_with_apply_ln():
    # Load solu-2l
    model = load_model("solu-2l")

    tokens, _ = get_ioi_tokens_and_answer_tokens(model)

    # Run the model and cache all activations
    _, cache = model.run_with_cache(tokens)

    # Get neuron result stack and apply ln seperately
    neuron_result_stack = cache.stack_neuron_results(layer=-1, pos_slice=-1)
    ref_scaled_residual_stack = cache.apply_ln_to_stack(neuron_result_stack, layer=-1, pos_slice=-1)

    # Get scaled_residual_stack using apply_ln parameter
    scaled_residual_stack = cache.stack_neuron_results(layer=-1, pos_slice=-1, apply_ln=True)

    assert torch.isclose(ref_scaled_residual_stack, scaled_residual_stack, atol=1e-7).all()
