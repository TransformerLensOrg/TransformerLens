import pytest
import torch
from fancy_einsum import einsum

from transformer_lens import HookedTransformer, utils
from transformer_lens.utils import Slice

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

    assert torch.isclose(ref_ave_logit_diffs, ave_logit_diffs, atol=1.1e-7).all()


@torch.no_grad
def test_logit_accumulated_resid_on_last_layer_variants():
    model = load_model("solu-2l")
    tokens, answer_tokens = get_ioi_tokens_and_answer_tokens(model)
    _, cache = model.run_with_cache(tokens)

    accumulated_resid = cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1)
    assert torch.equal(
        accumulated_resid,
        cache.accumulated_resid(layer=model.cfg.n_layers, incl_mid=True, pos_slice=-1),
    )

    assert torch.equal(
        accumulated_resid, cache.accumulated_resid(layer=None, incl_mid=True, pos_slice=-1)
    )


@torch.no_grad
def test_logit_accumulated_resid_without_mid():
    model = load_model("solu-2l")
    tokens, answer_tokens = get_ioi_tokens_and_answer_tokens(model)
    _, cache = model.run_with_cache(tokens)

    accumulated_resid, labels = cache.accumulated_resid(
        layer=-1, incl_mid=False, pos_slice=-1, return_labels=True
    )
    assert len(labels) == accumulated_resid.size(0)
    assert all("mid" not in label for label in labels)


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
        pos_slice=Slice(-1),
        tokens=answer_tokens[batch, 0],
        incorrect_tokens=answer_tokens[batch, 1],
    )
    assert torch.isclose(ref_logit_diffs[:, batch], logit_diffs).all()

    # Single token (int)
    batch = -1
    logit_diffs = cache.logit_attrs(
        accumulated_residual,
        batch_slice=Slice(batch),
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

    # Different shape for tokens and incorrect_tokens
    with pytest.raises(ValueError):
        cache.logit_attrs(
            accumulated_residual[:, batch, :],
            has_batch_dim=False,
            batch_slice=batch,
            pos_slice=-1,
            tokens=answer_tokens[batch, 0],
            incorrect_tokens=answer_tokens[batch, 0:1],
        )

    # No incorrect tokens
    ref_logit_diffs = einsum(
        "... d_model, ... d_model -> ...", scaled_residual_stack, answer_residual_directions[:, 0]
    )
    logit_diffs = cache.logit_attrs(
        accumulated_residual,
        pos_slice=-1,
        tokens=answer_tokens[:, 0],
        incorrect_tokens=None,
    )
    assert torch.isclose(ref_logit_diffs, logit_diffs).all()


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

    # Now do the same but using None as the layer and Slice(-1) as the pos_slice
    scaled_residual_stack, labels = cache.accumulated_resid(
        layer=None, incl_mid=True, pos_slice=Slice(-1), apply_ln=True, return_labels=True
    )
    assert torch.isclose(ref_scaled_residual_stack, scaled_residual_stack, atol=1e-7).all()

    expected_labels = []
    for l in range(model.cfg.n_layers + 1):
        if l == model.cfg.n_layers:
            expected_labels.append("final_post")
            continue
        expected_labels.append(f"{l}_pre")
        expected_labels.append(f"{l}_mid")

    assert labels == expected_labels


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
    scaled_residual_stack = cache.decompose_resid(layer=None, pos_slice=Slice(-1), apply_ln=True)

    assert torch.isclose(ref_scaled_residual_stack, scaled_residual_stack, atol=1e-7).all()


@torch.no_grad
def test_decompose_resid_including_attention():
    model = load_model("solu-2l")
    tokens, _ = get_ioi_tokens_and_answer_tokens(model)
    _, cache = model.run_with_cache(tokens)

    ref_attention_resids = torch.stack(
        [cache["attn_out", l][:, -1] for l in range(model.cfg.n_layers)]
    )
    residual_stack = cache.decompose_resid(
        layer=1, pos_slice=Slice(-1), mlp_input=True, apply_ln=False, incl_embeds=False, mode="attn"
    )

    assert torch.isclose(ref_attention_resids, residual_stack, atol=1e-7).all()


@torch.no_grad
def test_stack_head_results_with_apply_ln():
    # Load solu-2l
    model = load_model("solu-2l")

    tokens, _ = get_ioi_tokens_and_answer_tokens(model)

    # Run the model and cache all activations
    _, cache = model.run_with_cache(tokens)

    # Get per head resid stack and apply ln seperately (cribbed notebook code)
    per_head_residual = cache.stack_head_results(layer=-1, pos_slice=-1)
    ref_scaled_residual_stack = cache.apply_ln_to_stack(
        per_head_residual, layer=None, pos_slice=Slice(-1)
    )

    # Get scaled_residual_stack using apply_ln parameter
    scaled_residual_stack = cache.stack_head_results(layer=-1, pos_slice=-1, apply_ln=True)

    assert torch.isclose(ref_scaled_residual_stack, scaled_residual_stack, atol=1e-7).all()


@torch.no_grad
def test_stack_head_results_including_remainder():
    model = load_model("solu-2l")
    tokens, _ = get_ioi_tokens_and_answer_tokens(model)
    _, cache = model.run_with_cache(tokens)

    ref_resid_post = cache["resid_post", 0][None, :, -1]
    per_head_residual, labels = cache.stack_head_results(
        layer=1, pos_slice=-1, incl_remainder=True, return_labels=True
    )
    remainder = ref_resid_post - per_head_residual[:-1].sum(dim=0)
    assert torch.isclose(remainder, per_head_residual[-1]).all()
    assert labels[:-1] == [f"L0H{i}" for i in range(model.cfg.n_heads)]
    assert labels[-1] == "remainder"

    ref_resid_post = cache["resid_post", -1][None, :, -1]
    per_head_residual, labels = cache.stack_head_results(
        layer=0, pos_slice=-1, incl_remainder=True, return_labels=True
    )
    assert torch.isclose(ref_resid_post, per_head_residual, atol=1e-7).all()
    assert len(labels) == 1
    assert labels[-1] == "remainder"

    per_head_residual, labels = cache.stack_head_results(
        layer=0, pos_slice=-1, incl_remainder=False, return_labels=True
    )
    assert torch.isclose(per_head_residual, torch.zeros_like(per_head_residual)).all()
    assert len(labels) == 0


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
    scaled_residual_stack = cache.stack_neuron_results(layer=-1, pos_slice=Slice(-1), apply_ln=True)

    assert torch.isclose(ref_scaled_residual_stack, scaled_residual_stack, atol=1e-7).all()


@torch.no_grad
def test_stack_neuron_results_including_remainder():
    model = load_model("solu-2l")
    tokens, _ = get_ioi_tokens_and_answer_tokens(model)
    _, cache = model.run_with_cache(tokens)

    ref_resid_post = cache["resid_post", 0][None, :, -1]
    neuron_result_stack, labels = cache.stack_neuron_results(
        layer=1, pos_slice=Slice(-1), incl_remainder=True, return_labels=True
    )
    remainder = ref_resid_post - neuron_result_stack[:-1].sum(dim=0)
    assert torch.isclose(remainder, neuron_result_stack[-1]).all()
    assert labels[:-1] == [f"L0N{i}" for i in range(model.cfg.d_mlp)]
    assert labels[-1] == "remainder"

    ref_resid_post = cache["resid_post", -1][None, :, -1]
    neuron_result_stack, labels = cache.stack_neuron_results(
        layer=0, pos_slice=-1, incl_remainder=True, return_labels=True
    )
    assert torch.isclose(ref_resid_post, neuron_result_stack, atol=1e-7).all()
    assert len(labels) == 1
    assert labels[-1] == "remainder"

    neuron_result_stack, labels = cache.stack_neuron_results(
        layer=0, pos_slice=-1, incl_remainder=False, return_labels=True
    )
    assert torch.isclose(neuron_result_stack, torch.zeros_like(neuron_result_stack)).all()
    assert len(labels) == 0


@torch.no_grad
def test_stack_neuron_results_using_neuron_slice():
    model = load_model("solu-2l")
    tokens, _ = get_ioi_tokens_and_answer_tokens(model)
    _, cache = model.run_with_cache(tokens)

    neuron_result_stack, labels = cache.stack_neuron_results(
        layer=1, pos_slice=Slice(-1), neuron_slice=Slice([0, 1, 2]), return_labels=True
    )
    assert labels == [f"L0N{i}" for i in range(3)]


@torch.no_grad
def test_remove_batch_dim():
    model = load_model("solu-2l")
    tokens, _ = get_ioi_tokens_and_answer_tokens(model)
    _, cache = model.run_with_cache(tokens[:1])

    assert cache.has_batch_dim
    shapes_before_removal = {key: cache.cache_dict[key].shape for key in cache.cache_dict}

    # Removing batch dim changes the shape of the cached tensors
    cache.remove_batch_dim()
    assert not cache.has_batch_dim
    assert all(
        shapes_before_removal[key][1:] == cache.cache_dict[key].shape
        for key in shapes_before_removal
    )

    # Removing batch dim again does not change anything
    cache.remove_batch_dim()
    assert not cache.has_batch_dim
    assert all(
        shapes_before_removal[key][1:] == cache.cache_dict[key].shape
        for key in shapes_before_removal
    )


@torch.no_grad
def test_remove_batch_dim_fails_if_batch_gt_1():
    model = load_model("solu-2l")
    tokens, _ = get_ioi_tokens_and_answer_tokens(model)
    _, cache = model.run_with_cache(tokens)

    assert cache.has_batch_dim
    with pytest.raises(AssertionError):
        cache.remove_batch_dim()


@torch.no_grad
def test_retrieve_activations():
    model = load_model("solu-2l")
    tokens, _ = get_ioi_tokens_and_answer_tokens(model)
    _, cache = model.run_with_cache(tokens)

    key = ("scale", 1, "ln1")
    str_key = utils.get_act_name(*key)
    assert torch.equal(cache[key], cache[str_key])

    key = ("scale", -1, "ln1")
    str_key = f"scale{model.cfg.n_layers - 1}ln1"
    assert torch.equal(cache[key], cache[str_key])

    key = ("k", -1, None)
    str_key = f"blocks.{model.cfg.n_layers - 1}.attn.hook_k"
    assert torch.equal(cache[key], cache[str_key])

    key = "embed"
    str_key = utils.get_act_name(key)
    assert torch.equal(cache[key], cache[str_key])

    key = ("embed", None)
    str_key = utils.get_act_name(*key)
    assert torch.equal(cache[key], cache[str_key])


@torch.no_grad
def test_get_items():
    model = load_model("solu-2l")
    tokens, _ = get_ioi_tokens_and_answer_tokens(model)
    _, cache = model.run_with_cache(tokens)

    assert all(
        cache_key == cache_dict_key and torch.equal(cache_val, cache_dict_val)
        for (cache_key, cache_val), (cache_dict_key, cache_dict_val) in zip(
            cache.items(), cache.cache_dict.items()
        )
    )


@torch.no_grad
def test_get_values():
    model = load_model("solu-2l")
    tokens, _ = get_ioi_tokens_and_answer_tokens(model)
    _, cache = model.run_with_cache(tokens)

    assert all(
        torch.equal(cache_val, cache_dict_val)
        for cache_val, cache_dict_val in zip(cache.values(), cache.cache_dict.values())
    )


@torch.no_grad
def test_get_keys():
    model = load_model("solu-2l")
    tokens, _ = get_ioi_tokens_and_answer_tokens(model)
    _, cache = model.run_with_cache(tokens)

    assert all(
        cache_key == cache_dict_key
        for cache_key, cache_dict_key in zip(cache.keys(), cache.cache_dict.keys())
    )


@torch.no_grad
def test_apply_slice_to_batch_dim():
    model = load_model("solu-2l")
    tokens, _ = get_ioi_tokens_and_answer_tokens(model)
    _, cache = model.run_with_cache(tokens)

    assert cache.has_batch_dim
    batch_slice = Slice((2, 4))
    new_cache = cache.apply_slice_to_batch_dim(batch_slice)

    assert new_cache.has_batch_dim
    assert all(torch.equal(cache[key][2:4], new_cache[key]) for key in cache.cache_dict)

    batch_slice = 2
    new_cache = cache.apply_slice_to_batch_dim(batch_slice)

    assert not new_cache.has_batch_dim
    assert all(torch.equal(cache[key][2], new_cache[key]) for key in cache.cache_dict)


@torch.no_grad
def test_toggle_autodiff():
    model = load_model("solu-2l")
    tokens, _ = get_ioi_tokens_and_answer_tokens(model)
    _, cache = model.run_with_cache(tokens)

    assert not torch.is_grad_enabled()
    cache.toggle_autodiff(mode=True)
    assert torch.is_grad_enabled()


@torch.no_grad
def test_stack_activation():
    model = load_model("solu-2l")
    tokens, _ = get_ioi_tokens_and_answer_tokens(model)
    _, cache = model.run_with_cache(tokens)

    stack = cache.stack_activation("scale", -1, "ln1")
    assert all(
        torch.equal(cache[("scale", layer, "ln1")], stack[layer])
        for layer in range(model.cfg.n_layers)
    )

    stack = cache.stack_activation("scale", 1, "ln1")
    assert all(torch.equal(cache[("scale", layer, "ln1")], stack[layer]) for layer in range(1))


@torch.no_grad
def test_get_full_resid_decomposition():
    model = load_model("solu-2l")
    tokens, _ = get_ioi_tokens_and_answer_tokens(model)
    _, cache = model.run_with_cache(tokens)

    ref_head_stack, ref_head_stack_labels = cache.stack_head_results(
        layer=model.cfg.n_layers, pos_slice=Slice(-1), apply_ln=True, return_labels=True
    )
    ref_mlp_stack, ref_mlp_stack_labels = cache.decompose_resid(
        layer=model.cfg.n_layers,
        mlp_input=False,
        pos_slice=Slice(-1),
        incl_embeds=False,
        mode="mlp",
        apply_ln=True,
        return_labels=True,
    )
    ref_embed = cache.apply_ln_to_stack(
        cache["embed"][None, :, -1], pos_slice=Slice(-1), mlp_input=False
    )
    ref_pos_embed = cache.apply_ln_to_stack(
        cache["pos_embed"][None, :, -1], pos_slice=Slice(-1), mlp_input=False
    )

    ref_bias = model.accumulated_bias(model.cfg.n_layers, mlp_input=False, include_mlp_biases=False)
    ref_bias = ref_bias.expand((1,) + ref_head_stack.shape[1:])
    ref_bias = cache.apply_ln_to_stack(ref_bias, pos_slice=Slice(-1), mlp_input=False)

    head_stack_len = ref_head_stack.size(0)
    mlp_stack_len = ref_mlp_stack.size(0)

    residual_stack, residual_stack_labels = cache.get_full_resid_decomposition(
        layer=-1, pos_slice=-1, apply_ln=True, expand_neurons=False, return_labels=True
    )
    assert torch.isclose(ref_head_stack, residual_stack[:head_stack_len], atol=1e-7).all()
    assert ref_head_stack_labels == residual_stack_labels[:head_stack_len]

    assert torch.isclose(
        ref_mlp_stack, residual_stack[head_stack_len : head_stack_len + mlp_stack_len], atol=1e-7
    ).all()
    assert (
        ref_mlp_stack_labels
        == residual_stack_labels[head_stack_len : head_stack_len + mlp_stack_len]
    )

    assert torch.isclose(
        ref_embed,
        residual_stack[head_stack_len + mlp_stack_len : head_stack_len + mlp_stack_len + 1],
        atol=1e-7,
    ).all()
    assert "embed" == residual_stack_labels[head_stack_len + mlp_stack_len]

    assert torch.isclose(
        ref_pos_embed,
        residual_stack[head_stack_len + mlp_stack_len + 1 : head_stack_len + mlp_stack_len + 2],
        atol=1e-7,
    ).all()
    assert "pos_embed" == residual_stack_labels[head_stack_len + mlp_stack_len + 1]

    assert torch.isclose(
        ref_bias,
        residual_stack[head_stack_len + mlp_stack_len + 2 : head_stack_len + mlp_stack_len + 3],
        atol=1e-7,
    ).all()
    assert "bias" == residual_stack_labels[head_stack_len + mlp_stack_len + 2]


@torch.no_grad
def test_get_full_resid_decomposition_with_neurons_expanded():
    model = load_model("solu-2l")
    tokens, _ = get_ioi_tokens_and_answer_tokens(model)
    _, cache = model.run_with_cache(tokens)

    ref_head_stack, ref_head_stack_labels = cache.stack_head_results(
        layer=1, pos_slice=Slice(-1), apply_ln=True, return_labels=True
    )
    ref_neuron_stack, ref_neuron_labels = cache.stack_neuron_results(
        1, pos_slice=Slice(-1), return_labels=True
    )
    ref_neuron_stack = cache.apply_ln_to_stack(ref_neuron_stack, layer=1, pos_slice=Slice(-1))

    head_stack_len = ref_head_stack.size(0)
    neuron_stack_len = ref_neuron_stack.size(0)

    residual_stack, residual_stack_labels = cache.get_full_resid_decomposition(
        layer=1, pos_slice=Slice(-1), apply_ln=True, expand_neurons=True, return_labels=True
    )

    assert torch.isclose(
        ref_neuron_stack,
        residual_stack[head_stack_len : head_stack_len + neuron_stack_len],
        atol=1e-7,
    ).all()
    assert (
        ref_neuron_labels
        == residual_stack_labels[head_stack_len : head_stack_len + neuron_stack_len]
    )


@torch.no_grad
def test_get_full_resid_decomposition_without_applying_ln():
    model = load_model("solu-2l")
    tokens, _ = get_ioi_tokens_and_answer_tokens(model)
    _, cache = model.run_with_cache(tokens)

    ref_head_stack = cache.stack_head_results(
        layer=1, pos_slice=Slice(-1), apply_ln=True, return_labels=False
    )
    ref_neuron_stack = cache.stack_neuron_results(1, pos_slice=Slice(-1), return_labels=False)

    head_stack_len = ref_head_stack.size(0)
    neuron_stack_len = ref_neuron_stack.size(0)

    residual_stack = cache.get_full_resid_decomposition(
        layer=1, pos_slice=Slice(-1), apply_ln=False, expand_neurons=True, return_labels=False
    )

    assert torch.isclose(
        ref_neuron_stack,
        residual_stack[head_stack_len : head_stack_len + neuron_stack_len],
        atol=1e-7,
    ).all()


@torch.no_grad
def test_get_full_resid_decomposition_attn_only_model():
    model = load_model("attn-only-2l")
    tokens, _ = get_ioi_tokens_and_answer_tokens(model)
    _, cache = model.run_with_cache(tokens)

    ref_head_stack = cache.stack_head_results(
        layer=1, pos_slice=Slice(-1), apply_ln=False, return_labels=False
    )

    head_stack_len = ref_head_stack.size(0)

    residual_stack = cache.get_full_resid_decomposition(
        layer=1, pos_slice=Slice(-1), apply_ln=False, expand_neurons=False, return_labels=False
    )

    assert torch.isclose(ref_head_stack, residual_stack[:head_stack_len], atol=1e-7).all()


@torch.no_grad
def test_compute_test_head_results_does_not_compute_results_twice():
    model = load_model("attn-only-2l")
    tokens, _ = get_ioi_tokens_and_answer_tokens(model)
    _, cache = model.run_with_cache(tokens)

    assert "blocks.0.attn.hook_result" not in cache.cache_dict
    cache.compute_head_results()
    assert "blocks.0.attn.hook_result" in cache.cache_dict

    # set infinity to the first element of the head results
    assert cache.cache_dict["blocks.0.attn.hook_result"][0, 0, 0, 0] != float("inf")
    cache.cache_dict["blocks.0.attn.hook_result"][0, 0, 0, 0] = float("inf")
    cache.compute_head_results()

    # assert the value has not changed
    assert cache.cache_dict["blocks.0.attn.hook_result"][0, 0, 0, 0] == float("inf")


@torch.no_grad
def test_get_neuron_results():
    model = load_model("solu-2l")
    tokens, _ = get_ioi_tokens_and_answer_tokens(model)
    _, cache = model.run_with_cache(tokens)

    layer = 1
    ref_neuron_acts = (
        cache[f"blocks.{layer}.mlp.hook_post"][:, -1, :2, None] * model.blocks[layer].mlp.W_out[:2]
    )

    neuron_acts = cache.get_neuron_results(
        layer,
        neuron_slice=[0, 1],
        pos_slice=-1,
    )

    assert torch.isclose(ref_neuron_acts, neuron_acts).all()


@torch.no_grad
def test_get_neuron_results_without_slice():
    model = load_model("solu-2l")
    tokens, _ = get_ioi_tokens_and_answer_tokens(model)
    _, cache = model.run_with_cache(tokens)

    layer = 1
    ref_neuron_acts = (
        cache[f"blocks.{layer}.mlp.hook_post"][..., None] * model.blocks[layer].mlp.W_out
    )

    neuron_acts = cache.get_neuron_results(
        layer,
        neuron_slice=None,
        pos_slice=None,
    )

    assert torch.isclose(ref_neuron_acts, neuron_acts).all()
