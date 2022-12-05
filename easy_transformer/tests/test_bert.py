import functools

import torch as t

# to understand the reason for the [type: ignore] directive, see stuff like https://github.com/huggingface/transformers/issues/18464
from transformers import AutoModelForMaskedLM  # type: ignore
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.auto.tokenization_auto import AutoTokenizer

import easy_transformer.utils as utils
from easy_transformer import EasyBERT

MODEL_NAME = "bert-base-uncased"
TEXT = "Hello world!"
DEVICE = "cpu"  # for compatibility


def test_that_logit_output_is_the_same():
    atol = 1e-4
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    our_model = EasyBERT.from_pretrained(MODEL_NAME)
    our_model.eval()  # To turn off dropout
    our_output = our_model(TEXT)

    n_tokens_in_input = tokenizer(TEXT, return_tensors="pt")["input_ids"].shape[1]
    assert n_tokens_in_input == 5
    assert our_output.logits.shape == (
        1,
        n_tokens_in_input,
        our_model.config.vocab_size,
    )
    assert our_output.embedding.shape == (
        1,
        n_tokens_in_input,
        our_model.config.d_model,
    )

    assert (
        AutoModelForMaskedLM.from_pretrained is not None
    )  # recommended by https://github.com/microsoft/pylance-release/issues/333#issuecomment-688522371
    hugging_face_model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    hf_output: MaskedLMOutput = hugging_face_model(
        **tokenizer(TEXT, return_tensors="pt"),
        output_hidden_states=True,
    )

    assert hf_output.hidden_states is not None

    assert t.allclose(our_output.embedding, hf_output.hidden_states[0], atol=atol)

    assert our_output.logits.shape == hf_output.logits.shape
    assert t.allclose(our_output.logits, hf_output.logits, atol=atol)
    assert t.allclose(our_output.loss, t.tensor(15.7515), atol=atol)


def test_api_compatbility():
    """
    The following tests were inspired by [EasyTransformer_Demo.ipynb].
    """
    model = EasyBERT.from_pretrained(MODEL_NAME)  # type: ignore
    prompt = "Jay-Z and Wayne potentially are the GOATs."
    assert model.to_str_tokens(prompt) == [
        "[CLS]",
        "jay",
        "-",
        "z",
        "and",
        "wayne",
        "potentially",
        "are",
        "the",
        "goats",
        ".",
        "[SEP]",
    ]

    embed_or_first_layer = lambda name: (name[:6] != "blocks" or name[:8] == "blocks.0")
    hook_name_to_shapes = {}

    def print_shape(tensor, hook):
        hook_name_to_shapes[hook.name] = tensor.shape

    model.run_with_hooks(prompt, forward_hooks=[(embed_or_first_layer, print_shape)])

    assert hook_name_to_shapes == {
        "embeddings": t.Size([1, 12, 768]),
    }

    def print_corner(tensor, hook):
        print(hook.name)
        print(utils.get_corner(tensor))

    logits = model.run_with_hooks(
        prompt, forward_hooks=[(embed_or_first_layer, print_corner)]
    )

    assert logits == 50

    logits, cache = model.run_with_cache(prompt)
    for name in cache:
        if embed_or_first_layer(name):
            # TODO make this append to a list instead of printing
            print(name, cache[name].shape)

    # Example - prune heads 0, 3 and 7 from layer 3 and heads 8 and 9 from layer 7
    layer = 3
    head_indices = t.tensor([0, 3, 7])
    layer_2 = 7
    head_indices_2 = t.tensor([8, 9])

    def prune_fn_1(z, hook):
        # The shape of the z tensor is batch x pos x head_index x d_head
        z[:, :, head_indices, :] = 0.0
        return z

    def prune_fn_2(z, hook):
        # The shape of the z tensor is batch x pos x head_index x d_head
        z[:, :, head_indices_2, :] = 0.0
        return z

    logits = model.run_with_hooks(
        prompt,
        forward_hooks=[
            (f"blocks.{layer}.attn.hook_z", prune_fn_1),
            (f"blocks.{layer_2}.attn.hook_z", prune_fn_2),
        ],
    )

    model.reset_hooks()

    def filter_hook_attn(name):
        split_name = name.split(".")
        return split_name[-1] == "hook_attn"

    def restrict_attn(attn, hook):
        # Attn has shape batch x head_index x query_pos x key_pos
        n_ctx = attn.size(-2)
        key_pos = t.arange(n_ctx)[None, :]
        query_pos = t.arange(n_ctx)[:, None]
        mask = (key_pos > (query_pos - 2)).to(DEVICE)
        ZERO = t.tensor(0.0).to(DEVICE)
        attn = t.where(mask, attn, ZERO)
        return attn

    text = "GPU go brrrr"
    original_logits = model(text)
    logits = model.run_with_hooks(
        text, forward_hooks=[(filter_hook_attn, restrict_attn)]
    )
    print("New logits")
    print(utils.get_corner(logits, 3))
    print("Original logits")
    print(utils.get_corner(original_logits, 3))

    model.reset_hooks()

    def filter_hook_attn(name):
        split_name = name.split(".")
        return split_name[-1] == "hook_attn"

    def restrict_attn(attn, hook):
        # Attn has shape batch x head_index x query_pos x key_pos
        n_ctx = attn.size(-2)
        key_pos = t.arange(n_ctx)[None, :]
        query_pos = t.arange(n_ctx)[:, None]
        mask = (key_pos > (query_pos - 2)).to(DEVICE)
        ZERO = t.tensor(0.0).to(DEVICE)
        attn = t.where(mask, attn, ZERO)
        return attn

    text = "GPU go brrrr"
    original_logits = model(text)
    logits = model.run_with_hooks(
        text, forward_hooks=[(filter_hook_attn, restrict_attn)]
    )
    print("New logits")
    print(utils.get_corner(logits, 3))
    print("Original logits")
    print(utils.get_corner(original_logits, 3))

    model.reset_hooks()

    def filter_hook_attn(name):
        split_name = name.split(".")
        return split_name[-1] == "hook_attn"

    def restrict_attn(attn, hook):
        # Attn has shape batch x head_index x query_pos x key_pos
        n_ctx = attn.size(-2)
        key_pos = t.arange(n_ctx)[None, :]
        query_pos = t.arange(n_ctx)[:, None]
        mask = (key_pos > (query_pos - 2)).to(DEVICE)
        ZERO = t.tensor(0.0).to(DEVICE)
        attn = t.where(mask, attn, ZERO)
        return attn

    text = "GPU go brrrr"
    original_logits = model(text)
    logits = model.run_with_hooks(
        text, forward_hooks=[(filter_hook_attn, restrict_attn)]
    )
    print("New logits")
    print(utils.get_corner(logits, 3))
    print("Original logits")
    print(utils.get_corner(original_logits, 3))

    # Finding the dataset example that most activates a given neuron

    # We focus on neuron 13 in layer 5
    model.reset_hooks(clear_contexts=True)
    animal_texts = [
        "The dog was green",
        "The cat was blue",
        "The squid was magenta",
        "The blobfish was grey",
    ]
    layer = 5
    neuron_index = 13

    def best_act_hook(neuron_acts, hook, text):
        if "best" not in hook.ctx:
            hook.ctx["best"] = -1e3
        print("Neuron acts:", neuron_acts[0, :, neuron_index])
        if hook.ctx["best"] < neuron_acts[0, :, neuron_index].max():
            print(
                f'Updating best act from {hook.ctx["best"]} to {neuron_acts[0, :, neuron_index].max().item()}'
            )
            hook.ctx["best"] = neuron_acts[0, :, neuron_index].max().item()
            hook.ctx["text"] = text

    for animal_text in animal_texts:
        (print(model.to_str_tokens(animal_text)))
        # Use partial to give the hook access to the relevant text
        model.run_with_hooks(
            animal_text,
            forward_hooks=[
                (
                    f"blocks.{layer}.mlp.hook_post",
                    functools.partial(best_act_hook, text=animal_text),
                )
            ],
        )
    print()
    print(
        "Maximally activating dataset example:",
        model.name_to_hook[f"blocks.{layer}.mlp.hook_post"].ctx["text"],
    )
    model.reset_hooks(clear_contexts=True)
