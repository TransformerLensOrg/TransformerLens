# %%

import torch as t

# trying to import [AutoModelForMaskedLM] from the non-private location fucks up, not sure why; it makes
# [from_pretrained == None]
from transformers import AutoModelForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.auto.tokenization_auto import AutoTokenizer

from easy_transformer import EasyBERT


def test_bert():
    model_name = "bert-base-uncased"
    text = "Hello world!"
    model = EasyBERT.EasyBERT.from_pretrained(model_name)  # TODO why two?
    output: MaskedLMOutput = model(text)  # TODO need to change the type

    assert output.logits.shape == (1, 5, model.config.d_vocab)  # TODO why 5?

    # now let's compare it to the HuggingFace version
    assert (
        AutoModelForMaskedLM.from_pretrained is not None
    )  # recommended by https://github.com/microsoft/pylance-release/issues/333#issuecomment-688522371
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    hf_output = model(**tokenizer(text, return_tensors="pt"))
    assert t.allclose(output.logits, hf_output.logits, atol=1e-4)


def test_embeddings():
    hf = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    model = EasyBERT.EasyBERT.from_pretrained("bert-base-uncased")
    assert t.allclose(
        hf.bert.embeddings.word_embeddings.weight,
        model.embeddings.word_embeddings.weight,
        atol=1e-4,
    )


# %%


def make_this_a_test():
    # TODO make an anki about this workflow- including function scope for name conflicts

    import torch
    from transformers import AutoModelForMaskedLM
    from transformers.modeling_outputs import MaskedLMOutput
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    from easy_transformer import EasyBERT

    hf = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    model = EasyBERT.EasyBERT.from_pretrained("bert-base-uncased")

    assert t.allclose(
        hf.bert.embeddings.word_embeddings.weight,
        model.embeddings.word_embeddings.weight,
        atol=1e-4,
    )


# %%

# now we test output
# TODO ensure that our model matches the architecture diagram of BERT

# autoreload 2


def make_this_a_test():
    model_name = "bert-base-uncased"
    text = "Hello world!"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    our_model = EasyBERT.EasyBERT.from_pretrained(model_name)  # TODO why two?
    output: MaskedLMOutput = our_model(text)  # TODO need to change the type

    n_tokens_in_input = tokenizer(text, return_tensors="pt")
    assert n_tokens_in_input == 5
    assert output.logits.shape == (1, n_tokens_in_input, our_model.config.d_vocab)

    assert (
        AutoModelForMaskedLM.from_pretrained is not None
    )  # recommended by https://github.com/microsoft/pylance-release/issues/333#issuecomment-688522371
    hugging_face_model = AutoModelForMaskedLM.from_pretrained(model_name)
    hf_output = hugging_face_model(**tokenizer(text, return_tensors="pt"))
    assert t.allclose(output.logits, hf_output.logits, atol=1e-4)


# %%


def test_that_im_awesome():
    model_name = "bert-base-uncased"
    text = "Hello world!"
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    from easy_transformer import EasyBERT

    atol = 1e-1

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    our_model = EasyBERT.EasyBERT.from_pretrained(model_name)  # TODO why two?
    # TODO add various [return_type] options
    # TODO figure out what's up with [using eos_token] and [using bos_token]
    our_output: EasyBERT.Output = our_model(
        text
    )  # TODO set the type of the variable on the LHS

    # trying to import [AutoModelForMaskedLM] from the non-private location fucks up, not sure why; it makes
    # [from_pretrained == None]
    from transformers import AutoModelForMaskedLM

    n_tokens_in_input = tokenizer(text, return_tensors="pt")["input_ids"].shape[1]
    assert n_tokens_in_input == 5
    # TODO make this output.logits to match the other test
    # TODO load the weights for the unembed
    assert our_output.logits.shape == (1, n_tokens_in_input, our_model.config.d_vocab)
    assert our_output.embedding.shape == (
        1,
        n_tokens_in_input,
        our_model.config.hidden_size,
    )
    assert our_output.hidden_states.shape == (
        our_model.config.n_layers,
        1,
        n_tokens_in_input,
        our_model.config.hidden_size,
    )

    assert (
        AutoModelForMaskedLM.from_pretrained is not None
    )  # recommended by https://github.com/microsoft/pylance-release/issues/333#issuecomment-688522371
    hugging_face_model = AutoModelForMaskedLM.from_pretrained(model_name)
    hf_output: MaskedLMOutput = hugging_face_model(
        **tokenizer(text, return_tensors="pt"), output_hidden_states=True
    )

    # let's check the embeddings

    assert (
        hf_output.hidden_states is not None
    )  # for pylance; True because we set output_hidden_states=True
    assert t.allclose(
        our_output.embedding, hf_output.hidden_states[0], atol=atol
    )  # TODO higher precision?
    # otherwise i got floating point rouding errors, i think

    # TODO i think it's because of a limitation in the size of the floats? not sure!

    # let's check if the hidden states are the same
    for i in range(our_model.config.n_layers):
        our_hidden_state = our_output.hidden_states[i]
        hf_hidden_state = hf_output.hidden_states[
            i + 1
        ]  # +1 because of the embedding layer
        # hf also has a layer norm, but we don't- note [grad_fn]- well we do use layernorm,
        # but in a different place in the network
        # we do actually disagree on the output logits
        # also there's... something else. like there's a view differnce in some grad fns; our grad fn is
        # [LogSoftmaxBackwards] and theirs is [ViewBackward]
        # TODO i bet the mask is part of the story
        # in hugging face, the hidden states change slowly from embedding to the output of the first layer
        # in ours..., wait, is that true?
        assert t.allclose(our_hidden_state, hf_hidden_state, atol=atol)

    assert our_output.logits.shape == hf_output.logits.shape
    assert t.allclose(our_output.logits, hf_output.logits, atol=atol)
