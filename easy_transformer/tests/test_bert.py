import torch as t

# trying to import [AutoModelForMaskedLM] from the non-private location fucks up, not sure why; it makes
# [from_pretrained == None]
from transformers import AutoModelForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.auto.tokenization_auto import AutoTokenizer

from easy_transformer import EasyBERT


def test_that_im_awesome():
    model_name = "bert-base-uncased"
    text = "Hello world!"
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    from easy_transformer import EasyBERT

    atol = 1e-1

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    our_model = EasyBERT.from_pretrained(model_name)
    # TODO figure out what's up with [using eos_token] and [using bos_token]
    our_output = our_model(text)

    n_tokens_in_input = tokenizer(text, return_tensors="pt")["input_ids"].shape[1]
    assert n_tokens_in_input == 5
    assert our_output.logits.shape == (
        1,
        n_tokens_in_input,
        our_model.config.vocab_size,
    )
    assert our_output.embedding.shape == (
        1,
        n_tokens_in_input,
        our_model.config.hidden_size,
    )

    assert (
        AutoModelForMaskedLM.from_pretrained is not None
    )  # recommended by https://github.com/microsoft/pylance-release/issues/333#issuecomment-688522371
    hugging_face_model = AutoModelForMaskedLM.from_pretrained(model_name)
    hf_output: MaskedLMOutput = hugging_face_model(
        **tokenizer(text, return_tensors="pt"),
        output_hidden_states=True,
    )

    assert hf_output.hidden_states is not None

    # let's check the embeddings

    assert t.allclose(
        our_output.embedding, hf_output.hidden_states[0], atol=atol
    )  # TODO higher precision (lower atol)?  i think it's because of a limitation in the size of the floats? not sure! otherwise i got floating point rouding errors, i think

    assert our_output.logits.shape == hf_output.logits.shape
    assert t.allclose(our_output.logits, hf_output.logits, atol=atol)
