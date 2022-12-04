import torch as t

# to understand the reason for the [type: ignore] directive, see stuff like https://github.com/huggingface/transformers/issues/18464
from transformers import AutoModelForMaskedLM  # type: ignore
from transformers.modeling_outputs import MaskedLMOutput

from easy_transformer import EasyBERT


def test_that_token_output_is_the_same():
    model_name = "bert-base-uncased"
    text = "Hello world!"
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    atol = 1e-4

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    our_model = EasyBERT.from_pretrained(model_name)
    # TODO maybe turn this off in general? i think ET does
    our_model.eval()  # To turn off dropout
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

    assert t.allclose(our_output.embedding, hf_output.hidden_states[0], atol=atol)

    assert our_output.logits.shape == hf_output.logits.shape
    assert t.allclose(our_output.logits, hf_output.logits, atol=atol)
