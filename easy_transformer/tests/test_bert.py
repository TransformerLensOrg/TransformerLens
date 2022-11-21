# %%

from typeguard.importhook import install_import_hook

install_import_hook("easy_transformer")

from torchtyping import TensorType as patch_typeguard  # TODO is this right?

patch_typeguard()

import torch

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
    assert torch.allclose(output.logits, hf_output.logits, atol=1e-4)
