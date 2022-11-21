import logging
from typing import Dict

import torch
from transformers import (
    AutoModelForMaskedLM,  # unfortunately the suggestion to import from the non-private location doesn't work; it makes [from_pretrained == None]
)
from transformers.models.auto.tokenization_auto import AutoTokenizer

from easy_transformer.hook_points import HookedRootModule

from .. import loading_from_pretrained as loading
from . import EasyBERTConfig, embeddings


class EasyBERT(HookedRootModule):
    # written in this style because this guarantees that [self] isn't set inside here
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        **model_kwargs,
    ):
        # TODO use these other parameters
        logging.info(f"Loading model: {model_name}")
        official_model_name = loading.get_official_model_name(model_name)
        config = EasyBERTConfig.EasyBERTConfig(
            n_layers=12,
            n_heads=12,
            hidden_size=768,
            dropout=0.1,
            model_name=official_model_name,
            d_vocab=30522,
        )  # TODO fancier :P
        assert (
            AutoModelForMaskedLM.from_pretrained is not None
        )  # recommended by some github link TODO find it
        state_dict = AutoModelForMaskedLM.from_pretrained(
            official_model_name
        ).state_dict()
        print(f"{state_dict.keys()=}")
        model = cls(config, **model_kwargs)
        model.load_and_process_state_dict(state_dict)
        logging.info(
            f"Finished loading pretrained model {model_name} into EasyTransformer!"
        )

        return model

    def __init__(self, config: EasyBERTConfig.EasyBERTConfig, **kwargs):
        super().__init__()
        self.config = config
        self.embeddings = embeddings.Embeddings(config)

    def load_and_process_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.embeddings.load_and_process_state_dict(state_dict["bert.embeddings"])
        """
        ['bert.encoder.layer.0.attention.self.query.weight', 'bert.encoder.layer.0.attention.self.query.bias', 'bert.encoder.layer.0.attention.self.key.weight', 'bert.encoder.layer.0.attention.self.key.bias', 'bert.encoder.layer.0.attention.self.value.weight', 'bert.encoder.layer.0.attention.self.value.bias', 'bert.encoder.layer.0.attention.output.dense.weight', 'bert.encoder.layer.0.attention.output.dense.bias', 'bert.encoder.layer.0.attention.output.LayerNorm.weight', 'bert.encoder.layer.0.attention.output.LayerNorm.bias', 'bert.encoder.layer.0.intermediate.dense.weight', 'bert.encoder.layer.0.intermediate.dense.bias', 'bert.encoder.layer.0.output.dense.weight', 'bert.encoder.layer.0.output.dense.bias', 'bert.encoder.layer.0.output.LayerNorm.weight', 'bert.encoder.layer.0.output.LayerNorm.bias', 'bert.encoder.layer.1.attention.self.query.weight', 'bert.encoder.layer.1.attention.self.query.bias', 'bert.encoder.layer.1.attention.self.key.weight', 'bert.encoder.layer.1.attention.self.key.bias', 'bert.encoder.layer.1.attention.self.value.weight', 'bert.encoder.layer.1.attention.self.value.bias', 'bert.encoder.layer.1.attention.output.dense.weight', 'bert.encoder.layer.1.attention.output.dense.bias', 'bert.encoder.layer.1.attention.output.LayerNorm.weight', 'bert.encoder.layer.1.attention.output.LayerNorm.bias', 'bert.encoder.layer.1.intermediate.dense.weight', 'bert.encoder.layer.1.intermediate.dense.bias', 'bert.encoder.layer.1.output.dense.weight', 'bert.encoder.layer.1.output.dense.bias', 'bert.encoder.layer.1.output.LayerNorm.weight', 'bert.encoder.layer.1.output.LayerNorm.bias', 'bert.encoder.layer.2.attention.self.query.weight', 'bert.encoder.layer.2.attention.self.query.bias', 'bert.encoder.layer.2.attention.self.key.weight', 'bert.encoder.layer.2.attention.self.key.bias', 'bert.encoder.layer.2.attention.self.value.weight', 'bert.encoder.layer.2.attention.self.value.bias', 'bert.encoder.layer.2.attention.output.dense.weight', 'bert.encoder.layer.2.attention.output.dense.bias', 'bert.encoder.layer.2.attention.output.LayerNorm.weight', 'bert.encoder.layer.2.attention.output.LayerNorm.bias', 'bert.encoder.layer.2.intermediate.dense.weight', 'bert.encoder.layer.2.intermediate.dense.bias', 'bert.encoder.layer.2.output.dense.weight', 'bert.encoder.layer.2.output.dense.bias', 'bert.encoder.layer.2.output.LayerNorm.weight', 'bert.encoder.layer.2.output.LayerNorm.bias', 'bert.encoder.layer.3.attention.self.query.weight', 'bert.encoder.layer.3.attention.self.query.bias', 'bert.encoder.layer.3.attention.self.key.weight', 'bert.encoder.layer.3.attention.self.key.bias', 'bert.encoder.layer.3.attention.self.value.weight', 'bert.encoder.layer.3.attention.self.value.bias', 'bert.encoder.layer.3.attention.output.dense.weight', 'bert.encoder.layer.3.attention.output.dense.bias', 'bert.encoder.layer.3.attention.output.LayerNorm.weight', 'bert.encoder.layer.3.attention.output.LayerNorm.bias', 'bert.encoder.layer.3.intermediate.dense.weight', 'bert.encoder.layer.3.intermediate.dense.bias', 'bert.encoder.layer.3.output.dense.weight', 'bert.encoder.layer.3.output.dense.bias', 'bert.encoder.layer.3.output.LayerNorm.weight', 'bert.encoder.layer.3.output.LayerNorm.bias', 'bert.encoder.layer.4.attention.self.query.weight', 'bert.encoder.layer.4.attention.self.query.bias', 'bert.encoder.layer.4.attention.self.key.weight', 'bert.encoder.layer.4.attention.self.key.bias', 'bert.encoder.layer.4.attention.self.value.weight', 'bert.encoder.layer.4.attention.self.value.bias', 'bert.encoder.layer.4.attention.output.dense.weight', 'bert.encoder.layer.4.attention.output.dense.bias', 'bert.encoder.layer.4.attention.output.LayerNorm.weight', 'bert.encoder.layer.4.attention.output.LayerNorm.bias', 'bert.encoder.layer.4.intermediate.dense.weight', 'bert.encoder.layer.4.intermediate.dense.bias', 'bert.encoder.layer.4.output.dense.weight', 'bert.encoder.layer.4.output.dense.bias', 'bert.encoder.layer.4.output.LayerNorm.weight', 'bert.encoder.layer.4.output.LayerNorm.bias', 'bert.encoder.layer.5.attention.self.query.weight', 'bert.encoder.layer.5.attention.self.query.bias', 'bert.encoder.layer.5.attention.self.key.weight', 'bert.encoder.layer.5.attention.self.key.bias', 'bert.encoder.layer.5.attention.self.value.weight', 'bert.encoder.layer.5.attention.self.value.bias', 'bert.encoder.layer.5.attention.output.dense.weight', 'bert.encoder.layer.5.attention.output.dense.bias', 'bert.encoder.layer.5.attention.output.LayerNorm.weight', 'bert.encoder.layer.5.attention.output.LayerNorm.bias', 'bert.encoder.layer.5.intermediate.dense.weight', 'bert.encoder.layer.5.intermediate.dense.bias', 'bert.encoder.layer.5.output.dense.weight', 'bert.encoder.layer.5.output.dense.bias', 'bert.encoder.layer.5.output.LayerNorm.weight', 'bert.encoder.layer.5.output.LayerNorm.bias', 'bert.encoder.layer.6.attention.self.query.weight', 'bert.encoder.layer.6.attention.self.query.bias', 'bert.encoder.layer.6.attention.self.key.weight', 'bert.encoder.layer.6.attention.self.key.bias', 'bert.encoder.layer.6.attention.self.value.weight', 'bert.encoder.layer.6.attention.self.value.bias', 'bert.encoder.layer.6.attention.output.dense.weight', 'bert.encoder.layer.6.attention.output.dense.bias', 'bert.encoder.layer.6.attention.output.LayerNorm.weight', 'bert.encoder.layer.6.attention.output.LayerNorm.bias', 'bert.encoder.layer.6.intermediate.dense.weight', 'bert.encoder.layer.6.intermediate.dense.bias', 'bert.encoder.layer.6.output.dense.weight', 'bert.encoder.layer.6.output.dense.bias', 'bert.encoder.layer.6.output.LayerNorm.weight', 'bert.encoder.layer.6.output.LayerNorm.bias', 'bert.encoder.layer.7.attention.self.query.weight', 'bert.encoder.layer.7.attention.self.query.bias', 'bert.encoder.layer.7.attention.self.key.weight', 'bert.encoder.layer.7.attention.self.key.bias', 'bert.encoder.layer.7.attention.self.value.weight', 'bert.encoder.layer.7.attention.self.value.bias', 'bert.encoder.layer.7.attention.output.dense.weight', 'bert.encoder.layer.7.attention.output.dense.bias', 'bert.encoder.layer.7.attention.output.LayerNorm.weight', 'bert.encoder.layer.7.attention.output.LayerNorm.bias', 'bert.encoder.layer.7.intermediate.dense.weight', 'bert.encoder.layer.7.intermediate.dense.bias', 'bert.encoder.layer.7.output.dense.weight', 'bert.encoder.layer.7.output.dense.bias', 'bert.encoder.layer.7.output.LayerNorm.weight', 'bert.encoder.layer.7.output.LayerNorm.bias', 'bert.encoder.layer.8.attention.self.query.weight', 'bert.encoder.layer.8.attention.self.query.bias', 'bert.encoder.layer.8.attention.self.key.weight', 'bert.encoder.layer.8.attention.self.key.bias', 'bert.encoder.layer.8.attention.self.value.weight', 'bert.encoder.layer.8.attention.self.value.bias', 'bert.encoder.layer.8.attention.output.dense.weight', 'bert.encoder.layer.8.attention.output.dense.bias', 'bert.encoder.layer.8.attention.output.LayerNorm.weight', 'bert.encoder.layer.8.attention.output.LayerNorm.bias', 'bert.encoder.layer.8.intermediate.dense.weight', 'bert.encoder.layer.8.intermediate.dense.bias', 'bert.encoder.layer.8.output.dense.weight', 'bert.encoder.layer.8.output.dense.bias', 'bert.encoder.layer.8.output.LayerNorm.weight', 'bert.encoder.layer.8.output.LayerNorm.bias', 'bert.encoder.layer.9.attention.self.query.weight', 'bert.encoder.layer.9.attention.self.query.bias', 'bert.encoder.layer.9.attention.self.key.weight', 'bert.encoder.layer.9.attention.self.key.bias', 'bert.encoder.layer.9.attention.self.value.weight', 'bert.encoder.layer.9.attention.self.value.bias', 'bert.encoder.layer.9.attention.output.dense.weight', 'bert.encoder.layer.9.attention.output.dense.bias', 'bert.encoder.layer.9.attention.output.LayerNorm.weight', 'bert.encoder.layer.9.attention.output.LayerNorm.bias', 'bert.encoder.layer.9.intermediate.dense.weight', 'bert.encoder.layer.9.intermediate.dense.bias', 'bert.encoder.layer.9.output.dense.weight', 'bert.encoder.layer.9.output.dense.bias', 'bert.encoder.layer.9.output.LayerNorm.weight', 'bert.encoder.layer.9.output.LayerNorm.bias', 'bert.encoder.layer.10.attention.self.query.weight', 'bert.encoder.layer.10.attention.self.query.bias', 'bert.encoder.layer.10.attention.self.key.weight', 'bert.encoder.layer.10.attention.self.key.bias', 'bert.encoder.layer.10.attention.self.value.weight', 'bert.encoder.layer.10.attention.self.value.bias', 'bert.encoder.layer.10.attention.output.dense.weight', 'bert.encoder.layer.10.attention.output.dense.bias', 'bert.encoder.layer.10.attention.output.LayerNorm.weight', 'bert.encoder.layer.10.attention.output.LayerNorm.bias', 'bert.encoder.layer.10.intermediate.dense.weight', 'bert.encoder.layer.10.intermediate.dense.bias', 'bert.encoder.layer.10.output.dense.weight', 'bert.encoder.layer.10.output.dense.bias', 'bert.encoder.layer.10.output.LayerNorm.weight', 'bert.encoder.layer.10.output.LayerNorm.bias', 'bert.encoder.layer.11.attention.self.query.weight', 'bert.encoder.layer.11.attention.self.query.bias', 'bert.encoder.layer.11.attention.self.key.weight', 'bert.encoder.layer.11.attention.self.key.bias', 'bert.encoder.layer.11.attention.self.value.weight', 'bert.encoder.layer.11.attention.self.value.bias', 'bert.encoder.layer.11.attention.output.dense.weight', 'bert.encoder.layer.11.attention.output.dense.bias', 'bert.encoder.layer.11.attention.output.LayerNorm.weight', 'bert.encoder.layer.11.attention.output.LayerNorm.bias', 'bert.encoder.layer.11.intermediate.dense.weight', 'bert.encoder.layer.11.intermediate.dense.bias', 'bert.encoder.layer.11.output.dense.weight', 'bert.encoder.layer.11.output.dense.bias', 'bert.encoder.layer.11.output.LayerNorm.weight', 'bert.encoder.layer.11.output.LayerNorm.bias', 'cls.predictions.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.decoder.weight', 'cls.predictions.decoder.bias'])
        """
        pass

    def forward(self, input: str):
        # TODO silly goose
        assert (
            AutoModelForMaskedLM.from_pretrained is not None
        )  # recommended by https://github.com/microsoft/pylance-release/issues/333#issuecomment-688522371
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        model = AutoModelForMaskedLM.from_pretrained(self.config.model_name)
        return model(**tokenizer(input, return_tensors="pt"))
