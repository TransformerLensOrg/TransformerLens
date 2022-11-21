import logging
from typing import Dict

import torch
from transformers import (
    AutoModelForMaskedLM,  # unfortunately the suggestion to import from the non-private location doesn't work; it makes [from_pretrained == None]
)
from transformers.models.auto.tokenization_auto import AutoTokenizer

from easy_transformer.hook_points import HookedRootModule

from .. import loading_from_pretrained as loading
from . import EasyBERTConfig, embeddings, encoder


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
            max_len=512,
        )  # TODO fancier :P
        assert (
            AutoModelForMaskedLM.from_pretrained is not None
        )  # recommended by some github link TODO find it
        state_dict = AutoModelForMaskedLM.from_pretrained(
            official_model_name
        ).state_dict()
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
        self.encoder = encoder.Encoder(config)

    def __load_embedding_state_dict__(self, state_dict: Dict[str, torch.Tensor]):
        self.embeddings.word_embeddings.load_state_dict(
            {"weight": state_dict["bert.embeddings.word_embeddings.weight"]}
        )
        self.embeddings.position_embeddings.load_state_dict(
            {"weight": state_dict["bert.embeddings.position_embeddings.weight"]}
        )
        self.embeddings.token_type_embeddings.load_state_dict(
            {"weight": state_dict["bert.embeddings.token_type_embeddings.weight"]}
        )
        self.embeddings.ln.load_state_dict(
            {
                "weight": state_dict["bert.embeddings.LayerNorm.weight"],
                "bias": state_dict["bert.embeddings.LayerNorm.bias"],
            }
        )

    def __load_cls_state_dict__(self, state_dict: Dict[str, torch.Tensor]):
        """
        'cls.predictions.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.decoder.weight', 'cls.predictions.decoder.bias'])"""
        pass  # TODO do we need to do this? I think we do

    def __load_layer_state_dict__(
        self, layer_index: int, state_dict: Dict[str, torch.Tensor]
    ):
        # TODO
        base_name = f"bert.encoder.layer.{layer_index}."

        attention = self.encoder.layers[layer_index].attention

        assert isinstance(attention, encoder.MultiHeadAttention)

        attention.linear_layers[0].load_state_dict(
            {
                "weight": state_dict[base_name + "attention.self.query.weight"],
                "bias": state_dict[base_name + "attention.self.query.bias"],
            }
        )

        attention.linear_layers[1].load_state_dict(
            {
                "weight": state_dict[base_name + "attention.self.key.weight"],
                "bias": state_dict[base_name + "attention.self.key.bias"],
            }
        )

        attention.linear_layers[2].load_state_dict(
            {
                "weight": state_dict[base_name + "attention.self.value.weight"],
                "bias": state_dict[base_name + "attention.self.value.bias"],
            }
        )

        # TODO add layer norm weights

    def __load_encoder_state_dict__(self, state_dict: Dict[str, torch.Tensor]):
        for layer_index in range(self.config.n_layers):
            self.__load_layer_state_dict__(layer_index, state_dict)

    def load_and_process_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.__load_embedding_state_dict__(state_dict)
        self.__load_encoder_state_dict__(state_dict)
        self.__load_cls_state_dict__(state_dict)

    def forward(self, x, segment_info):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        embedded = self.embeddings(input, segment_info)
        encoded = self.encoder(embedded, mask)
        return encoded
