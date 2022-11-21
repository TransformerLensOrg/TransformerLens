from typing import Dict, List, Optional, Tuple, Union

import torch
from torchtyping import TensorType as TT

from easy_transformer.hook_points import HookedRootModule

from ..past_key_value_caching import (
    EasyTransformerKeyValueCache,
    EasyTransformerKeyValueCacheEntry,
)

# TODO maybe have an __init__.py in bert/ that imports all the bert stuff



class EasyBERT(HookedRootModule):
    """
    TODO say less / say more about how we dont support this

    apologize for duplicated code and or take it out
    """

    # TODO add tensor types
    # TODO make [return_type] an enum
    def forward(
        self,
        input: Union[str, TT["batch", "pos"]],
        return_type: Optional[str] = "logits",
        prepend_bos: bool = True,
        past_kv_cache: Optional[EasyTransformerKeyValueCache] = None,
        token_types: Optional[torch.Tensor] = None,
        attention_mask: Optional[
            TT["batch", "pos", bool]
        ] = None,  # TODO take these out of [EasyTransformer.py] where appropriate
    ) -> Union[None, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # TODO jsmith read this more
        """
        [input] values of type [str] are automatically tokenized to a batch of a single element. [prepend_bos] only applies when inputting a text string.
        [token_types] is a tensor of the same shape as input, with each element being the token type of the corresponding token in [input]. If None, token types are assumed to be all 0s.
        In [attention_mask], an element is [True] if and only if the corresponding token in input is NOT a padding token. If [attention_mask == None], all tokens are assumed to be non-padding tokens (i.e. the mask is all [True]). Often formatted as a tensor of ints (False=0, True=1) rather than bools, but it's equivalent

        [return_type] : The type of output to return. Can be one of: None (return nothing, don't calculate logits), 'logits' (return logits), 'loss' (return cross-entropy loss), 'both' (return logits and loss)
        """
        if type(input) == str or type(input) == list:
            # If text, convert to tokens (batch_size=1)
            assert (
                self.tokenizer is not None
            ), "Must provide a tokenizer if passing a string to the model"
            # TODO "BERT-family models cannot be passed strings"
            tokens = self.to_tokens(input, prepend_bos=prepend_bos)
        else:
            tokens = input
        assert isinstance(tokens, torch.Tensor)
        # If we're doing caching, then we reuse keys and values from previous runs, as that's the only
        # way that past activations will affect the final logits. The cache contains those so we don't
        # need to recompute them. This is useful for generating text. As we have absolute positional
        # encodings, to implement this we have a `pos_offset` variable, defaulting to zero, which says
        # to offset which positional encodings are used (cached keys and values were calculated with
        # their own positional encodings).
        if past_kv_cache is None:
            pos_offset = 0
        else:
            batch_size, ctx_length = tokens.shape
            (
                cached_batch_size,
                cache_ctx_length,
                num_heads_in_cache,
                d_head_in_cache,
            ) = past_kv_cache[0].past_keys.shape
            assert cached_batch_size == batch_size
            assert num_heads_in_cache == self.cfg.n_heads
            assert d_head_in_cache == self.cfg.d_head
            # If we want to generate from the empty string, we'd pass in an empty cache, so we need to handle that case
            assert (
                cache_ctx_length == 0 or ctx_length == 1
            ), "Pass in one token at a time after loading cache"
            pos_offset = cache_ctx_length

        if self.cfg.bert_family:
            residual = self.bert_embed(tokens, token_types)
            shortformer_pos_embed = None

            for i, block in enumerate(self.blocks):
                # Note that each block includes skip connections, so we don't need residual + block(residual)
                residual = block(
                    residual,
                    past_kv_cache_entry=past_kv_cache[i]
                    if past_kv_cache is not None
                    else None,  # Cache is contains a list of EasyTransformerKeyValueCache objects, one for each block
                    shortformer_pos_embed=shortformer_pos_embed,
                    attention_mask=attention_mask,
                )  # [batch, pos, d_model]

        if self.cfg.bert_family:
            # BERT has a final linear layer that's different from the rest of the blocks
            residual = self.bert_final(residual)
        else:
            embed = self.hook_embed(self.embed(tokens))  # [batch, pos, d_model]
            pos_embed = self.hook_pos_embed(
                self.pos_embed(tokens, pos_offset)
            )  # [batch, pos, d_model]
            if self.cfg.positional_embedding_type != "shortformer":
                residual = embed + pos_embed  # [batch, pos, d_model]
                shortformer_pos_embed = None
            else:
                # If we're using shortformer style attention, we don't add the positional embedding to the residual stream. See EasyTransformerConfig for details
                residual = embed
                shortformer_pos_embed = pos_embed

        for i, block in enumerate(self.blocks):
            # Note that each block includes skip connections, so we don't need
            # residual + block(residual)
            residual = block(
                residual,
                past_kv_cache_entry=past_kv_cache[i]
                if past_kv_cache is not None
                else None,  # Cache is contains a list of EasyTransformerKeyValueCache objects, one for each block
                shortformer_pos_embed=shortformer_pos_embed,
                attention_mask=attention_mask,
            )  # [batch, pos, d_model]

        if self.cfg.bert_family:
            # BERT has a final linear layer that's different from the rest of the blocks
            residual = self.bert_final(residual)

        if return_type is None:
            return None
        else:

            residual = self.ln_final(residual)  # [batch, pos, d_vocab]
            logits = self.unembed(residual)  # [batch, pos, d_vocab]
            if return_type == "logits":
                return logits
            else:
                assert (
                    not self.cfg.bert_family
                ), "BERT-family models do not have a supported loss function"
                loss = lm_cross_entropy_loss(logits, tokens)
                if return_type == "loss":
                    return loss
                elif return_type == "both":
                    return {"logits": logits, "loss": loss}
                else:
                    logging.warning(f"Invalid return_type passed in: {return_type}")
                    return None
