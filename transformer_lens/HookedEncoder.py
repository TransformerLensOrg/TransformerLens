from __future__ import annotations
from functools import lru_cache

import logging
from typing import Dict, Literal, Optional, Tuple, Union, cast, overload

import torch
from einops import repeat
from jaxtyping import Float, Int
from torch import nn
from transformers import AutoTokenizer
from typeguard import typeguard_ignore

import transformer_lens.loading_from_pretrained as loading
from transformer_lens import ActivationCache, FactoredMatrix, HookedTransformerConfig
from transformer_lens.components import BertBlock, BertEmbed, BertMLMHead, Unembed
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.utilities import devices


class HookedEncoder(HookedRootModule):
    def __init__(self, cfg, tokenizer=None, move_to_device=True, **kwargs):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig(**cfg)
        elif isinstance(cfg, str):
            raise ValueError(
                "Please pass in a config dictionary or HookedTransformerConfig object. If you want to load a pretrained model, use HookedEncoder.from_pretrained() instead."
            )
        self.cfg = cfg

        assert (
            self.cfg.n_devices == 1
        ), "Multiple devices not supported for HookedEncoder"
        if move_to_device:
            self.to(self.cfg.device)

        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif self.cfg.tokenizer_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer_name)
        else:
            self.tokenizer = None

        if self.cfg.d_vocab == -1:
            # If we have a tokenizer, vocab size can be inferred from it.
            assert (
                self.tokenizer is not None
            ), "Must provide a tokenizer if d_vocab is not provided"
            self.cfg.d_vocab = max(self.tokenizer.vocab.values()) + 1
        if self.cfg.d_vocab_out == -1:
            self.cfg.d_vocab_out = self.cfg.d_vocab

        self.embed = BertEmbed(self.cfg)
        self.blocks = nn.ModuleList(
            [BertBlock(self.cfg) for _ in range(self.cfg.n_layers)]
        )
        self.mlm_head = BertMLMHead(cfg)
        self.unembed = Unembed(self.cfg)

        self.hook_full_embed = HookPoint()

        self.setup()

    @overload
    def forward(
        self,
        input: Int[torch.Tensor, "batch pos"],
        return_type: Literal["logits"],
        token_type_ids=None,
        one_zero_attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
    ) -> Float[torch.Tensor, "batch pos d_vocab"]:
        ...

    @overload
    def forward(
        self,
        input: Int[torch.Tensor, "batch pos"],
        return_type: Literal[None],
        token_type_ids=None,
        one_zero_attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
    ) -> Optional[Float[torch.Tensor, "batch pos d_vocab"]]:
        ...

    def forward(
        self,
        input: Int[torch.Tensor, "batch pos"],
        return_type: Literal[None, "logits"] = "logits",
        token_type_ids=None,
        one_zero_attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
    ) -> Optional[Float[torch.Tensor, "batch pos d_vocab"]]:
        tokens = input
        if tokens.device.type != self.cfg.device:
            tokens = tokens.to(self.cfg.device)
            if one_zero_attention_mask is not None:
                one_zero_attention_mask = one_zero_attention_mask.to(self.cfg.device)

        resid = self.hook_full_embed(self.embed(tokens, token_type_ids))

        large_negative_number = -1e5
        additive_attention_mask = (
            large_negative_number
            * repeat(1 - one_zero_attention_mask, "batch pos -> batch 1 1 pos")
            if one_zero_attention_mask is not None
            else None
        )

        for block in self.blocks:
            resid = block(resid, additive_attention_mask)
        resid = self.mlm_head(resid)
        logits = self.unembed(resid)

        if return_type is None:
            return
        return logits

    @overload
    def run_with_cache(
        self, *model_args, return_cache_object: Literal[True] = True, **kwargs
    ) -> Tuple[Float[torch.Tensor, "batch pos d_vocab"], ActivationCache]:
        ...

    @overload
    def run_with_cache(
        self, *model_args, return_cache_object: Literal[False] = False, **kwargs
    ) -> Tuple[Float[torch.Tensor, "batch pos d_vocab"], Dict[str, torch.Tensor]]:
        ...

    def run_with_cache(
        self, *model_args, return_cache_object=True, remove_batch_dim=False, **kwargs
    ) -> Tuple[
        Float[torch.Tensor, "batch pos d_vocab"],
        Union[ActivationCache, Dict[str, torch.Tensor]],
    ]:
        """
        Wrapper around run_with_cache in HookedRootModule. If return_cache_object is True, this will return an
        ActivationCache object, with a bunch of useful HookedTransformer specific methods, otherwise it will return a
        dictionary of activations as in HookedRootModule.
        """
        out, cache_dict = super().run_with_cache(
            *model_args, remove_batch_dim=remove_batch_dim, **kwargs
        )
        if return_cache_object:
            cache = ActivationCache(
                cache_dict, self, has_batch_dim=not remove_batch_dim
            )
            return out, cache
        else:
            return out, cache_dict

    def to(self, device_or_dtype, print_details=True):
        return devices.move_to_and_update_config(self, device_or_dtype, print_details)

    def cuda(self):
        # Wrapper around cuda that also changes self.cfg.device
        return self.to("cuda")

    def cpu(self):
        # Wrapper around cuda that also changes self.cfg.device
        return self.to("cpu")

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        checkpoint_index=None,
        checkpoint_value=None,
        hf_model=None,
        device=None,
        **model_kwargs,
    ) -> HookedEncoder:
        logging.warning(
            "HookedEncoder is still in beta. Please be aware that model preprocessing "
            "(e.g. LayerNorm folding) is not yet supported and backward compatibility "
            "is not guaranteed."
        )

        official_model_name = loading.get_official_model_name(model_name)

        cfg = loading.get_pretrained_model_config(
            official_model_name,
            checkpoint_index=checkpoint_index,
            checkpoint_value=checkpoint_value,
            fold_ln=False,
            device=device,
            n_devices=1,
        )

        state_dict = loading.get_pretrained_state_dict(
            official_model_name, cfg, hf_model
        )

        model = cls(cfg, **model_kwargs)

        model.load_state_dict(state_dict, strict=False)

        print(f"Loaded pretrained model {model_name} into HookedTransformer")

        return model

    @property
    @typeguard_ignore
    def W_U(self) -> Float[torch.Tensor, "d_model d_vocab"]:
        """
        Convenience to get the unembedding matrix (ie the linear map from the final residual stream to the output logits)
        """
        return self.unembed.W_U

    @property
    @typeguard_ignore
    def b_U(self) -> Float[torch.Tensor, "d_vocab"]:
        return self.unembed.b_U

    @property
    @typeguard_ignore
    def W_E(self) -> Float[torch.Tensor, "d_vocab d_model"]:
        """
        Convenience to get the embedding matrix
        """
        return self.embed.embed.W_E

    @property
    @typeguard_ignore
    def W_pos(self) -> Float[torch.Tensor, "n_ctx d_model"]:
        """
        Convenience function to get the positional embedding. Only works on models with absolute positional embeddings!
        """
        return self.embed.pos_embed.W_pos

    @property
    @typeguard_ignore
    def W_E_pos(self) -> Float[torch.Tensor, "d_vocab+n_ctx d_model"]:
        """
        Concatenated W_E and W_pos. Used as a full (overcomplete) basis of the input space, useful for full QK and full OV circuits.
        """
        return torch.cat([self.W_E, self.W_pos], dim=0)

    @property
    @typeguard_ignore
    @lru_cache(maxsize=None)
    def W_K(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the key weights across all layers"""
        return torch.stack(
            [cast(BertBlock, block).attn.W_K for block in self.blocks], dim=0
        )

    @property
    @typeguard_ignore
    @lru_cache(maxsize=None)
    def W_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the query weights across all layers"""
        return torch.stack(
            [cast(BertBlock, block).attn.W_Q for block in self.blocks], dim=0
        )

    @property
    @typeguard_ignore
    @lru_cache(maxsize=None)
    def W_V(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the value weights across all layers"""
        return torch.stack(
            [cast(BertBlock, block).attn.W_V for block in self.blocks], dim=0
        )

    @property
    @typeguard_ignore
    @lru_cache(maxsize=None)
    def W_O(self) -> Float[torch.Tensor, "n_layers n_heads d_head d_model"]:
        """Stacks the attn output weights across all layers"""
        return torch.stack(
            [cast(BertBlock, block).attn.W_O for block in self.blocks], dim=0
        )

    @property
    @typeguard_ignore
    @lru_cache(maxsize=None)
    def W_in(self) -> Float[torch.Tensor, "n_layers d_model d_mlp"]:
        """Stacks the MLP input weights across all layers"""
        return torch.stack(
            [cast(BertBlock, block).mlp.W_in for block in self.blocks], dim=0
        )

    @property
    @typeguard_ignore
    @lru_cache(maxsize=None)
    def W_out(self) -> Float[torch.Tensor, "n_layers d_mlp d_model"]:
        """Stacks the MLP output weights across all layers"""
        return torch.stack(
            [cast(BertBlock, block).mlp.W_out for block in self.blocks], dim=0
        )

    @property
    @typeguard_ignore
    @lru_cache(maxsize=None)
    def b_K(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the key biases across all layers"""
        return torch.stack(
            [cast(BertBlock, block).attn.b_K for block in self.blocks], dim=0
        )

    @property
    @typeguard_ignore
    @lru_cache(maxsize=None)
    def b_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the query biases across all layers"""
        return torch.stack(
            [cast(BertBlock, block).attn.b_Q for block in self.blocks], dim=0
        )

    @property
    @typeguard_ignore
    @lru_cache(maxsize=None)
    def b_V(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the value biases across all layers"""
        return torch.stack(
            [cast(BertBlock, block).attn.b_V for block in self.blocks], dim=0
        )

    @property
    @typeguard_ignore
    @lru_cache(maxsize=None)
    def b_O(self) -> Float[torch.Tensor, "n_layers d_model"]:
        """Stacks the attn output biases across all layers"""
        return torch.stack(
            [cast(BertBlock, block).attn.b_O for block in self.blocks], dim=0
        )

    @property
    @typeguard_ignore
    @lru_cache(maxsize=None)
    def b_in(self) -> Float[torch.Tensor, "n_layers d_mlp"]:
        """Stacks the MLP input biases across all layers"""
        return torch.stack(
            [cast(BertBlock, block).mlp.b_in for block in self.blocks], dim=0
        )

    @property
    @typeguard_ignore
    @lru_cache(maxsize=None)
    def b_out(self) -> Float[torch.Tensor, "n_layers d_model"]:
        """Stacks the MLP output biases across all layers"""
        return torch.stack(
            [cast(BertBlock, block).mlp.b_out for block in self.blocks], dim=0
        )

    @property
    @typeguard_ignore
    def QK(self):
        return FactoredMatrix(self.W_Q, self.W_K.transpose(-2, -1))

    @property
    @typeguard_ignore
    def OV(self):
        return FactoredMatrix(self.W_V, self.W_O)

    def all_head_labels(self):
        return [
            f"L{l}H{h}"
            for l in range(self.cfg.n_layers)
            for h in range(self.cfg.n_heads)
        ]
