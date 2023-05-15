from typing import Dict, Optional

import torch
from einops import repeat
from jaxtyping import Int
from torch import nn

import transformer_lens.loading_from_pretrained as loading
from transformer_lens import HookedTransformerConfig
from transformer_lens.components import BertBlock, BertEmbed, BertMLMHead, Unembed
from transformer_lens.hook_points import HookedRootModule, HookPoint


class HookedEncoder(HookedRootModule):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig(**cfg)
        elif isinstance(cfg, str):
            raise ValueError(
                "Please pass in a config dictionary or HookedTransformerConfig object. If you want to load a pretrained model, use HookedEncoder.from_pretrained() instead."
            )
        self.cfg = cfg

        self.embed = BertEmbed(self.cfg)
        self.blocks = nn.ModuleList(
            [BertBlock(self.cfg) for _ in range(self.cfg.n_layers)]
        )
        self.mlm_head = BertMLMHead(cfg)
        self.unembed = Unembed(self.cfg)

        self.hook_full_embed = HookPoint()

        self.setup()

    def forward(
        self,
        x: Int[torch.Tensor, "batch pos"],
        token_type_ids=None,
        one_zero_attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
    ):
        resid = self.hook_full_embed(self.embed(x, token_type_ids))

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

        return logits

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        checkpoint_index=None,
        checkpoint_value=None,
        hf_model=None,
        device=None,
        n_devices=1,
        move_state_dict_to_device=True,
        **model_kwargs,
    ):
        official_model_name = loading.get_official_model_name(model_name)

        cfg = loading.get_pretrained_model_config(
            official_model_name,
            checkpoint_index=checkpoint_index,
            checkpoint_value=checkpoint_value,
            # TODO: implement layernorm folding?
            fold_ln=False,
            device=device,
            n_devices=n_devices,
        )

        state_dict = loading.get_pretrained_state_dict(
            official_model_name, cfg, hf_model
        )

        model = cls(cfg, **model_kwargs)
        model.load_and_process_state_dict(state_dict)

        print(f"Loaded pretrained model {model_name} into HookedTransformer")

        return model

    def load_and_process_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        # TODO: Add in preprocessing
        # TODO: fill in missing keys rather than using strict=False
        self.load_state_dict(state_dict, strict=False)

    # def run_with_cache(
    #     self, *model_args, return_cache_object=True, remove_batch_dim=False, **kwargs
    # ):
    #     out, cache_dict = super().run_with_cache(
    #         *model_args, remove_batch_dim=remove_batch_dim, **kwargs
    #     )
    #     return out, cache_dict
