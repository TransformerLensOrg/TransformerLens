import math
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from jaxtyping import Float, Int

from transformer_lens.components.abstract_attention import AbstractAttention
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class T5Attention(AbstractAttention):
    r"""
    T5 attention - with relative attention bias and cross-attention support
    This realisation expects you to precompute relative positional bias, and then feed it to forward
    like
    ```python
    attn = T5Attention(cfg, has_relative_attention_bias=True)
    positional_bias = attn.compute_relative_attention_bias(query_len, key_len, device=device)
    result = attn(query, key, value, position_bias=positional_bias)
    ```
    """

    def __init__(
        self,
        cfg: Union[Dict, HookedTransformerConfig],
        has_relative_attention_bias: bool = False,
        attn_type: str = "global",
        layer_id: Optional[int] = None,
    ):
        super().__init__(cfg, attn_type, layer_id)
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg
        self.has_relative_attention_bias: bool = has_relative_attention_bias

        if self.has_relative_attention_bias:
            if (
                cfg.relative_attention_num_buckets is None
                or cfg.relative_attention_max_distance is None
            ):
                raise ValueError(
                    "You need to specify relative_attention_num_buckets and relative_attention_max_distance  in config to use relative attention bias"
                )

            self.relative_attention_num_buckets = cfg.relative_attention_num_buckets
            self.relative_attention_max_distance = cfg.relative_attention_max_distance
            self.rel_pos_bias = nn.Embedding(self.relative_attention_num_buckets, self.cfg.n_heads)
            self.rel_pos_hook = HookPoint()

        self.W_K = nn.Parameter(
            torch.empty(self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head, dtype=cfg.dtype)
        )
        self.W_V = nn.Parameter(
            torch.empty(self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head, dtype=cfg.dtype)
        )
        self.b_K = nn.Parameter(torch.zeros(self.cfg.n_heads, self.cfg.d_head, dtype=cfg.dtype))
        self.b_V = nn.Parameter(torch.zeros(self.cfg.n_heads, self.cfg.d_head, dtype=cfg.dtype))

    @staticmethod
    def _relative_position_bucket(
        relative_position: Int[torch.Tensor, "query_pos kv_pos"],
        bidirectional=True,
        num_buckets=32,
        max_distance=128,
    ) -> Int[torch.Tensor, "query_pos kv_pos"]:
        """
        added from
        https://github.com/huggingface/transformers/blob/e0c3cee17085914bbe505c159beeb8ae39bc37dd/src/transformers/models/t5/modeling_t5.py#L382
        which is adapted from
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593


        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = torch.zeros_like(relative_position)

        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_relative_attention_bias(
        self, query_length: int, key_length: int, device=None
    ) -> Float[torch.Tensor, "1 head_index pos kv_pos"]:
        """Compute binned relative position bias"""
        if device is None:
            device = self.rel_pos_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.rel_pos_bias(
            relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, query_length, key_length)
        return values
