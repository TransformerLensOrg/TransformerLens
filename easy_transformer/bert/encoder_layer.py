import einops
import torch.nn as nn
from fancy_einsum import einsum
from torchtyping import TensorType as TT

from .EasyBERTConfig import EasyBERTConfig

# TODO delete unused files


class SelfAttention(nn.Module):
    def __init__(self, config: EasyBERTConfig):
        super().__init__()
        self.config = config
        # TODO someday make head_size distinct so that this module can be parallel
        self.w_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.w_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.w_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.w_o = nn.Linear(config.hidden_size, config.hidden_size)

    def attention_pattern(
        self, x: TT["batch", "seq", "hidden"]
    ) -> TT["batch", "head", "seq", "seq"]:
        # TODO double check that this is correct
        q = self.w_q(x)
        k = self.w_k(x)
        q = einops.rearrange(
            q,
            "batch seq (head head_size) -> batch head seq head_size",
            head=self.config.n_heads,
        )
        k = einops.rearrange(
            k,
            "batch seq (head head_size) -> batch head seq head_size",
            head=self.config.n_heads,
        )
        result = einsum(
            "batch head seq_q head_size, batch head seq_k head_size -> batch head seq_q seq_k",
            q,
            k,
        )
        return result / (self.config.hidden_size**0.5)

    def forward(
        self, x: TT["batch", "seq", "hidden"], mask=None
    ) -> TT["batch", "seq", "intermediate"]:
        # here we do the computation per head
        attention = (
            self.attention_pattern(x)
            if mask is None
            else self.attention_pattern(x) + mask
        )
        attention = attention.softmax(dim=-1)
        v = self.w_v(x)
        v = einops.rearrange(
            v,
            "b seq (head head_size) -> b head seq head_size",
            head=self.config.n_heads,
        )
        combined_values = einsum(
            "b head seq_k head_size, b head seq_q seq_k -> b head seq_q head_size",
            v,
            attention,
        )
        # now collapse back to the original shape
        rearranged = einops.rearrange(
            combined_values, "b head seq head_size -> b seq (head head_size)"
        )
        return self.w_o(rearranged)


class Attention(nn.Module):
    def __init__(self, config: EasyBERTConfig):
        super().__init__()
        self.self_attention = SelfAttention(config)
        self.dropout = nn.Dropout(config.dropout)
        self.ln = nn.LayerNorm(config.hidden_size)  # TODO double check the epsilon

    def forward(
        self, x: TT["batch", "seq", "hidden"], mask=None
    ) -> TT["batch", "seq", "hidden"]:
        original_x = x  # for a residual connection
        x = self.self_attention(x, mask=mask)
        x = self.dropout(x)
        return self.ln(x + original_x)


class MLP(nn.Module):
    def __init__(self, config: EasyBERTConfig):
        super().__init__()
        self.mlp_size = 4 * config.hidden_size  # TODO double check this
        self.w_1 = nn.Linear(config.hidden_size, self.mlp_size)  # aka 'up' layer
        self.gelu = nn.GELU()
        self.w_2 = nn.Linear(self.mlp_size, config.hidden_size)  # aka 'down' layer
        self.dropout = nn.Dropout(config.dropout)
        self.ln = nn.LayerNorm(config.hidden_size)  # TODO double check the epsilon

    def forward(self, x: TT["batch", "seq", "hidden"]) -> TT["batch", "seq", "hidden"]:
        original_x = x  # for a residual connection
        x = self.w_1(x)
        x = self.gelu(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return self.ln(x + original_x)


class EncoderLayer(nn.Module):
    def __init__(self, config: EasyBERTConfig):
        super().__init__()
        self.config = config
        self.attention = Attention(config)  # TODO rename
        self.mlp = MLP(config)

    # TODO confirm that's the right size
    def forward(
        self, x: TT["batch", "seq", "hidden"], mask=None
    ) -> TT["batch", "seq", "hidden"]:
        return self.mlp(self.attention(x, mask))
