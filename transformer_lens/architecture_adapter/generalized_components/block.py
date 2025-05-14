import torch
import torch.nn as nn


class BlockBridge(nn.Module):
    def __init__(self, ln1, attn, ln2, mlp, original_component=None):
        super().__init__()
        self.ln1 = ln1
        self.attn = attn
        self.ln2 = ln2
        self.mlp = mlp
        self.original_component = original_component  # Optionally keep reference to original block

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        x = self.ln1(hidden_states)
        attn_out = self.attn(x, attention_mask=attention_mask, position_ids=position_ids, **kwargs)
        x = x + attn_out
        x_ln2 = self.ln2(x)
        mlp_out = self.mlp(x_ln2)
        x = x + mlp_out
        return x 