"""Mock models for testing."""

import torch.nn as nn


class MockGemma3Model(nn.Module):
    """A mock implementation of the Gemma 3 model architecture for testing purposes.
    
    This mock model replicates the key architectural components of Gemma 3:
    - Embedding layer (embed_tokens)
    - Multiple transformer layers with:
        - Input and post-attention layer norms
        - Self-attention with Q, K, V, O projections
        - MLP with up, gate, and down projections
    - Final layer norm
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(1000, 512)
        self.model.layers = nn.ModuleList([
            nn.Module() for _ in range(2)
        ])
        for layer in self.model.layers:
            layer.input_layernorm = nn.LayerNorm(512)
            layer.post_attention_layernorm = nn.LayerNorm(512)
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = nn.Linear(512, 512)
            layer.self_attn.k_proj = nn.Linear(512, 512)
            layer.self_attn.v_proj = nn.Linear(512, 512)
            layer.self_attn.o_proj = nn.Linear(512, 512)
            layer.mlp = nn.Module()
            layer.mlp.up_proj = nn.Linear(512, 2048)
            layer.mlp.gate_proj = nn.Linear(512, 2048)
            layer.mlp.down_proj = nn.Linear(2048, 512)
        self.model.norm = nn.LayerNorm(512)
        self.embed_tokens = self.model.embed_tokens  # For shared embedding/unembedding 