from torch import nn
import torch
import torch.nn.functional as F
import einops
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens.HookedSAEConfig import HookedSAEConfig

from typing import Union, Dict


class HookedSAE(HookedRootModule):
    """Hooked AutoEncoder.
    
    Implements a standard SAE with a TransformerLens hooks for SAE activations
    
    Should probably just be used for inference / analysis, not training.
    
    This is just a template, feel free to copy this into a notebook and change it!
    """
    def __init__(self, cfg: Union[HookedSAEConfig, Dict]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedSAEConfig(**cfg)
        elif isinstance(cfg, str):
            raise ValueError(
                "Please pass in a config dictionary or HookedSAEConfig object. If you want to load a "
                "pretrained SAE, use HookedSAE.from_pretrained() instead."
            )
        self.cfg = cfg

        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.cfg.d_in, self.cfg.d_sae, dtype=self.cfg.dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.cfg.d_sae, self.cfg.d_in, dtype=self.cfg.dtype)))
        self.b_enc = nn.Parameter(torch.zeros(self.cfg.d_sae, dtype=self.cfg.dtype))
        self.b_dec = nn.Parameter(torch.zeros(self.cfg.d_in, dtype=self.cfg.dtype))
        
        self.hook_sae_input = HookPoint()
        self.hook_sae_acts_pre = HookPoint()
        self.hook_sae_acts_post = HookPoint() # Hook point for the SAEs hidden activations
        self.hook_sae_recons = HookPoint()
        self.hook_sae_error = HookPoint()
        self.hook_sae_output = HookPoint()

        self.to(self.cfg.device)
    
    def forward(self, input: torch.tensor):
        """SAE Forward Pass.
        
        Args:
            input: The input tensor of activations to the SAE. Shape [..., d_in]
        """
        self.hook_sae_input(input)
        if input.shape[-1] == self.cfg.d_in:
            x = input
        else:
            x = einops.rearrange(input, "... n_heads d_head -> ... (n_heads d_head)")
        assert x.shape[-1] == self.cfg.d_in, f"Input shape {x.shape} does not match SAE input size {self.cfg.d_in}"
    
        x_cent = x - self.b_dec
        sae_acts_pre = self.hook_sae_acts_pre(
            einops.einsum(
                x_cent, self.W_enc,
                "... d_in, d_in d_sae -> ... d_sae"  
            ) + self.b_enc # [..., d_sae]
        )
        sae_acts_post = self.hook_sae_acts_post(F.relu(sae_acts_pre)) # [..., d_sae]
        x_reconstruct = self.hook_sae_recons(
            (einops.einsum(
                sae_acts_post, self.W_dec,
                "... d_sae, d_sae d_in -> ... d_in"
            ) + self.b_dec).reshape(input.shape)
        )
        
        with torch.no_grad():
            # Recompute everything without hooks to get true error term
            sae_acts_pre_clean = einops.einsum(
                x_cent, self.W_enc,
                "... d_in, d_in d_sae -> ... d_sae"  
            ) + self.b_enc # [..., d_sae]
            sae_acts_post_clean = F.relu(sae_acts_pre_clean)
            x_reconstruct_clean = (einops.einsum(
                sae_acts_post_clean, self.W_dec,
                "... d_sae, d_sae d_in -> ... d_in"
            ) + self.b_dec).reshape(input.shape)
            
            error = self.hook_sae_error(input - x_reconstruct_clean)
  
        output = self.hook_sae_output(x_reconstruct + error)
        if self.cfg.use_error_term:
            return output

        return x_reconstruct
    