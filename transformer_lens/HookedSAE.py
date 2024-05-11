from typing import Dict, Union

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import nn

from transformer_lens.hook_points import (  # Hooking utilities
    HookedRootModule,
    HookPoint,
)
from transformer_lens.HookedSAEConfig import HookedSAEConfig


class HookedSAE(HookedRootModule):
    """Hooked SAE.

    Implements a standard SAE with a TransformerLens hooks for SAE activations

    Designed for inference / analysis, not training. For training, see Joseph Bloom's SAELens (https://github.com/jbloomAus/SAELens)

    Note that HookedSAETransformer is fairly modular, and doesn't make strong assumptions about the architecture of the SAEs that get attached. We provide HookedSAE as a useful default class, but if you want to eg experiment with other SAE architectures, you can just copy the HookedSAE code into a notebook, edit it, and add instances of the new SAE class to a HookedSAETransformer (e.g. with HookedSAETransformer.add_sae(sae))
    """

    def __init__(self, cfg: Union[HookedSAEConfig, Dict]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedSAEConfig(**cfg)
        elif isinstance(cfg, str):
            raise ValueError("Please pass in a config dictionary or HookedSAEConfig object.")
        self.cfg = cfg

        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg.d_in, self.cfg.d_sae, dtype=self.cfg.dtype)
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.cfg.d_sae, self.cfg.d_in, dtype=self.cfg.dtype)
            )
        )
        self.b_enc = nn.Parameter(torch.zeros(self.cfg.d_sae, dtype=self.cfg.dtype))
        self.b_dec = nn.Parameter(torch.zeros(self.cfg.d_in, dtype=self.cfg.dtype))

        self.hook_sae_input = HookPoint()
        self.hook_sae_acts_pre = HookPoint()
        self.hook_sae_acts_post = HookPoint()
        self.hook_sae_recons = HookPoint()
        self.hook_sae_error = HookPoint()
        self.hook_sae_output = HookPoint()

        self.to(self.cfg.device)
        self.setup()

    def forward(self, input: Float[torch.Tensor, "... d_in"]) -> Float[torch.Tensor, "... d_in"]:
        """SAE Forward Pass.

        Args:
            input: The input tensor of activations to the SAE. Shape [..., d_in].
                Also supports hook_z activations of shape [..., n_heads, d_head], where n_heads * d_head = d_in, for attention output (hook_z) SAEs.

        Returns:
            output: The reconstructed output tensor from the SAE, with the error term optionally added. Same shape as input (eg [..., d_in])
        """
        self.hook_sae_input(input)
        if input.shape[-1] == self.cfg.d_in:
            x = input
        else:
            # Assume this this is an attention output (hook_z) SAE
            assert self.cfg.hook_name.endswith(
                "_z"
            ), f"You passed in an input shape {input.shape} does not match SAE input size {self.cfg.d_in} for hook_name {self.cfg.hook_name}. This is only supported for attn output (hook_z) SAEs."
            x = einops.rearrange(input, "... n_heads d_head -> ... (n_heads d_head)")
        assert (
            x.shape[-1] == self.cfg.d_in
        ), f"Input shape {x.shape} does not match SAE input size {self.cfg.d_in}"

        x_cent = x - self.b_dec
        # WARNING: if editing this block of code, also edit the error computation inside `if self.cfg.use_error_term`
        sae_acts_pre = self.hook_sae_acts_pre(
            einops.einsum(x_cent, self.W_enc, "... d_in, d_in d_sae -> ... d_sae")
            + self.b_enc  # [..., d_sae]
        )
        sae_acts_post = self.hook_sae_acts_post(F.relu(sae_acts_pre))  # [..., d_sae]
        x_reconstruct = self.hook_sae_recons(
            (
                einops.einsum(sae_acts_post, self.W_dec, "... d_sae, d_sae d_in -> ... d_in")
                + self.b_dec
            ).reshape(input.shape)
        )
        # END WARNING

        if self.cfg.use_error_term:
            with torch.no_grad():
                # Recompute everything without hooks to get true error term
                # Otherwise, the output with error term will always equal input, even for causal interventions that affect x_reconstruct
                # This is in a no_grad context to detach the error, so we can compute SAE feature gradients (eg for attribution patching). See A.3 in https://arxiv.org/pdf/2403.19647.pdf for more detail
                # NOTE: we can't just use `sae_error = input - x_reconstruct.detach()` or something simpler, since this would mean intervening on features would mean ablating features still results in perfect reconstruction.
                sae_acts_pre_clean = (
                    einops.einsum(x_cent, self.W_enc, "... d_in, d_in d_sae -> ... d_sae")
                    + self.b_enc
                )  # [..., d_sae]
                sae_acts_post_clean = F.relu(sae_acts_pre_clean)
                x_reconstruct_clean = (
                    einops.einsum(
                        sae_acts_post_clean,
                        self.W_dec,
                        "... d_sae, d_sae d_in -> ... d_in",
                    )
                    + self.b_dec
                ).reshape(input.shape)

                sae_error = self.hook_sae_error(input - x_reconstruct_clean)
            return self.hook_sae_output(x_reconstruct + sae_error)

        return self.hook_sae_output(x_reconstruct)
