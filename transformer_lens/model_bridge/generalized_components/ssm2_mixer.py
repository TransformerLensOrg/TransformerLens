"""Wrap-don't-reimplement bridge for HF's Mamba2Mixer, plus SSD effective attention."""
from typing import Any

import torch

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class SSM2MixerBridge(GeneralizedComponent):
    """Opaque wrapper around Mamba-2's Mamba2Mixer.

    Structural differences from Mamba-1:
    - No x_proj/dt_proj; in_proj fuses gate, hidden_B_C, and dt into one output.
    - Has an inner norm (``MambaRMSNormGated``) taking two inputs; exposed at
      ``mixer.inner_norm`` (renamed from HF's ``norm``) to disambiguate from the
      block-level norm.
    - Multi-head with ``num_heads``, ``head_dim``, ``n_groups`` (GQA-like).
    - ``A_log``, ``dt_bias``, ``D`` are ``[num_heads]`` parameters reached via
      ``GeneralizedComponent.__getattr__`` delegation.

    Decode-step caveat: ``conv1d.hook_out`` fires only on prefill during
    stateful generation; see ``DepthwiseConv1DBridge`` for the reason.
    """

    hook_aliases = {
        "hook_in_proj": "in_proj.hook_out",
        "hook_conv": "conv1d.hook_out",
        "hook_inner_norm": "inner_norm.hook_out",
        "hook_ssm_out": "hook_out",
    }

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Hook the input, delegate to HF torch_forward, hook the output."""
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. "
                "Call set_original_component() first."
            )

        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            hooked = self.hook_in(args[0])
            args = (hooked,) + args[1:]
        elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
            kwargs["hidden_states"] = self.hook_in(kwargs["hidden_states"])

        output = self.original_component(*args, **kwargs)

        if isinstance(output, tuple) and len(output) > 0:
            first = output[0]
            if isinstance(first, torch.Tensor):
                return (self.hook_out(first),) + output[1:]
            return output
        if isinstance(output, torch.Tensor):
            return self.hook_out(output)
        return output

    def compute_effective_attention(
        self,
        cache: ActivationCache,
        layer_idx: int,
        include_dt_scaling: bool = False,
    ) -> torch.Tensor:
        """Materialize Mamba-2's effective attention matrix M = L ⊙ (C B^T).

        Via State Space Duality (SSD), Mamba-2's SSM is equivalent to causal
        attention with a per-step per-head learned decay — see "The Hidden
        Attention of Mamba" (Ali et al., ACL 2025). Extracts B, C from
        ``conv1d.hook_out`` (post conv + SiLU) and dt from ``in_proj.hook_out``,
        then reads ``A_log`` and ``dt_bias`` via ``__getattr__`` delegation.

        Args:
            cache: ActivationCache from ``run_with_cache`` containing the
                in_proj and conv1d hooks for this layer.
            layer_idx: Block index for this mixer. Required because submodule
                bridges don't know their own position in the block list.
            include_dt_scaling: False (default) returns the attention-like
                form M_att = L ⊙ (C B^T). True multiplies each column j by
                dt[j], giving the strict reconstruction form that satisfies
                ``y[i] = sum_j M[i,j] * x[j] + D * x[i]``.

        Returns:
            Tensor of shape ``[batch, num_heads, seq_len, seq_len]`` with the
            upper triangle (j > i) zeroed.

        Cost is O(batch · num_heads · seq_len²); use on short sequences (≤2k).
        """
        if self.config is None:
            raise RuntimeError("SSM2MixerBridge.config must be set")

        in_proj_key = f"blocks.{layer_idx}.mixer.in_proj.hook_out"
        conv1d_key = f"blocks.{layer_idx}.mixer.conv1d.hook_out"
        if in_proj_key not in cache or conv1d_key not in cache:
            raise RuntimeError(
                f"compute_effective_attention needs {in_proj_key!r} and "
                f"{conv1d_key!r} in cache. Run `run_with_cache()` on the bridge "
                "before calling this method."
            )

        cfg = self.config
        num_heads: int = cfg.n_heads
        head_dim: int = cfg.d_head
        intermediate_size: int = getattr(cfg, "intermediate_size", num_heads * head_dim)
        state_size: int = getattr(cfg, "state_size", 128)
        n_groups: int = getattr(cfg, "n_groups", 1)

        # Mirror HF's tuple convention so downstream equality checks stay consistent
        time_step_limit = getattr(cfg, "time_step_limit", (0.0, float("inf")))
        time_step_min = float(time_step_limit[0])
        time_step_max = float(time_step_limit[1])

        in_proj_out = cache[in_proj_key]  # [batch, seq, proj_size]
        conv1d_out = cache[conv1d_key]  # [batch, conv_dim, seq + conv_kernel - 1]
        batch_size, seq_len = in_proj_out.shape[0], in_proj_out.shape[1]

        # Match HF's SSM numerical precision
        in_proj_out_f = in_proj_out.float()
        conv1d_out_f = conv1d_out.float()

        # dt is the last num_heads features of in_proj output, post softplus+clamp
        dt_raw = in_proj_out_f[..., -num_heads:]
        dt_bias = self.dt_bias.float()
        dt = torch.nn.functional.softplus(dt_raw + dt_bias)
        dt = torch.clamp(dt, time_step_min, time_step_max)  # [batch, seq, num_heads]

        # B, C come from the conv1d output after trimming to seq_len and applying SiLU
        conv_trimmed = conv1d_out_f[..., :seq_len]
        conv_activated = torch.nn.functional.silu(conv_trimmed).transpose(1, 2)
        split_sizes = [intermediate_size, n_groups * state_size, n_groups * state_size]
        _hidden_x, B_flat, C_flat = conv_activated.split(split_sizes, dim=-1)
        B = B_flat.view(batch_size, seq_len, n_groups, state_size)
        C = C_flat.view(batch_size, seq_len, n_groups, state_size)

        # GQA-style: each of n_groups B/C pairs is replicated to cover n_heads // n_groups heads
        heads_per_group = num_heads // n_groups
        B_h = B.repeat_interleave(heads_per_group, dim=2)
        C_h = C.repeat_interleave(heads_per_group, dim=2)

        A = -torch.exp(self.A_log.float())  # [num_heads]

        # L[i, j] = exp(sum_{k=j+1}^{i} A[h] * dt[k, h]) for i >= j, else 0
        # Computed as exp(cumsum[i] - cumsum[j]) since cumsum[j] includes dt[j],
        # so the remaining sum runs from k=j+1 to k=i.
        log_a = dt * A[None, None, :]
        cumsum_log_a = torch.cumsum(log_a, dim=1)
        cs = cumsum_log_a.permute(0, 2, 1)  # [batch, num_heads, seq]
        L_log = cs[:, :, :, None] - cs[:, :, None, :]
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=L_log.device)
        )
        L = torch.where(
            causal_mask[None, None, :, :],
            torch.exp(L_log),
            torch.zeros_like(L_log),
        )

        # CB[b, h, i, j] = <C[b, i, h], B[b, j, h]>
        CB = torch.einsum("bihs,bjhs->bhij", C_h, B_h)

        M = L * CB  # [batch, num_heads, seq, seq]

        if include_dt_scaling:
            # Multiply column j by dt[j, h] to absorb the B discretization
            dt_col = dt.permute(0, 2, 1)[:, :, None, :]
            M = M * dt_col

        return M
