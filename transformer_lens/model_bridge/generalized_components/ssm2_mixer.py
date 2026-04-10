"""Mamba-2 mixer bridge component.

Wraps HF's Mamba2Mixer as an opaque component with hook_in/hook_out. Submodule
bridges (in_proj, conv1d, inner_norm, out_proj) are swapped into the HF mixer's
module tree by `replace_remote_component`, so they fire automatically when the
HF forward accesses those attributes.

Structural differences from Mamba-1:
- No `x_proj` or `dt_proj` — Mamba-2's in_proj produces gate + hidden_B_C + dt
  in a single projection. The effective split is 3-way; HF's 5-way split has
  two `d_mlp` slots that are always size 0 in current configs.
- Has an inner norm (`MambaRMSNormGated`) that takes two inputs
  (hidden_states, gate) — wrapped by `GatedRMSNormBridge` and exposed at
  `mixer.inner_norm` (renamed from HF's `norm` for clarity vs the block norm).
- Multi-head structure: `num_heads`, `head_dim`, `n_groups` (GQA-like).
- `nn.Parameter`s: `A_log [num_heads]`, `dt_bias [num_heads]`, `D [num_heads]`
  — all accessible via GeneralizedComponent's __getattr__ fallback.

Also provides `compute_effective_attention()` — the Mamba-2 SSD interpretability
feature that materializes the T×T attention-equivalent matrix `M = L ⊙ (C B^T)`
for direct comparison with transformer attention patterns.
"""
from typing import Any, Mapping

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class SSM2MixerBridge(GeneralizedComponent):
    """Bridge component for Mamba-2 Mamba2Mixer.

    Separate from SSMMixerBridge because Mamba-2's mixer has a fundamentally
    different submodule set (no x_proj/dt_proj, has inner_norm/dt_bias).
    """

    hook_aliases = {
        "hook_in_proj": "in_proj.hook_out",
        "hook_conv": "conv1d.hook_out",
        "hook_inner_norm": "inner_norm.hook_out",
        "hook_ssm_out": "hook_out",
    }

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass: hook_in → HF mixer → hook_out.

        The HF mixer's torch_forward calls `self.in_proj`, `self.conv1d`,
        `self.norm`, `self.out_proj` — each swapped for a bridge submodule —
        so submodule hooks fire automatically inside the opaque forward.
        """
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
        cache: Mapping[str, torch.Tensor],
        layer_idx: int,
        include_dt_scaling: bool = False,
    ) -> torch.Tensor:
        """Materialize Mamba-2's effective attention matrix M = L ⊙ (C B^T).

        Exploits the State Space Duality (SSD) framework: Mamba-2's SSM layer
        can be expressed as a quadratic form equivalent to causal attention with
        a per-step, per-head learned decay. This is the key interpretability
        feature for Mamba-2 — it lets researchers compare SSM layers directly
        to transformer attention patterns (see "The Hidden Attention of Mamba",
        Ali et al., ACL 2025).

        Requires cached activations from `run_with_cache()` on the bridge;
        specifically `blocks.{layer_idx}.mixer.in_proj.hook_out` (to extract dt)
        and `blocks.{layer_idx}.mixer.conv1d.hook_out` (to extract B, C after
        conv+SiLU). Reads A_log and dt_bias directly from the wrapped HF mixer
        via the GeneralizedComponent `__getattr__` fallback.

        Args:
            cache: ActivationCache or dict from ``run_with_cache``.
            layer_idx: Which block this mixer belongs to (used to look up cache
                keys). Submodule bridges don't know their own layer index
                because `self.name` only stores the local remote path.
            include_dt_scaling: If False (default), returns the "attention-like"
                matrix ``M_att = L ⊙ (C B^T)`` — the form used in the Hidden
                Attention paper for interpretability. If True, multiplies each
                column j by ``dt[j]``, giving the strict reconstruction form:
                ``y[i] = sum_j M[i, j] * x[j] + D * x[i]`` where x is the
                post-conv, pre-dt-scaling hidden_states. Use True for
                verifying reconstruction against the actual SSM output.

        Returns:
            Tensor of shape ``[batch, num_heads, seq_len, seq_len]``. Upper
            triangle (j > i) is zero by construction.

        Cost:
            O(batch · num_heads · seq_len²). The dominant memory is the L and
            M matrices, each ``batch × num_heads × seq_len²`` floats. Use on
            short sequences (<= 2k) for analysis; not designed for training.

        Raises:
            RuntimeError: If config is not set or required cache keys are missing.
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

        # Time step clamp limits (default [0, inf] for standard configs)
        time_step_limit = getattr(cfg, "time_step_limit", [0.0, float("inf")])
        time_step_min = float(time_step_limit[0])
        time_step_max = float(time_step_limit[1])

        in_proj_out = cache[in_proj_key]  # [batch, seq, proj_size]
        conv1d_out = cache[conv1d_key]  # [batch, conv_dim, seq + conv_kernel - 1]
        batch_size, seq_len = in_proj_out.shape[0], in_proj_out.shape[1]

        # Promote everything to float32 to match HF's SSM path precision.
        in_proj_out_f = in_proj_out.float()
        conv1d_out_f = conv1d_out.float()

        # --- Extract dt from in_proj output (last num_heads features) ---
        dt_raw = in_proj_out_f[..., -num_heads:]  # [batch, seq, num_heads]
        dt_bias = self.dt_bias.float()  # [num_heads]
        dt = torch.nn.functional.softplus(dt_raw + dt_bias)
        dt = torch.clamp(dt, time_step_min, time_step_max)  # [batch, seq, num_heads]

        # --- Extract B, C from post-conv hidden_states_B_C ---
        # conv1d output is pre-trim; trim to seq_len, apply SiLU, transpose back
        conv_trimmed = conv1d_out_f[..., :seq_len]  # [batch, conv_dim, seq]
        conv_activated = torch.nn.functional.silu(conv_trimmed).transpose(1, 2)
        # [batch, seq, conv_dim] where conv_dim = intermediate_size + 2*n_groups*state_size
        split_sizes = [intermediate_size, n_groups * state_size, n_groups * state_size]
        _hidden_x, B_flat, C_flat = conv_activated.split(split_sizes, dim=-1)
        # B, C: [batch, seq, n_groups, state_size]
        B = B_flat.view(batch_size, seq_len, n_groups, state_size)
        C = C_flat.view(batch_size, seq_len, n_groups, state_size)
        # Replicate groups to heads (GQA-style): [batch, seq, num_heads, state_size]
        heads_per_group = num_heads // n_groups
        B_h = B.repeat_interleave(heads_per_group, dim=2)
        C_h = C.repeat_interleave(heads_per_group, dim=2)

        # --- A = -exp(A_log) ---
        A = -torch.exp(self.A_log.float())  # [num_heads]

        # --- Compute L (per-step decay) ---
        # log_a[b, t, h] = A[h] * dt[b, t, h]
        log_a = dt * A[None, None, :]  # [batch, seq, num_heads]
        cumsum_log_a = torch.cumsum(log_a, dim=1)  # [batch, seq, num_heads]
        # Permute to [batch, num_heads, seq] for pairwise diff.
        cs = cumsum_log_a.permute(0, 2, 1)
        # L[b, h, i, j] = exp(cumsum[b, h, i] - cumsum[b, h, j]) for i >= j.
        # Derivation: sum_{k=j+1}^{i} log_a[k] = cumsum[i] - cumsum[j] (since cumsum[j]
        # includes log_a[j], so the remaining sum is from k=j+1 to k=i).
        L_log = cs[:, :, :, None] - cs[:, :, None, :]  # [batch, num_heads, seq_i, seq_j]
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=L_log.device)
        )
        L = torch.where(
            causal_mask[None, None, :, :],
            torch.exp(L_log),
            torch.zeros_like(L_log),
        )

        # --- Compute C @ B^T bilinear form per head ---
        # CB[b, h, i, j] = sum_s C[b, i, h, s] * B[b, j, h, s]
        CB = torch.einsum("bihs,bjhs->bhij", C_h, B_h)

        # --- Final attention matrix ---
        M = L * CB  # [batch, num_heads, seq, seq]

        if include_dt_scaling:
            # Discretize B by multiplying each column j by dt[j, h]
            dt_col = dt.permute(0, 2, 1)[:, :, None, :]  # [batch, num_heads, 1, seq_j]
            M = M * dt_col

        return M
