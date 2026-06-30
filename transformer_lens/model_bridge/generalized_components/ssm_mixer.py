"""Wrap-don't-reimplement bridge for HF's MambaMixer (Mamba-1), plus S6 effective attention."""
from typing import Any

import torch

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class SSMMixerBridge(GeneralizedComponent):
    """Opaque wrapper around Mamba-1's MambaMixer.

    Submodules (in_proj, conv1d, x_proj, dt_proj, out_proj) are swapped into
    the HF mixer by ``replace_remote_component``, so their hooks fire when
    slow_forward accesses them. ``A_log`` and ``D`` reach the user via
    ``GeneralizedComponent.__getattr__`` delegation.

    Decode-step caveat: ``conv1d.hook_out`` fires only on prefill during
    stateful generation; see ``DepthwiseConv1DBridge`` for the reason.
    """

    hook_aliases = {
        "hook_in_proj": "in_proj.hook_out",
        "hook_conv": "conv1d.hook_out",
        "hook_x_proj": "x_proj.hook_out",
        "hook_dt_proj": "dt_proj.hook_out",
        "hook_ssm_out": "hook_out",
    }

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Hook the input, delegate to HF slow_forward, hook the output."""
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. "
                "Call set_original_component() first."
            )

        # Hook the hidden_states input (positional or keyword)
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            hooked = self.hook_in(args[0])
            args = (hooked,) + args[1:]
        elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
            kwargs["hidden_states"] = self.hook_in(kwargs["hidden_states"])

        output = self.original_component(*args, **kwargs)

        # Hook the primary output tensor, preserving tuple structure
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
        per_state_coord: bool = False,
    ) -> torch.Tensor:
        """Materialize Mamba-1's per-channel effective attention from cached hooks.

        Mamba-1's S6 selective scan is equivalent to causal attention with a
        per-channel, per-state-coordinate learned decay — see "The Hidden
        Attention of Mamba" (Ali et al., ACL 2025). Unlike Mamba-2 there is no
        head grouping: each of the ``intermediate_size`` channels has its own
        ``A`` row, so the "head" axis here is the channel axis::

            M[c, i, j] = sum_n C[i, n] · prod_{k=j+1..i} exp(A[c,n]·dt[c,k]) · B[j, n]

        Reads B/C from ``x_proj.hook_out`` and dt from ``dt_proj.hook_out``
        (dt = softplus of that output); A from the module via ``__getattr__``.
        Read-only: no ``forward()`` re-run.

        Args:
            cache: ActivationCache from ``run_with_cache`` with this layer's
                x_proj and dt_proj hooks.
            layer_idx: Block index for this mixer.
            include_dt_scaling: False (default) returns the attention-like form;
                True multiplies column j by dt[c, j], giving the reconstruction
                form satisfying ``y[c,i] = sum_j M[c,i,j]·x[c,j] + D[c]·x[c,i]``
                (x is the post-conv SiLU input; y the pre-gate scan output).
            per_state_coord: False (default) sums over the state coordinate and
                returns ``[batch, channels, seq, seq]``. True returns the
                unsummed ``[batch, channels, state, seq, seq]`` tensor (the
                paper's D·N matrices) — OFF by default.

        Returns:
            ``[batch, intermediate_size, seq, seq]``, or
            ``[batch, intermediate_size, state_size, seq, seq]`` when
            ``per_state_coord`` is True.

        Peak memory is O(batch · intermediate_size · state_size · seq²) — the
        per-(channel, state) decay tensor — even for the summed default; use on
        short sequences.
        """
        if self.config is None:
            raise RuntimeError("SSMMixerBridge.config must be set")

        x_proj_key = f"blocks.{layer_idx}.mixer.x_proj.hook_out"
        dt_proj_key = f"blocks.{layer_idx}.mixer.dt_proj.hook_out"
        for key in (x_proj_key, dt_proj_key):
            if key not in cache:
                raise RuntimeError(
                    f"compute_effective_attention needs {key!r} in cache. Run "
                    "`run_with_cache()` on the bridge before calling this method."
                )

        cfg = self.config
        oc = self.original_component

        def _mamba_dim(module_attr: str, cfg_attr: str, default: Any) -> Any:
            if oc is not None:
                val = getattr(oc, module_attr, None)
                if val is not None:
                    return val
            return getattr(cfg, cfg_attr, default)

        state_size = int(_mamba_dim("ssm_state_size", "state_size", 16))
        dt_rank = int(_mamba_dim("time_step_rank", "time_step_rank", 0))

        x_proj_out = cache[x_proj_key].float()  # [batch, seq, dt_rank + 2*state]
        dt_proj_out = cache[dt_proj_key].float()  # [batch, seq, intermediate_size]
        seq_len = dt_proj_out.shape[1]

        # B, C from x_proj output (shared across channels); first dt_rank cols are dt low-rank.
        _time_step, B, C = x_proj_out.split([dt_rank, state_size, state_size], dim=-1)

        # Per-channel dt = softplus(dt_proj output), channel-first.
        dt = torch.nn.functional.softplus(dt_proj_out).transpose(1, 2)  # [batch, channels, seq]
        A = -torch.exp(self.A_log.float())  # [channels, state]

        # Per-(channel, state) causal decay:
        # decay[c, n, i, j] = exp(A[c,n] · (cumdt[c,i] - cumdt[c,j])) for i >= j, else 0.
        cumdt = torch.cumsum(dt, dim=-1)  # [batch, channels, seq]
        dcs = cumdt[:, :, :, None] - cumdt[:, :, None, :]  # [batch, channels, i, j]
        decay_exp = (
            A[None, :, :, None, None] * dcs[:, :, None, :, :]
        )  # [batch, channels, state, i, j]
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=decay_exp.device)
        )
        decay = torch.where(
            causal_mask[None, None, None, :, :],
            torch.exp(decay_exp),
            torch.zeros_like(decay_exp),
        )

        # M_coord[c, n, i, j] = C[i, n] · decay · B[j, n]
        C_col = C.permute(0, 2, 1)[:, None, :, :, None]  # [batch, 1, state, i, 1]
        B_col = B.permute(0, 2, 1)[:, None, :, None, :]  # [batch, 1, state, 1, j]
        M_coord = C_col * decay * B_col  # [batch, channels, state, i, j]

        if include_dt_scaling:
            M_coord = M_coord * dt[:, :, None, None, :]  # × dt[c, j]

        if per_state_coord:
            return M_coord
        return M_coord.sum(dim=2)  # [batch, channels, seq, seq]
