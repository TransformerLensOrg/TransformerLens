"""Wrap-don't-reimplement bridge for HF's MambaMixer (Mamba-1), plus S6 effective attention."""
from typing import Any, NamedTuple, Optional

import torch

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.ssm_protocol import (
    SSMStateHookMixin,
)


class _S6Terms(NamedTuple):
    """S6 intermediates reconstructed from cached Mamba-1 hooks (per-channel)."""

    dt: torch.Tensor  # [batch, channels, seq]
    decay: torch.Tensor  # [batch, channels, state, i, j] causal (upper triangle zero)
    B: torch.Tensor  # [batch, seq, state] (shared across channels)
    C: torch.Tensor  # [batch, seq, state] (shared across channels)


class SSMMixerBridge(SSMStateHookMixin, GeneralizedComponent):
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
        # Canonical SSM vocabulary (additive): Mamba-1 exposes the discrete time
        # step via dt_proj. B/C are bundled in x_proj (not separately hooked).
        "hook_ssm_dt": "dt_proj.hook_out",
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)  # mixin adds hook_ssm_state + eager_scan
        # Real per-step write term dt·x·B, per-channel [batch, channels, seq, state].
        self.hook_ssm_write = HookPoint()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Hook the input, run HF slow_forward (or the eager scan), hook the output."""
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. "
                "Call set_original_component() first."
            )

        hidden_states: Optional[torch.Tensor] = None
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            hidden_states = self.hook_in(args[0])
            args = (hidden_states,) + args[1:]
        elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
            hidden_states = self.hook_in(kwargs["hidden_states"])
            kwargs["hidden_states"] = hidden_states

        # Eager-scan slow path: opt-in, prefill only (no cache_params). Keyed on
        # explicit state, never on hook-registry introspection.
        if self.eager_scan and hidden_states is not None and kwargs.get("cache_params") is None:
            output: Any = self._eager_scan_forward(hidden_states, kwargs.get("attention_mask"))
        else:
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

    def _eager_scan_forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Reimplement the Mamba-1 mixer prefill forward with an eager Python S6 scan.

        Reuses HF's in_proj / conv1d / x_proj / dt_proj / out_proj submodules (so
        their hooks still fire) and reimplements ONLY the S6 recurrence::

            write_t[c, s] = dt_t[c] · x_t[c] · B_t[s]        -> hook_ssm_write [b, c, seq, s]
            S_t[c, s]     = exp(A[c,s]·dt_t[c]) · S_{t-1} + write_t
            S             = stack_t S_t                       -> hook_ssm_state [b, c, seq, s]
            y_t[c]        = sum_s C_t[s] · S_t[c, s] + D[c] · x_t[c]

        Intervening on hook_ssm_write re-runs the recurrence (propagates to later
        states); hook_ssm_state is the post-scan trajectory, so a patch there
        changes only the same-position output y_t = C_t·S_t — patch hook_ssm_write
        for a propagating state edit.

        When the wrapped mixer exposes ``dt_layernorm`` / ``b_layernorm`` /
        ``c_layernorm`` (Jamba), those norms run after ``x_proj`` so the scan
        matches HF's selective-param path.

        Kernel-divergence caveat: reproduces HF's scan only to fp tolerance
        (≈1e-6 fp32), never bit-for-bit. O(seq) Python; materializes an
        O(b·channels·seq·state) write/state tensor. Prefill only.
        """
        oc: Any = self.original_component
        batch, seq_len, _ = hidden_states.shape
        d_inner = oc.intermediate_size
        state = oc.ssm_state_size
        dt_rank = oc.time_step_rank

        def _mask_cf(states: torch.Tensor) -> torch.Tensor:
            # HF MambaMixer masks whenever attention_mask is present (no batch guard,
            # unlike Mamba-2), so match that — batch-1 padded inputs included.
            if attention_mask is not None:
                return states * attention_mask.unsqueeze(1).to(states.dtype)
            return states

        # 1-2. in_proj + gate split, conv — reuse the HF submodule bridges (hooks fire).
        # HF masks padding on the channel-first x both before and after the conv.
        projected = oc.in_proj(hidden_states).transpose(1, 2)  # [b, 2*d_inner, seq]
        x_raw, gate = projected.chunk(2, dim=1)  # [b, d_inner, seq]
        x = oc.act(oc.conv1d(_mask_cf(x_raw))[..., :seq_len])  # [b, d_inner, seq]
        x = _mask_cf(x)

        # 3. x_proj -> dt low-rank / B / C ; dt_proj -> dt (softplus, channel-first).
        # Jamba (and forks) apply RMSNorm to dt/B/C after x_proj; stock Mamba-1 does not.
        ssm_params = oc.x_proj(x.transpose(1, 2))  # [b, seq, dt_rank + 2*state]
        time_step, B, C = ssm_params.split([dt_rank, state, state], dim=-1)  # B, C: [b, seq, state]
        time_step, B, C = self._apply_selective_param_norms(oc, time_step, B, C)
        dt = torch.nn.functional.softplus(oc.dt_proj(time_step)).transpose(1, 2).float()
        A = -torch.exp(self.A_log.float())  # [d_inner, state]
        x_f, B_f, C_f = x.float(), B.float(), C.float()

        # 4. Eager recurrence with intervention hooks.
        writes = (dt * x_f)[:, :, :, None] * B_f[:, None, :, :]  # [b, d_inner, seq, state]
        writes = self.hook_ssm_write(writes)

        ssm_state = torch.zeros(batch, d_inner, state, dtype=writes.dtype, device=writes.device)
        states = []
        for t in range(seq_len):
            decay = torch.exp(A[None] * dt[:, :, t, None])  # [batch, d_inner, state]
            ssm_state = decay * ssm_state + writes[:, :, t]
            states.append(ssm_state)
        state_traj = torch.stack(states, dim=2)  # [batch, d_inner, seq, state]
        state_traj = self.hook_ssm_state(state_traj)

        y = torch.einsum("bcts,bts->bct", state_traj, C_f)  # [batch, d_inner, seq]
        y = y + self.D.float()[None, :, None] * x_f
        scan_output = y * oc.act(gate.float())  # HF gates with self.act, not hardcoded silu

        # 5. Output projection — reuse the HF submodule bridge.
        contextualized: torch.Tensor = oc.out_proj(
            scan_output.transpose(1, 2).to(hidden_states.dtype)
        )
        return contextualized

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

        Reads B/C from ``x_proj.hook_out`` (or post-norm
        ``b_layernorm`` / ``c_layernorm`` hooks when present, as on Jamba) and
        dt from ``dt_proj.hook_out`` (softplus of that output); A via
        ``__getattr__``. Read-only: no ``forward()`` re-run.

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
        t = self._s6_terms(cache, layer_idx)

        # M_coord[c, s, i, j] = C[i, s] · decay[c, s, i, j] · B[j, s]
        C_col = t.C.permute(0, 2, 1)[:, None, :, :, None]  # [batch, 1, state, i, 1]
        B_col = t.B.permute(0, 2, 1)[:, None, :, None, :]  # [batch, 1, state, 1, j]
        M_coord = C_col * t.decay * B_col  # [batch, channels, state, i, j]

        if include_dt_scaling:
            M_coord = M_coord * t.dt[:, :, None, None, :]  # × dt[c, j]

        if per_state_coord:
            return M_coord
        return M_coord.sum(dim=2)  # [batch, channels, seq, seq]

    def compute_ssm_state(
        self,
        cache: ActivationCache,
        layer_idx: int,
        time_step: Optional[int] = None,
    ) -> torch.Tensor:
        """Reconstruct Mamba-1's recurrent state ``S`` from cached hook values.

        S6 recurrence ``h_t[c,s] = exp(A[c,s]·dt_t[c])·h_{t-1}[c,s] + dt_t[c]·x_t[c]·B_t[s]``
        unrolls to::

            S_t[c, s] = sum_{j<=t} decay[c, s, t, j] · dt_j[c] · x_j[c] · B_j[s]

        with the same per-(channel, state) ``decay`` used by
        ``compute_effective_attention``. ``x`` is the post-conv SiLU input
        (``SiLU(conv1d.hook_out)``). Read-only: no ``forward()`` re-run. Verify
        with ``y_t[c] = sum_s C_t[s]·S_t[c,s] + D[c]·x_t[c]``.

        On padded batches the cached hooks are unmasked, so ``S`` is exact only at
        non-pad positions; pad-position state is out of contract.

        Args:
            cache: ActivationCache from ``run_with_cache`` with this layer's
                x_proj, dt_proj, and conv1d hooks.
            layer_idx: Block index for this mixer.
            time_step: If given, return only ``S`` at that position
                (``[batch, channels, state]``); None returns every step.

        Returns:
            ``[batch, channels, seq, state]`` for all steps, or
            ``[batch, channels, state]`` for a single ``time_step``.

        Peak memory is O(batch · channels · state · seq²) — the per-(channel,
        state) decay tensor, built in full by ``_s6_terms`` regardless of
        ``time_step`` (which bounds only the returned tensor, not this peak); use
        on short sequences.
        """
        conv_key = f"blocks.{layer_idx}.mixer.conv1d.hook_out"
        if conv_key not in cache:
            raise RuntimeError(
                f"compute_ssm_state needs {conv_key!r} in cache. Run "
                "`run_with_cache()` on the bridge before calling this method."
            )
        t = self._s6_terms(cache, layer_idx)
        seq_len = t.dt.shape[-1]
        oc: Any = self.original_component
        # x = act(conv output), the SSM scan input; channel-first [batch, channels, seq].
        x = oc.act(cache[conv_key].float()[..., :seq_len])
        dtx = t.dt * x  # dt_j[c]·x_j[c], [batch, channels, seq]
        # S_t[c, s] = sum_{j<=t} decay[c, s, t, j] · dtx[c, j] · B_j[s]
        if time_step is not None:
            return torch.einsum("bcsj,bcj,bjs->bcs", t.decay[:, :, :, time_step, :], dtx, t.B)
        return torch.einsum("bcsij,bcj,bjs->bcis", t.decay, dtx, t.B)

    @staticmethod
    def _apply_selective_param_norms(
        oc: Any,
        time_step: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply Jamba-style RMSNorms on dt/B/C when the HF mixer exposes them.

        Stock Mamba-1 has no ``dt_layernorm`` / ``b_layernorm`` / ``c_layernorm``;
        Jamba's ``JambaMambaMixer`` does. Calling through the (possibly bridged)
        attributes keeps their hooks firing on the eager-scan path.
        """
        dt_ln = getattr(oc, "dt_layernorm", None)
        if dt_ln is not None:
            time_step = dt_ln(time_step)
        b_ln = getattr(oc, "b_layernorm", None)
        if b_ln is not None:
            B = b_ln(B)
        c_ln = getattr(oc, "c_layernorm", None)
        if c_ln is not None:
            C = c_ln(C)
        return time_step, B, C

    def _s6_terms(self, cache: ActivationCache, layer_idx: int) -> _S6Terms:
        """Reconstruct the shared S6 intermediates (dt, per-(channel,state) decay, B, C).

        Reused by ``compute_effective_attention`` and ``compute_ssm_state``. Reads
        B/C from ``x_proj.hook_out`` (shared across channels), or — when present —
        from Jamba's post-norm ``b_layernorm`` / ``c_layernorm`` hooks. ``dt`` comes
        from ``dt_proj.hook_out`` (softplus); that already sits after Jamba's
        ``dt_layernorm``. A via ``__getattr__``. Dims come from the wrapped HF
        mixer; cfg is the fallback (mirrors the Mamba-2 bridge).
        """
        if self.config is None:
            raise RuntimeError("SSMMixerBridge.config must be set")

        x_proj_key = f"blocks.{layer_idx}.mixer.x_proj.hook_out"
        dt_proj_key = f"blocks.{layer_idx}.mixer.dt_proj.hook_out"
        for key in (x_proj_key, dt_proj_key):
            if key not in cache:
                raise RuntimeError(
                    f"S6 reconstruction needs {key!r} in cache. Run "
                    "`run_with_cache()` on the bridge before calling this."
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
        dt_proj_out = cache[dt_proj_key].float()  # [batch, seq, channels]
        seq_len = dt_proj_out.shape[1]

        # Prefer post-norm B/C when the adapter mapped Jamba's selective-param LNs.
        b_ln_key = f"blocks.{layer_idx}.mixer.b_layernorm.hook_out"
        c_ln_key = f"blocks.{layer_idx}.mixer.c_layernorm.hook_out"
        if b_ln_key in cache and c_ln_key in cache:
            B = cache[b_ln_key].float()
            C = cache[c_ln_key].float()
        else:
            # Stock Mamba-1: B, C are the raw x_proj tails (shared across channels).
            _time_step, B, C = x_proj_out.split([dt_rank, state_size, state_size], dim=-1)

        dt = torch.nn.functional.softplus(dt_proj_out).transpose(1, 2)  # [batch, channels, seq]
        A = -torch.exp(self.A_log.float())  # [channels, state]

        # decay[c, s, i, j] = exp(A[c,s]·(cumdt[c,i]-cumdt[c,j])) for i >= j, else 0.
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
        return _S6Terms(dt=dt, decay=decay, B=B, C=C)
