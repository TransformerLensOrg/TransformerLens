"""Wrap-don't-reimplement bridge for HF's Mamba2Mixer, plus SSD effective attention."""
from typing import Any, NamedTuple, Optional

import torch

from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class _SSDTerms(NamedTuple):
    """SSD intermediates reconstructed from cached Mamba-2 hooks (HF discretization)."""

    dt: torch.Tensor  # [batch, seq, num_heads]
    L: torch.Tensor  # [batch, num_heads, seq, seq] causal decay (upper triangle zero)
    x: torch.Tensor  # [batch, seq, num_heads, head_dim] SSM input per head
    B: torch.Tensor  # [batch, seq, num_heads, state] (group-expanded)
    C: torch.Tensor  # [batch, seq, num_heads, state] (group-expanded)


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
        # Canonical SSM vocabulary (additive). Mamba-2 fuses B/C/dt into the
        # in_proj/conv1d outputs, so only the mixer output has a granular hook by
        # default; B/C/dt/decay are reconstructed via compute_effective_attention /
        # compute_ssm_state. hook_ssm_write / hook_ssm_state are real HookPoints
        # that fire only on the opt-in eager-scan intervention path.
        "hook_ssm_out": "hook_out",
    }

    # Opt-in slow path: when True (and not generating), forward() runs an eager
    # Python scan instead of HF's fused kernel, firing hook_ssm_write /
    # hook_ssm_state so they can be intervened on. Default False — the standard
    # run_with_cache path is untouched. See _eager_scan_forward.
    eager_scan: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Per-step write term (dt·x⊗B) and recurrent-state trajectory. Only fire
        # on the eager-scan path; intervening on them re-runs the recurrence.
        self.hook_ssm_write = HookPoint()
        self.hook_ssm_state = HookPoint()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Hook the input, run HF torch_forward (or the eager scan), hook the output."""
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

        # Eager-scan slow path: opt-in flag, prefill only (no cache_params). Keyed
        # on explicit state, never on hook-registry introspection.
        if self.eager_scan and hidden_states is not None and kwargs.get("cache_params") is None:
            output: Any = self._eager_scan_forward(hidden_states, kwargs.get("attention_mask"))
        else:
            output = self.original_component(*args, **kwargs)

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
        """Reimplement the Mamba-2 mixer prefill forward with an eager Python scan.

        Reuses HF's in_proj / conv1d / inner-norm / out_proj submodules (so their
        hooks still fire) and reimplements ONLY the SSD recurrence::

            write_t = dt_t · (x_t ⊗ B_t)            -> hook_ssm_write [b, seq, h, d, n]
            S_t     = exp(dt_t·A) · S_{t-1} + write_t
            S       = stack_t S_t                    -> hook_ssm_state [b, seq, h, d, n]
            y_t     = C_t · S_t                       (recomputed from the hooked state)

        Intervening on hook_ssm_write re-runs the recurrence (the change propagates
        to every later state); hook_ssm_state is the post-scan trajectory, so a
        patch there changes only the same-position output y_t = C_t·S_t, not the
        forward recurrence — patch hook_ssm_write for a propagating state edit.

        Kernel-divergence caveat: this eager scan reproduces HF's fused chunk scan
        only to floating-point tolerance (≈1e-6 fp32), never bit-for-bit, on any
        family. It is O(seq) Python and materializes an O(b·seq·h·d·n) write/state
        tensor — orders of magnitude slower/heavier than the kernel. Prefill only.
        """
        oc: Any = self.original_component
        batch, seq_len, _ = hidden_states.shape
        intermediate, num_heads, head_dim = oc.intermediate_size, oc.num_heads, oc.head_dim
        n_groups, state = oc.n_groups, oc.ssm_state_size

        def _mask_pad(states: torch.Tensor) -> torch.Tensor:
            # Mirror HF apply_mask_to_padding_states (no-op for batch 1 / unpadded).
            if attention_mask is not None and attention_mask.shape[1] > 1 and batch > 1:
                return (states * attention_mask[:, :, None]).to(states.dtype)
            return states

        # 1-2. Projection + conv — reuse the HF submodule bridges (their hooks fire).
        # HF masks padding both before in_proj and after the conv (before B/C split).
        projected = oc.in_proj(_mask_pad(hidden_states))
        d_mlp = (projected.shape[-1] - 2 * intermediate - 2 * n_groups * state - num_heads) // 2
        _, _, gate, hidden_B_C, dt = projected.split(
            [d_mlp, d_mlp, intermediate, oc.conv_dim, num_heads], dim=-1
        )
        hidden_B_C = oc.act(oc.conv1d(hidden_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2))
        hidden_B_C = _mask_pad(hidden_B_C)
        x_flat, B_flat, C_flat = hidden_B_C.split(
            [intermediate, n_groups * state, n_groups * state], dim=-1
        )

        x = x_flat.reshape(batch, seq_len, num_heads, head_dim).float()
        heads_per_group = num_heads // n_groups
        B = (
            B_flat.reshape(batch, seq_len, n_groups, state)
            .repeat_interleave(heads_per_group, dim=2)
            .float()
        )
        C = (
            C_flat.reshape(batch, seq_len, n_groups, state)
            .repeat_interleave(heads_per_group, dim=2)
            .float()
        )

        dt = torch.nn.functional.softplus(dt.float() + self.dt_bias.float())
        dt = torch.clamp(dt, float(oc.time_step_limit[0]), float(oc.time_step_limit[1]))
        A = -torch.exp(self.A_log.float())  # [num_heads]

        # 3. Eager recurrence with intervention hooks.
        # write_t[b, t, h, d, n] = dt_t · x_t[d] · B_t[n]
        writes = dt[:, :, :, None, None] * x[:, :, :, :, None] * B[:, :, :, None, :]
        writes = self.hook_ssm_write(writes)  # [batch, seq, heads, head_dim, state]

        ssm_state = torch.zeros(batch, num_heads, head_dim, state, dtype=torch.float32)
        states = []
        for t in range(seq_len):
            decay = torch.exp(dt[:, t, :] * A[None, :])  # [batch, heads]
            ssm_state = decay[:, :, None, None] * ssm_state + writes[:, t]
            states.append(ssm_state)
        state_traj = torch.stack(states, dim=1)  # [batch, seq, heads, head_dim, state]
        state_traj = self.hook_ssm_state(state_traj)

        y = torch.einsum("bthn,bthdn->bthd", C, state_traj)
        y = y + self.D.float()[None, None, :, None] * x
        y = y.reshape(batch, seq_len, intermediate)

        # 4. Gated norm + output projection — reuse the HF submodule bridges.
        scan_output = oc.norm(y.to(hidden_states.dtype), gate)
        contextualized: torch.Tensor = oc.out_proj(scan_output)
        return contextualized

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
        terms = self._ssd_terms(cache, layer_idx)

        # CB[b, h, i, j] = <C[b, i, h], B[b, j, h]>
        CB = torch.einsum("bihs,bjhs->bhij", terms.C, terms.B)
        M = terms.L * CB  # [batch, num_heads, seq, seq]

        if include_dt_scaling:
            # Multiply column j by dt[j, h] to absorb the B discretization
            M = M * terms.dt.permute(0, 2, 1)[:, :, None, :]

        return M

    def compute_ssm_state(
        self,
        cache: ActivationCache,
        layer_idx: int,
        time_step: Optional[int] = None,
    ) -> torch.Tensor:
        """Reconstruct the recurrent SSM state ``S`` from cached hook values.

        Mamba-2's recurrence is ``S_t = dA_t · S_{t-1} + dt_t · (x_t ⊗ B_t)`` with
        ``dA_t[h] = exp(dt_t[h] · A[h])``, so post-hoc::

            S_t[h, p, n] = sum_{j<=t} L[t, j] · dt_j[h] · x_j[h, p] · B_j[h, n]

        where ``L`` is the same causal decay matrix used by
        ``compute_effective_attention``. Read-only: no ``forward()`` re-run.
        Verify with ``y_t = C_t · S_t + D · x_t == inner_norm.hook_in``.

        Args:
            cache: ActivationCache from ``run_with_cache`` with this layer's
                in_proj and conv1d hooks.
            layer_idx: Block index for this mixer.
            time_step: If given, return only ``S`` at that position
                (``[batch, num_heads, head_dim, state]``); avoids materializing
                the full tensor. None returns every step.

        Returns:
            ``[batch, num_heads, seq, head_dim, state]`` for all steps, or
            ``[batch, num_heads, head_dim, state]`` for a single ``time_step``.

        Memory is O(batch · num_heads · seq · head_dim · state) for the full
        tensor; pass ``time_step`` (or short sequences) when that is too large.
        """
        terms = self._ssd_terms(cache, layer_idx)
        dtx = terms.dt[..., None] * terms.x  # [batch, seq, num_heads, head_dim]
        if time_step is not None:
            # S_t = sum_{j<=t} L[t, j] · (dt_j x_j) ⊗ B_j
            return torch.einsum("bhj,bjhp,bjhn->bhpn", terms.L[:, :, time_step, :], dtx, terms.B)
        return torch.einsum("bhtj,bjhp,bjhn->bhtpn", terms.L, dtx, terms.B)

    def _ssd_terms(self, cache: ActivationCache, layer_idx: int) -> _SSDTerms:
        """Reconstruct the shared SSD intermediates (dt, decay L, per-head x/B/C).

        Mirrors HF Mamba2Mixer's discretization from the cached in_proj/conv1d
        outputs. Dims come from the wrapped HF mixer, not cfg: on a hybrid the
        shared cfg holds the *attention* dims (cfg.n_heads etc.), while the HF
        mixer always carries the true Mamba dims (as do A_log/dt_bias, already
        read off the module via __getattr__). cfg is the fallback.
        """
        if self.config is None:
            raise RuntimeError("SSM2MixerBridge.config must be set")

        in_proj_key = f"blocks.{layer_idx}.mixer.in_proj.hook_out"
        conv1d_key = f"blocks.{layer_idx}.mixer.conv1d.hook_out"
        if in_proj_key not in cache or conv1d_key not in cache:
            raise RuntimeError(
                f"SSD reconstruction needs {in_proj_key!r} and {conv1d_key!r} in "
                "cache. Run `run_with_cache()` on the bridge before calling this."
            )

        cfg = self.config
        oc = self.original_component

        def _mamba_dim(module_attr: str, cfg_attr: str, default: Any) -> Any:
            if oc is not None:
                val = getattr(oc, module_attr, None)
                if val is not None:
                    return val
            return getattr(cfg, cfg_attr, default)

        num_heads = int(_mamba_dim("num_heads", "n_heads", 0))
        head_dim = int(_mamba_dim("head_dim", "d_head", 0))
        intermediate_size = int(
            _mamba_dim("intermediate_size", "intermediate_size", num_heads * head_dim)
        )
        state_size = int(_mamba_dim("ssm_state_size", "state_size", 128))
        n_groups = int(_mamba_dim("n_groups", "n_groups", 1))
        time_step_limit = _mamba_dim("time_step_limit", "time_step_limit", (0.0, float("inf")))

        in_proj_out = cache[in_proj_key].float()  # [batch, seq, proj_size]
        conv1d_out = cache[conv1d_key].float()  # [batch, conv_dim, seq + kernel - 1]
        batch_size, seq_len = in_proj_out.shape[0], in_proj_out.shape[1]

        # dt: last num_heads features of in_proj, post softplus + clamp
        dt = torch.nn.functional.softplus(in_proj_out[..., -num_heads:] + self.dt_bias.float())
        dt = torch.clamp(dt, float(time_step_limit[0]), float(time_step_limit[1]))

        # x, B, C from conv1d output (trim to seq, SiLU, channel-last split)
        conv_activated = torch.nn.functional.silu(conv1d_out[..., :seq_len]).transpose(1, 2)
        x_flat, B_flat, C_flat = conv_activated.split(
            [intermediate_size, n_groups * state_size, n_groups * state_size], dim=-1
        )
        x = x_flat.view(batch_size, seq_len, num_heads, head_dim)
        # GQA-style: each of n_groups B/C pairs covers n_heads // n_groups heads
        heads_per_group = num_heads // n_groups
        B = B_flat.view(batch_size, seq_len, n_groups, state_size).repeat_interleave(
            heads_per_group, dim=2
        )
        C = C_flat.view(batch_size, seq_len, n_groups, state_size).repeat_interleave(
            heads_per_group, dim=2
        )

        # L[i, j] = exp(sum_{k=j+1}^{i} A[h] · dt[k, h]) for i >= j, else 0.
        # exp(cumsum[i] - cumsum[j]); cumsum[j] includes dt[j], so the sum is k>j.
        A = -torch.exp(self.A_log.float())  # [num_heads]
        cs = torch.cumsum(dt * A[None, None, :], dim=1).permute(0, 2, 1)  # [batch, num_heads, seq]
        L_log = cs[:, :, :, None] - cs[:, :, None, :]
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=L_log.device)
        )
        L = torch.where(causal_mask[None, None], torch.exp(L_log), torch.zeros_like(L_log))

        return _SSDTerms(dt=dt, L=L, x=x, B=B, C=C)
