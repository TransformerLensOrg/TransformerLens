"""GatedDeltaNet bridge for Qwen3.5/Qwen3Next linear-attention layers.

Reimplements forward (prefill only) to expose mech-interp-relevant intermediate
states. Falls back to HF native forward during autoregressive generation where
cache state management is required.
"""
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.nn.functional as F

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)

if TYPE_CHECKING:
    from transformer_lens.ActivationCache import ActivationCache


class GatedDeltaNetBridge(GeneralizedComponent):
    """Bridge for GatedDeltaNet linear-attention with full hook decomposition.

    Hooks (prefill, in execution order):
        hook_in: input hidden_states [batch, seq, d_model]
        hook_q_pre_conv: Q after projection + split, before conv [batch, seq, n_k_heads, head_k_dim]
        hook_k_pre_conv: K before conv [batch, seq, n_k_heads, head_k_dim]
        hook_v_pre_conv: V before conv [batch, seq, n_v_heads, head_v_dim]
        hook_conv_out: post-conv mixed QKV [batch, seq, key_dim*2 + value_dim]
        hook_q: Q after conv, pre-GQA-expansion [batch, seq, n_k_heads, head_k_dim]
        hook_k: K after conv [batch, seq, n_k_heads, head_k_dim]
        hook_v: V after conv [batch, seq, n_v_heads, head_v_dim]
        hook_beta: write strength (sigmoid of b), per v-head [batch, seq, n_v_heads]
        hook_log_decay: log-space decay g (negative; actual decay = exp(g)), per v-head [batch, seq, n_v_heads]
        hook_recurrence_out: output of linear recurrence kernel [batch, seq, n_v_heads, head_v_dim]
        hook_gate_input: z tensor before silu gating in GatedRMSNorm [batch, seq, n_v_heads, head_v_dim]
        hook_out: final output to residual stream [batch, seq, d_model]

    During generation (cache_params present), only hook_in/hook_out fire.

    Property aliases:
        W_in_proj_qkvz, W_in_proj_ba, W_out_proj, A_log, dt_bias
    """

    hook_aliases = {
        "hook_linear_attn_in": "hook_in",
        "hook_linear_attn_out": "hook_out",
    }

    property_aliases = {
        "W_in_proj_qkvz": "in_proj_qkvz.weight",
        "W_in_proj_ba": "in_proj_ba.weight",
        "W_out_proj": "out_proj.weight",
        "A_log": "A_log",
        "dt_bias": "dt_bias",
    }

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
        **kwargs,
    ):
        super().__init__(name, config=config, submodules=submodules or {}, **kwargs)
        # Pre-conv hooks (after projection, before causal convolution mixes positions)
        self.hook_q_pre_conv = HookPoint()
        self.hook_k_pre_conv = HookPoint()
        self.hook_v_pre_conv = HookPoint()
        # Conv output
        self.hook_conv_out = HookPoint()
        # Post-conv hooks (pre-GQA-expansion, pre-recurrence)
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        # Gate parameters (per v-head)
        self.hook_beta = HookPoint()
        self.hook_log_decay = HookPoint()
        # Recurrence output + gated norm input
        self.hook_recurrence_out = HookPoint()
        self.hook_gate_input = HookPoint()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.original_component is None:
            raise RuntimeError(f"Original component not set for {self.name}.")

        # Generation step → delegate to HF with only input/output hooks
        if kwargs.get("cache_params") is not None:
            return self._native_forward(*args, **kwargs)
        return self._hooked_forward(*args, **kwargs)

    def _native_forward(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to HF with hook_in/hook_out only (generation path)."""
        assert self.original_component is not None
        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = self.hook_in(kwargs["hidden_states"])
        elif len(args) > 0 and isinstance(args[0], torch.Tensor):
            args = (self.hook_in(args[0]),) + args[1:]

        output = self.original_component(*args, **kwargs)
        if isinstance(output, torch.Tensor):
            return self.hook_out(output)
        return output

    def _hooked_forward(self, *args: Any, **kwargs: Any) -> Any:
        """Reimplemented forward exposing all intermediate states (prefill)."""
        hf: Any = self.original_component

        if "hidden_states" in kwargs:
            hidden_states = kwargs["hidden_states"]
        elif len(args) > 0 and isinstance(args[0], torch.Tensor):
            hidden_states = args[0]
        else:
            raise ValueError("Could not find hidden_states")

        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None:
            from transformers.models.qwen3_next.modeling_qwen3_next import (
                apply_mask_to_padding_states,
            )

            hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        hidden_states = self.hook_in(hidden_states)
        batch_size, seq_len, _ = hidden_states.shape

        # --- Projections ---
        projected_qkvz = hf.in_proj_qkvz(hidden_states)
        projected_ba = hf.in_proj_ba(hidden_states)

        # Split into per-head Q, K, V, Z, beta_raw, alpha_raw
        query, key, value, z, b, a = hf.fix_query_key_value_ordering(projected_qkvz, projected_ba)

        # --- Pre-conv hooks (per-head shape, before conv mixes positions) ---
        query = self.hook_q_pre_conv(query)
        key = self.hook_k_pre_conv(key)
        value = self.hook_v_pre_conv(value)

        # Flatten for conv
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

        # --- Causal Convolution ---
        mixed_qkv = torch.cat((query, key, value), dim=-1).transpose(1, 2)
        if hf.causal_conv1d_fn is not None:
            mixed_qkv = hf.causal_conv1d_fn(
                x=mixed_qkv,
                weight=hf.conv1d.weight.squeeze(1),
                bias=hf.conv1d.bias,
                activation=hf.activation,
                seq_idx=None,
            )
        else:
            mixed_qkv = F.silu(hf.conv1d(mixed_qkv)[:, :, :seq_len])
        mixed_qkv = mixed_qkv.transpose(1, 2)

        mixed_qkv = self.hook_conv_out(mixed_qkv)

        # Split post-conv
        query, key, value = torch.split(
            mixed_qkv,
            [hf.key_dim, hf.key_dim, hf.value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, hf.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, hf.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, hf.head_v_dim)

        # --- Post-conv hooks (pre-GQA-expansion, pre-recurrence) ---
        query = self.hook_q(query)
        key = self.hook_k(key)
        value = self.hook_v(value)

        # --- Gate parameters (per v-head) ---
        beta = self.hook_beta(b.sigmoid())

        # g is log-space decay (negative); actual multiplicative decay = exp(g)
        g = -hf.A_log.float().exp() * F.softplus(a.float() + hf.dt_bias)
        g = self.hook_log_decay(g)

        # GQA expansion (Q/K from n_k_heads → n_v_heads)
        if hf.num_v_heads // hf.num_k_heads > 1:
            repeat = hf.num_v_heads // hf.num_k_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)

        # --- Core linear recurrence (opaque fused kernel) ---
        core_out, _ = hf.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        core_out = self.hook_recurrence_out(core_out)

        # --- Gated RMSNorm: norm(core_out) * silu(z) ---
        z = self.hook_gate_input(z)
        z_shape = z.shape
        core_out = hf.norm(
            core_out.reshape(-1, core_out.shape[-1]),
            z.reshape(-1, z.shape[-1]),
        )
        core_out = core_out.reshape(z_shape).reshape(batch_size, seq_len, -1)

        # --- Output projection ---
        output = hf.out_proj(core_out)
        return self.hook_out(output)

    def compute_effective_attention(
        self,
        cache: "ActivationCache",
        layer_idx: int,
    ) -> torch.Tensor:
        """Materialize the effective attention matrix from cached hook values.

        The gated delta rule recurrence is:
            S_t = exp(g_t) * S_{t-1} + beta_t * v_t @ k_t^T
            o_t = S_t^T @ q_t

        The effective attention M[i,j] = contribution of input j to output i:
            M[i,j] = (q_i^T @ k_j) * beta_j * prod_{t=j+1}^{i} exp(g_t)

        Note: the fused kernel applies L2-normalization to Q and K internally
        (use_qk_l2norm_in_kernel=True). The hooked Q/K are pre-normalization,
        so this reconstruction is approximate. For exact reconstruction, you'd
        need the normalized Q/K which aren't exposed by the kernel.

        Args:
            cache: ActivationCache from run_with_cache.
            layer_idx: Block index for this linear_attn layer.

        Returns:
            [batch, n_v_heads, seq, seq] causal attention matrix. Upper triangle
            (j > i) is zero.

        Cost is O(batch * n_heads * seq^2); use on short sequences.
        """
        prefix = f"blocks.{layer_idx}.linear_attn"
        q_key = f"{prefix}.hook_q"
        k_key = f"{prefix}.hook_k"
        beta_key = f"{prefix}.hook_beta"
        decay_key = f"{prefix}.hook_log_decay"

        for key in [q_key, k_key, beta_key, decay_key]:
            if key not in cache:
                raise RuntimeError(
                    f"compute_effective_attention needs {key!r} in cache. "
                    "Run run_with_cache() on the bridge first."
                )

        # [batch, seq, n_k_heads, head_k_dim] — pre-GQA-expansion
        q = cache[q_key].float()
        k = cache[k_key].float()
        beta = cache[beta_key].float()  # [batch, seq, n_v_heads]
        g = cache[decay_key].float()  # [batch, seq, n_v_heads]

        # GQA expansion to match n_v_heads
        if q.shape[2] < beta.shape[-1]:
            repeat = beta.shape[-1] // q.shape[2]
            q = q.repeat_interleave(repeat, dim=2)
            k = k.repeat_interleave(repeat, dim=2)

        batch, seq, n_heads, d_head = q.shape

        # QK similarity: [batch, n_heads, seq_i, seq_j]
        q_perm = q.permute(0, 2, 1, 3)  # [batch, n_heads, seq, d_head]
        k_perm = k.permute(0, 2, 1, 3)
        qk = torch.matmul(q_perm, k_perm.transpose(-2, -1))  # [batch, n_heads, seq, seq]

        # Cumulative decay: L[i,j] = prod_{t=j+1}^{i} exp(g_t) = exp(sum g[j+1..i])
        # g is [batch, seq, n_heads] → cumsum along seq
        g_perm = g.permute(0, 2, 1)  # [batch, n_heads, seq]
        cumsum_g = torch.cumsum(g_perm, dim=-1)
        # L_log[i,j] = cumsum[i] - cumsum[j]
        L_log = cumsum_g[:, :, :, None] - cumsum_g[:, :, None, :]

        causal_mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=q.device))
        L = torch.where(causal_mask[None, None], torch.exp(L_log), torch.zeros_like(L_log))

        # Beta broadcast: [batch, n_heads, 1, seq_j]
        beta_col = beta.permute(0, 2, 1)[:, :, None, :]

        # M[i,j] = qk[i,j] * beta[j] * L[i,j]
        M = qk * beta_col * L

        return M
