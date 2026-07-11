from collections.abc import Mapping

import einops
import torch

from transformer_lens.config.hooked_transformer_config import HookedTransformerConfig


def _identity(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


def _check(tensor: torch.Tensor, expected: tuple[int, ...], name: str) -> torch.Tensor:
    """Fail early and legibly when a source weight is not the shape the mapping
    assumes. A clear ValueError here beats an opaque einops/reshape error three
    frames deep, and turns a config/checkpoint mismatch into an actionable one."""
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"{name}: expected shape {expected}, got {tuple(tensor.shape)}. "
            "This usually means the HookedTransformerConfig does not match the "
            "checkpoint (wrong d_model / n_heads / d_mlp / n_experts)."
        )
    return tensor


def convert_maritime_pretrain_weights(
    old_state_dict: Mapping[str, torch.Tensor],
    cfg: HookedTransformerConfig,
) -> dict[str, torch.Tensor]:
    """Convert a checkpoint from the ``pretrain.py`` foundation-model framework
    (https://github.com/kombrellaro/maritime-intent-probe) into TransformerLens
    format, so its residual stream can be read with the standard hook API.

    The source is a decoder-only transformer with RoPE, RMSNorm, SwiGLU MLPs and
    optional dropless top-k MoE. Three conventions must line up exactly; each is
    named below so the mapping reads as a specification rather than a pile of
    index gymnastics:

    * RoPE rotates *adjacent* dimension pairs ([x0, x1] -> [-x1, x0]), so the
      matching config sets ``rotary_adjacent_pairs=True`` and applies rotation
      inside attention. Q/K/V therefore need no reordering here.
    * Attention fuses Q/K/V into one column-parallel projection stored as
      ``blocks.{l}.attn.qkv.weight`` of shape [3*d_model, d_model]; it is split
      into three per-head projections via :func:`split_heads`.
    * SwiGLU maps onto gated-MLP naming as gate -> W_gate, up -> W_in,
      down -> W_out. The MoE experts store these as ``nn.Linear`` ([out, in],
      kept as-is under a ``.weight`` key); the dense gated MLP stores raw
      [in, out] parameters (transposed, no suffix). Same mapping, two storage
      conventions -- and that single fact is the whole dense/MoE asymmetry,
      captured once in :func:`emit_gated_mlp`.

    Raises:
        ValueError: if a source weight is not the shape ``cfg`` implies, so a
            config/checkpoint mismatch fails with a clear message rather than an
            opaque reshape error.
    """
    n_heads, d_head, d_model = cfg.n_heads, cfg.d_head, cfg.d_model
    d_mlp = cfg.d_mlp

    def split_heads(w: torch.Tensor) -> torch.Tensor:
        # [feat, d_model] input projection -> [n_heads, d_model, d_head]
        return einops.rearrange(w, "(h dh) m -> h m dh", h=n_heads, dh=d_head)

    def merge_heads(w: torch.Tensor) -> torch.Tensor:
        # [d_model, feat] output projection -> [n_heads, d_head, d_model]
        return einops.rearrange(w, "m (h dh) -> h dh m", h=n_heads, dh=d_head)

    def zeros(*shape: int) -> torch.Tensor:
        return torch.zeros(*shape, dtype=cfg.dtype)

    def src(block: int, key: str) -> torch.Tensor:
        # a block's source weight, addressed by suffix
        return old_state_dict[f"blocks.{block}.{key}"]

    def emit_gated_mlp(
        prefix: str,
        gate: torch.Tensor,
        up: torch.Tensor,
        down: torch.Tensor,
        *,
        as_linear: bool,
    ) -> dict[str, torch.Tensor]:
        """gate/up/down -> W_gate/W_in/W_out under one of two storage styles:
        nn.Linear weights ([out, in], ``.weight`` suffix) are kept as-is; raw
        parameters ([in, out], no suffix) are transposed."""
        _check(gate, (d_mlp, d_model), f"{prefix}.W_gate")
        _check(up, (d_mlp, d_model), f"{prefix}.W_in")
        _check(down, (d_model, d_mlp), f"{prefix}.W_out")
        suffix = ".weight" if as_linear else ""
        orient = _identity if as_linear else torch.t
        return {
            f"{prefix}.W_gate{suffix}": orient(gate),
            f"{prefix}.W_in{suffix}": orient(up),
            f"{prefix}.W_out{suffix}": orient(down),
        }

    tied_unembed = "lm_head.weight" not in old_state_dict
    new_state_dict: dict[str, torch.Tensor] = {
        "embed.W_E": old_state_dict["embed.weight"],
        "ln_final.w": old_state_dict["norm_f.weight"],
        "unembed.W_U": old_state_dict["embed.weight" if tied_unembed else "lm_head.weight"].T,
        "unembed.b_U": zeros(cfg.d_vocab),
    }

    for layer in range(cfg.n_layers):
        p = f"blocks.{layer}"

        new_state_dict[f"{p}.ln1.w"] = src(layer, "norm1.weight")
        new_state_dict[f"{p}.ln2.w"] = src(layer, "norm2.weight")

        # Attention: one fused QKV projection -> three per-head Q/K/V, plus out.
        qkv = _check(src(layer, "attn.qkv.weight"), (3 * d_model, d_model), f"{p}.attn.qkv")
        for name, w in zip("QKV", qkv.chunk(3, dim=0)):
            new_state_dict[f"{p}.attn.W_{name}"] = split_heads(w)
            new_state_dict[f"{p}.attn.b_{name}"] = zeros(n_heads, d_head)
        w_o = _check(src(layer, "attn.proj.weight"), (d_model, d_model), f"{p}.attn.proj")
        new_state_dict[f"{p}.attn.W_O"] = merge_heads(w_o)
        new_state_dict[f"{p}.attn.b_O"] = zeros(d_model)

        # MLP: a dense SwiGLU, or a router plus a fleet of expert SwiGLUs.
        # Both conditions are load-bearing: num_experts guards the *config* side
        # (a dense config leaves it None and must take the dense branch even if a
        # stray router key existed), and the key check guards the *checkpoint*
        # side (an MoE config whose every-k-th-layer placement makes this
        # particular block dense -- moe_every > 1 -- has no router here).
        is_moe = cfg.num_experts is not None and f"{p}.mlp.router.weight" in old_state_dict
        if is_moe:
            router = _check(
                src(layer, "mlp.router.weight"), (cfg.num_experts, d_model), f"{p}.mlp.router"
            )
            new_state_dict[f"{p}.mlp.W_gate.weight"] = router
            for e in range(cfg.num_experts):
                ep = f"mlp.experts.{e}"
                new_state_dict.update(
                    emit_gated_mlp(
                        f"{p}.{ep}",
                        src(layer, f"{ep}.gate.weight"),
                        src(layer, f"{ep}.up.weight"),
                        src(layer, f"{ep}.down.weight"),
                        as_linear=True,
                    )
                )
        else:
            new_state_dict.update(
                emit_gated_mlp(
                    f"{p}.mlp",
                    src(layer, "mlp.gate.weight"),
                    src(layer, "mlp.up.weight"),
                    src(layer, "mlp.down.weight"),
                    as_linear=False,
                )
            )
            new_state_dict[f"{p}.mlp.b_in"] = zeros(d_mlp)
            new_state_dict[f"{p}.mlp.b_out"] = zeros(d_model)

    return new_state_dict
