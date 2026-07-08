"""Shared helpers for per-adapter unit tests."""

from transformer_lens.config import TransformerBridgeConfig


def make_bridge_cfg(architecture: str, **overrides) -> TransformerBridgeConfig:
    """Minimal TransformerBridgeConfig with the standard tiny test dims.

    Defaults: d_model=64, n_heads=8, n_layers=2, d_vocab=100, n_ctx=128,
    default_prepend_bos=False. d_head is derived from d_model/n_heads unless
    overridden. Pass any TransformerBridgeConfig field as a keyword.
    """
    cfg = dict(
        d_model=64,
        n_heads=8,
        n_layers=2,
        n_ctx=128,
        d_vocab=100,
        default_prepend_bos=False,
        architecture=architecture,
    )
    cfg.update(overrides)
    cfg.setdefault("d_head", cfg["d_model"] // cfg["n_heads"])
    return TransformerBridgeConfig(**cfg)
