"""Architecture adapter for TL-native models built via ``boot_native``.

Component mapping adapts to cfg: gated MLP → ``GatedMLPBridge``, RMS norm →
``RMSNormalizationBridge``, rotary drops ``pos_embed``, ``attn_only`` drops MLP.
"""

from typing import Any

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    MLPBridge,
    NormalizationBridge,
    PosEmbedBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


def _uses_rms(cfg: Any) -> bool:
    return (getattr(cfg, "normalization_type", None) or "LN").upper() in ("RMS", "RMSPRE")


def _uses_no_norm(cfg: Any) -> bool:
    return getattr(cfg, "normalization_type", None) is None


def _is_rotary(cfg: Any) -> bool:
    return (getattr(cfg, "positional_embedding_type", None) or "standard").lower() == "rotary"


def _make_norm_bridge(name: str, cfg: Any, *, force_rms: bool = False):
    if force_rms or _uses_rms(cfg):
        return RMSNormalizationBridge(name=name, config=cfg)
    if _uses_no_norm(cfg):
        return GeneralizedComponent(name=name, config=cfg)
    return NormalizationBridge(name=name, config=cfg)


def _make_mlp_bridge(cfg: Any):
    if cfg.gated_mlp:
        return GatedMLPBridge(
            name="mlp",
            config=cfg,
            submodules={
                "gate": LinearBridge(name="gate"),
                "in": LinearBridge(name="in"),
                "out": LinearBridge(name="out"),
            },
        )
    return MLPBridge(
        name="mlp",
        submodules={
            "in": LinearBridge(name="fc_in"),
            "out": LinearBridge(name="fc_out"),
        },
    )


def _make_block_submodules(cfg: Any) -> dict:
    submods: dict = {
        "ln1": _make_norm_bridge("ln1", cfg),
        "attn": AttentionBridge(
            name="attn",
            config=cfg,
            submodules={
                "q": LinearBridge(name="q"),
                "k": LinearBridge(name="k"),
                "v": LinearBridge(name="v"),
                "o": LinearBridge(name="o"),
            },
        ),
    }
    if not cfg.attn_only:
        submods["ln2"] = _make_norm_bridge("ln2", cfg)
        submods["mlp"] = _make_mlp_bridge(cfg)
    return submods


class NativeArchitectureAdapter(ArchitectureAdapter):
    """Adapter for ``NativeModel`` — TL-native, split-QKV, pre-LN; feature set
    driven by cfg (gated MLP, RMS norm, rotary, GQA, soft-cap, attn_only)."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        # Native layout already stores Q/K/V split; no rearranges needed.
        # Compatibility-mode fold_ln / center_writing_weights aren't wired up,
        # so gate the corresponding ProcessWeights paths off — folding without
        # the state-dict conversions would mis-place or drop weights.
        self.supports_fold_ln = False
        self.supports_center_writing_weights = False
        self.weight_processing_conversions = {}

        # Internal attribute names avoid collisions with bridge slot names
        # ("embed", "blocks", "ln_final", "unembed") — the bridge's __getattr__
        # forwards to original_model and would shadow add_module otherwise.
        mapping: dict = {
            "embed": EmbeddingBridge(name="tok_embed"),
        }
        if not _is_rotary(cfg):
            mapping["pos_embed"] = PosEmbedBridge(name="pos")
        block_bridge = BlockBridge(
            name="layers",
            config=self.cfg,
            submodules=_make_block_submodules(self.cfg),
        )
        # Under attn_only there's no ln2 / mlp to point at; drop the aliases
        # that would otherwise warn during _register_aliases.
        if self.cfg.attn_only:
            if block_bridge.hook_aliases is BlockBridge.hook_aliases:
                block_bridge.hook_aliases = dict(block_bridge.hook_aliases)
            block_bridge.hook_aliases.pop("hook_resid_mid", None)
            block_bridge.hook_aliases.pop("hook_mlp_out", None)
        mapping["blocks"] = block_bridge
        # final_rms forces RMS on the final norm independent of block norm —
        # matches Llama's TL config semantic.
        mapping["ln_final"] = _make_norm_bridge(
            "ln_out", self.cfg, force_rms=bool(getattr(self.cfg, "final_rms", False))
        )
        mapping["unembed"] = UnembeddingBridge(name="head")
        self.component_mapping = mapping

    def prepare_model(self, model: Any) -> None:
        """Reject modules whose attribute names collide with bridge slots.

        Bridge's ``__getattr__`` falls back to ``getattr(original_model, name)``
        for unknown attrs, so a name match — submodule, buffer, plain tensor,
        or property — makes ``add_module`` raise mid-setup with an opaque
        message. Failing here points at the real cause. Reserved set is derived
        from ``component_mapping.keys()`` so adapter variants stay in sync.
        """
        reserved = set(self.component_mapping.keys()) if self.component_mapping else set()
        collisions = sorted(name for name in reserved if hasattr(model, name))
        if collisions:
            raise ValueError(
                f"{type(model).__name__} cannot be wrapped by NativeArchitectureAdapter: "
                f"attribute names {collisions} collide with bridge component slots "
                f"({sorted(reserved)}). Rename these attributes to non-colliding names "
                f"(e.g. tok_embed, layers, ln_out, head) and update the adapter's "
                f"component_mapping ``name=`` fields to match."
            )
