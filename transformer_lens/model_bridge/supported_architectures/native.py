"""Architecture adapter for TL-native models built via ``boot_native``.

This adapter targets ``NativeModel`` ([sources/native/model.py]). Because the
native module's hierarchy is fully under our control, the component paths are
flat (no ``transformer.h.{i}`` prefix) and split-QKV is the natural layout —
no weight conversions are required for ordinary use.

The component mapping adapts to the cfg: gated MLP swaps in ``GatedMLPBridge``,
RMS norm swaps in ``RMSNormalizationBridge``, rotary skips ``pos_embed``, and
``attn_only`` drops the MLP branch.
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


def _uses_rms(cfg: Any) -> bool:
    return (getattr(cfg, "normalization_type", None) or "LN").upper() in ("RMS", "RMSPRE")


def _is_rotary(cfg: Any) -> bool:
    return (getattr(cfg, "positional_embedding_type", None) or "standard").lower() == "rotary"


def _make_norm_bridge(name: str, cfg: Any, *, force_rms: bool = False):
    if force_rms or _uses_rms(cfg):
        return RMSNormalizationBridge(name=name, config=cfg)
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

        # Native layout already stores Q/K/V split per-head in [d_model, n*d_head]
        # form. We skip weight_processing_conversions for ordinary use; compatibility
        # mode (fold_ln + center_writing_weights) can be added in a follow-up.
        # Until then, gate the corresponding ProcessWeights paths off: without the
        # state-dict key conversions wired up, folding would silently mis-place
        # weights or raise on missing keys.
        self.supports_fold_ln = False
        self.supports_center_writing_weights = False
        self.weight_processing_conversions = {}

        # Native model uses non-colliding attribute names ("tok_embed", "layers",
        # "ln_out", "head") because the bridge's __getattr__ forwards unknown
        # names to original_model.<name>, which would shadow the bridge's own
        # component slots ("embed", "blocks", "ln_final", "unembed") during
        # add_module if they matched 1:1.
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
        # Under attn_only the ln2 and mlp submodules are absent, but
        # BlockBridge's class-level hook_aliases still points
        # ``hook_resid_mid -> ln2.hook_in`` and ``hook_mlp_out -> mlp.hook_out``.
        # _register_aliases warns when those don't resolve. Drop them so the
        # warnings stay meaningful elsewhere — the pattern mirrors
        # ParallelBlockBridge ([block.py:405-407]).
        if self.cfg.attn_only:
            if block_bridge.hook_aliases is BlockBridge.hook_aliases:
                block_bridge.hook_aliases = dict(block_bridge.hook_aliases)
            block_bridge.hook_aliases.pop("hook_resid_mid", None)
            block_bridge.hook_aliases.pop("hook_mlp_out", None)
        mapping["blocks"] = block_bridge
        # ``final_rms`` opts into RMSNorm on the final norm regardless of
        # whether the blocks themselves use RMS — Llama-style configs do this.
        mapping["ln_final"] = _make_norm_bridge(
            "ln_out", self.cfg, force_rms=bool(getattr(self.cfg, "final_rms", False))
        )
        mapping["unembed"] = UnembeddingBridge(name="head")
        self.component_mapping = mapping

    def prepare_model(self, model: Any) -> None:
        """Reject modules whose attribute names would collide with bridge slots.

        The reserved-slot set is derived from ``self.component_mapping.keys()``
        at call time — single source of truth. A future variant that adds (or
        omits) a top-level slot extends the collision check automatically; no
        sibling list to keep in sync.

        The bridge's ``__getattr__`` falls back to ``getattr(original_model, name)``
        for unknown attributes — that resolves submodules, registered buffers,
        plain tensors set with ``self.x = ...``, and any property. Any of these
        will make ``add_module`` raise during bridge setup. We use ``hasattr``
        (not ``name in model._modules``) so the check covers all attribute
        shapes, not just registered nn.Modules.

        Failing here makes the diagnostic point at the real cause instead of a
        ``KeyError: "attribute 'embed' already exists"`` deep in component
        setup.
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
