"""InternLM2 architecture adapter."""

import sys
from typing import Any

import torch
import torch.nn as nn

from transformer_lens.conversion_utils.conversion_steps import RearrangeTensorConversion
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.compat import patch_dynamic_cache_v5
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    JointQKVPositionEmbeddingsAttentionBridge,
    LinearBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
)


class _InternLM2AttentionBridge(JointQKVPositionEmbeddingsAttentionBridge):
    """Attention bridge returning 3-tuple for InternLM2's decoder layer contract.

    InternLM2's decoder layer unpacks (hidden_states, attn_weights, present_key_value)
    from self.attention(), but the base bridge returns only (output, weights).
    """

    def _reconstruct_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs
    ) -> tuple:
        attn_output, attn_weights = super()._reconstruct_attention(q, k, v, **kwargs)
        past_key_value = kwargs.get("past_key_values", kwargs.get("past_key_value", None))
        return (attn_output, attn_weights, past_key_value)


def _patch_init_weights_for_internlm2() -> None:
    """Prevent _init_weights from re-randomizing loaded checkpoint weights.

    Transformers v5 calls _init_weights on all modules after weight
    materialization. For modules with real (non-meta) tensors, we must
    skip re-initialization to preserve the loaded checkpoint values.
    Same approach as openelm.py.
    """
    for key in list(sys.modules.keys()):
        if "internlm2" not in key.lower() or "modeling" not in key.lower():
            continue
        module = sys.modules[key]
        pretrained_cls = getattr(module, "InternLM2PreTrainedModel", None)
        if pretrained_cls is None or getattr(pretrained_cls, "_tl_patched", False):
            continue

        original_init_weights = pretrained_cls._init_weights

        def safe_init_weights(self, mod, _original=original_init_weights):  # type: ignore[no-untyped-def]
            first_param = next(mod.parameters(), None)
            if first_param is not None and first_param.device.type != "meta":
                return
            _original(self, mod)

        pretrained_cls._init_weights = safe_init_weights
        pretrained_cls._tl_patched = True


class InternLM2ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for InternLM2 models.

    InternLM2 uses remote code (trust_remote_code=True) and differs from Llama in:
    - Fused interleaved GQA wqkv weight (not standard [Q|K|V] split)
    - Non-standard module names: tok_embeddings, output, attention, feed_forward,
      wqkv/wo, w1(gate)/w3(up)/w2(down), attention_norm, ffn_norm
    - Per-layer rotary_emb (no model-level shared instance)
    - supports_fold_ln=False: fold_ln is done manually in preprocess_weights because
      the bridge state dict has the fused qkv key, not split q/k/v keys, so
      fold_layer_norm's extract_attention_tensors_for_folding would silently skip attn.

    Optional parameters (may not exist in state_dict):
    - blocks.{i}.attn.b_Q / b_K / b_V / b_O — config.bias=False on shipped models
    - blocks.{i}.mlp.b_gate / b_in / b_out — MLP always bias=False
    - blocks.{i}.ln1.b / ln2.b / ln_final.b — RMSNorm has no bias
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        self.cfg.normalization_type = "RMS"
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False
        self.cfg.uses_rms_norm = True
        self.cfg.eps_attr = "variance_epsilon"

        # Standard fold_ln silently skips attention when wqkv is fused (see class docstring).
        # preprocess_weights() handles it instead — same approach as phi3.py.
        self.supports_fold_ln = False

        if hasattr(cfg, "n_key_value_heads") and cfg.n_key_value_heads is not None:
            self.cfg.n_key_value_heads = cfg.n_key_value_heads

        n_kv_heads = getattr(cfg, "n_key_value_heads", None) or cfg.n_heads

        self.weight_processing_conversions = {
            "blocks.{i}.attn.q.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=cfg.n_heads),
            ),
            "blocks.{i}.attn.k.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=n_kv_heads),
            ),
            "blocks.{i}.attn.v.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=n_kv_heads),
            ),
            "blocks.{i}.attn.o.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("m (n h) -> n h m", n=cfg.n_heads),
            ),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.tok_embeddings"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="attention_norm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="ffn_norm", config=self.cfg),
                    "attn": _InternLM2AttentionBridge(
                        name="attention",
                        config=self.cfg,
                        split_qkv_matrix=self._split_internlm2_wqkv,
                        submodules={
                            "qkv": LinearBridge(name="wqkv"),
                            "o": LinearBridge(name="wo"),
                        },
                    ),
                    "mlp": GatedMLPBridge(
                        name="feed_forward",
                        config=self.cfg,
                        submodules={
                            "gate": LinearBridge(name="w1"),
                            "in": LinearBridge(name="w3"),
                            "out": LinearBridge(name="w2"),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="output", config=self.cfg),
        }

    def _split_internlm2_wqkv(
        self, attention_component: Any
    ) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
        """Split InternLM2's interleaved wqkv into separate Q, K, V linear modules.

        InternLM2 uses an interleaved GQA layout rather than the standard [Q_all|K_all|V_all].
        For each of n_kv_heads groups, the weight rows are:
          [q0, q1, ..., q(n_kv_groups-1), k, v]  (each slot = head_dim rows)
        i.e. gs = n_kv_groups + 2 slots per kv-head group.
        """
        wqkv = attention_component.wqkv
        w = wqkv.weight.data
        d_model = w.shape[1]
        has_bias = wqkv.bias is not None

        n_kv_heads = getattr(self.cfg, "n_key_value_heads", None) or self.cfg.n_heads
        n_kv_groups = self.cfg.n_heads // n_kv_heads
        head_dim = self.cfg.d_model // self.cfg.n_heads
        gs = n_kv_groups + 2

        w_grouped = w.reshape(n_kv_heads, gs, head_dim, d_model)
        q_w = w_grouped[:, :n_kv_groups, :, :].reshape(self.cfg.n_heads * head_dim, d_model)
        k_w = w_grouped[:, n_kv_groups, :, :].reshape(n_kv_heads * head_dim, d_model)
        v_w = w_grouped[:, n_kv_groups + 1, :, :].reshape(n_kv_heads * head_dim, d_model)

        q_b: torch.Tensor | None = None
        k_b: torch.Tensor | None = None
        v_b: torch.Tensor | None = None
        if has_bias:
            b = wqkv.bias.data
            b_grouped = b.reshape(n_kv_heads, gs, head_dim)
            q_b = b_grouped[:, :n_kv_groups, :].reshape(self.cfg.n_heads * head_dim)
            k_b = b_grouped[:, n_kv_groups, :].reshape(n_kv_heads * head_dim)
            v_b = b_grouped[:, n_kv_groups + 1, :].reshape(n_kv_heads * head_dim)

        def _make_linear(weight: torch.Tensor, bias: torch.Tensor | None) -> nn.Linear:
            lin = nn.Linear(d_model, weight.shape[0], bias=bias is not None)
            lin.weight = nn.Parameter(weight)
            if bias is not None:
                lin.bias = nn.Parameter(bias)
            return lin

        return _make_linear(q_w, q_b), _make_linear(k_w, k_b), _make_linear(v_w, v_b)

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Inject per-layer rotary embedding for component testing."""
        try:
            rotary_emb = hf_model.model.layers[0].attention.rotary_emb
        except (AttributeError, IndexError):
            return

        if bridge_model is not None and hasattr(bridge_model, "blocks"):
            for block in bridge_model.blocks:
                if hasattr(block, "attn"):
                    block.attn.set_rotary_emb(rotary_emb)

        attn_bridge = self.get_generalized_component("blocks.0.attn")
        attn_bridge.set_rotary_emb(rotary_emb)

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Patch transformers v5 incompatibilities before from_pretrained runs."""
        config = model_kwargs.get("config")
        if config is not None:
            tp = getattr(config, "pretraining_tp", 1)
            if tp > 1:
                raise ValueError(
                    f"InternLM2 adapter does not support pretraining_tp={tp}; "
                    "only pretraining_tp=1 is supported for logit correctness."
                )

        patch_dynamic_cache_v5()

        # Force-import the remote modeling module so we can patch _init_weights.
        try:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            get_class_from_dynamic_module(
                "modeling_internlm2.InternLM2ForCausalLM",
                model_name,
            )
        except Exception:
            pass

        _patch_init_weights_for_internlm2()

    def preprocess_weights(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Fold layer norms into QKV and MLP weights.

        Standard fold_ln can't reach split Q/K/V when wqkv is fused in the bridge state dict.
        We extract and fold here, then write split keys so RearrangeTensorConversion can follow.
        MLP projections (w1/w2/w3) are separate linears so they fold normally.
        Mirrors phi3.py.preprocess_weights, adapted for InternLM2's layout.
        """
        fold_ln = getattr(self, "_fold_ln_requested", True)
        if not fold_ln:
            return state_dict

        n_kv_heads = getattr(self.cfg, "n_key_value_heads", None) or self.cfg.n_heads
        n_kv_groups = self.cfg.n_heads // n_kv_heads
        head_dim = self.cfg.d_model // self.cfg.n_heads
        gs = n_kv_groups + 2

        for i in range(self.cfg.n_layers):
            # --- Fold ln1 into Q/K/V (extracted from interleaved wqkv) ---
            qkv_key = f"blocks.{i}.attn.qkv.weight"
            ln1_key = f"blocks.{i}.ln1.weight"
            if qkv_key in state_dict and ln1_key in state_dict:
                ln1_w = state_dict[ln1_key].float()
                qkv_w = state_dict[qkv_key].float()
                d_model = qkv_w.shape[1]
                orig_dtype = state_dict[qkv_key].dtype

                w_grouped = qkv_w.reshape(n_kv_heads, gs, head_dim, d_model)
                q_w = w_grouped[:, :n_kv_groups, :, :].reshape(self.cfg.n_heads * head_dim, d_model)
                k_w = w_grouped[:, n_kv_groups, :, :].reshape(n_kv_heads * head_dim, d_model)
                v_w = w_grouped[:, n_kv_groups + 1, :, :].reshape(n_kv_heads * head_dim, d_model)

                state_dict[f"blocks.{i}.attn.q.weight"] = (q_w * ln1_w[None, :]).to(orig_dtype)
                state_dict[f"blocks.{i}.attn.k.weight"] = (k_w * ln1_w[None, :]).to(orig_dtype)
                state_dict[f"blocks.{i}.attn.v.weight"] = (v_w * ln1_w[None, :]).to(orig_dtype)
                del state_dict[qkv_key]
                state_dict[ln1_key] = torch.ones_like(state_dict[ln1_key])

            qkv_bias_key = f"blocks.{i}.attn.qkv.bias"
            if qkv_bias_key in state_dict:
                b = state_dict[qkv_bias_key]
                expected_len = (self.cfg.n_heads + 2 * n_kv_heads) * head_dim
                if b.shape[0] != expected_len:
                    raise ValueError(
                        f"Unexpected wqkv bias shape at layer {i}: {b.shape[0]} "
                        f"(expected {expected_len}). Cannot split interleaved bias."
                    )
                orig_dtype = b.dtype
                b_f = b.float()
                b_grouped = b_f.reshape(n_kv_heads, gs, head_dim)
                q_b = b_grouped[:, :n_kv_groups, :].reshape(self.cfg.n_heads * head_dim)
                k_b = b_grouped[:, n_kv_groups, :].reshape(n_kv_heads * head_dim)
                v_b = b_grouped[:, n_kv_groups + 1, :].reshape(n_kv_heads * head_dim)
                state_dict[f"blocks.{i}.attn.q.bias"] = q_b.to(orig_dtype)
                state_dict[f"blocks.{i}.attn.k.bias"] = k_b.to(orig_dtype)
                state_dict[f"blocks.{i}.attn.v.bias"] = v_b.to(orig_dtype)
                del state_dict[qkv_bias_key]

            # --- Fold ln2 into MLP gate (w1) and up (w3) projections ---
            ln2_key = f"blocks.{i}.ln2.weight"
            if ln2_key in state_dict:
                ln2_w = state_dict[ln2_key].float()
                for mlp_key in [
                    f"blocks.{i}.mlp.gate.weight",
                    f"blocks.{i}.mlp.in.weight",
                ]:
                    if mlp_key in state_dict:
                        orig_dtype = state_dict[mlp_key].dtype
                        state_dict[mlp_key] = (state_dict[mlp_key].float() * ln2_w[None, :]).to(
                            orig_dtype
                        )
                state_dict[ln2_key] = torch.ones_like(state_dict[ln2_key])

        # --- Fold ln_final into unembed ---
        ln_final_key = "ln_final.weight"
        unembed_key = "unembed.weight"
        if ln_final_key in state_dict and unembed_key in state_dict:
            ln_w = state_dict[ln_final_key].float()
            u_w = state_dict[unembed_key].float()
            orig_dtype = state_dict[unembed_key].dtype
            if u_w.shape[-1] == ln_w.shape[0]:
                state_dict[unembed_key] = (u_w * ln_w[None, :]).to(orig_dtype)
            elif u_w.shape[0] == ln_w.shape[0]:
                state_dict[unembed_key] = (u_w * ln_w[:, None]).to(orig_dtype)
            state_dict[ln_final_key] = torch.ones_like(state_dict[ln_final_key])

        return state_dict
