"""Baichuan architecture adapter.

Supports both BaiChuanForCausalLM (v1) and BaichuanForCausalLM (v2).
Both use combined QKV via W_pack with RoPE, RMSNorm, and gated MLP.
"""

import importlib.util
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


class _BaichuanAttentionBridge(JointQKVPositionEmbeddingsAttentionBridge):
    """Attention bridge for Baichuan's v4-era decoder-layer contract.

    Baichuan predates HF's Cache API and differs from the base bridge in two
    ways we have to own:

    1. **Rotary from position_ids**: HF passes `position_ids` (not a
       pre-computed `position_embeddings` tuple), so we call the per-layer
       `rotary_emb(v, seq_len=kv_seq_len)` ourselves and slice cos/sin by
       `position_ids`.
    2. **Legacy (k, v) cache tuple**: HF's DecoderLayer passes
       `past_key_value=(k, v)` (singular, per-layer legacy tuple) and expects
       `self_attn(...)` to return a matching `(k_full, v_full)` as
       `present_key_value` so Model.forward's `next_decoder_cache` accumulates
       real tensors. The base bridge's `_update_kv_cache` only handles the
       Cache-object plural path, so we reimplement the attention body here
       (mirroring HF's own Attention.forward).
    """

    def _reconstruct_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs
    ) -> tuple:
        assert self.original_component is not None
        assert self.config is not None
        num_heads = self.config.n_heads
        num_kv_heads = getattr(self.config, "n_key_value_heads", None) or num_heads

        q, k, v, batch_size, seq_len, head_dim = self._reshape_qkv_to_heads(
            q, k, v, num_heads, num_kv_heads
        )

        past_kv_raw = kwargs.get("past_key_value")
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None
        if (
            isinstance(past_kv_raw, tuple)
            and len(past_kv_raw) >= 2
            and isinstance(past_kv_raw[0], torch.Tensor)
            and isinstance(past_kv_raw[1], torch.Tensor)
        ):
            past_key_value = (past_kv_raw[0], past_kv_raw[1])
        past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0

        # Rotary: derive cos/sin over the full kv_seq_len, index by position_ids.
        if "position_embeddings" not in kwargs:
            rotary_emb = getattr(self.original_component, "rotary_emb", None)
            position_ids = kwargs.get("position_ids")
            if rotary_emb is not None and position_ids is not None:
                kv_seq_len = seq_len + past_len
                cos, sin = rotary_emb(v, seq_len=kv_seq_len)
                cos = cos.squeeze(1).squeeze(0)[position_ids]
                sin = sin.squeeze(1).squeeze(0)[position_ids]
                kwargs["position_embeddings"] = (cos, sin)

        position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and isinstance(position_embeddings, tuple):
            cos, sin = self._apply_position_embedding_hooks(position_embeddings)
            q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        # Concat prior (k, v) — already rotary-applied from its own step.
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=-2)
            v = torch.cat([past_key_value[1], v], dim=-2)

        # Build present cache from pre-GQA-expansion (k, v) so downstream
        # steps don't pay for duplicated heads.
        use_cache = bool(kwargs.get("use_cache", False))
        present_key_value = (k, v) if use_cache else None

        if num_kv_heads != num_heads:
            n_rep = num_heads // num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        kv_seq_len = k.shape[-2]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** (-0.5))
        attention_mask = kwargs.get("attention_mask", None)
        attn_scores = self._apply_reconstruct_attention_mask(
            attn_scores=attn_scores,
            attention_mask=attention_mask,
            seq_len=kv_seq_len,
            q_seq_len=seq_len,
        )
        attn_scores = self.hook_attn_scores(attn_scores)
        attn_weights = self._softmax_dropout_pattern(attn_scores)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = self._reshape_attn_output(
            attn_output, batch_size, seq_len, num_heads, head_dim
        )
        if (
            bool(getattr(self.config, "use_attn_result", False))
            and hasattr(self, "o")
            and self.o.original_component is not None
        ):
            attn_output = self.o.hook_in(attn_output)
            z_4d = attn_output.view(batch_size, seq_len, num_heads, head_dim)
            attn_output = self._compute_per_head_result(z_4d, num_heads, head_dim)
        else:
            attn_output = self._apply_output_projection(attn_output)

        return (attn_output, attn_weights, present_key_value)


def _patch_init_weights_for_baichuan() -> None:
    """Prevent _init_weights from re-randomizing loaded checkpoint weights.

    Transformers v5 calls _init_weights on all modules after weight
    materialization. For modules with real (non-meta) tensors, we must
    skip re-initialization to preserve the loaded checkpoint values.
    """
    for key in list(sys.modules.keys()):
        if "baichuan" not in key.lower() or "modeling" not in key.lower():
            continue
        module = sys.modules[key]
        # Both v1 (BaiChuan) and v2 (Baichuan) define a PreTrainedModel subclass
        for cls_name in ("BaiChuanPreTrainedModel", "BaichuanPreTrainedModel", "PreTrainedModel"):
            pretrained_cls = getattr(module, cls_name, None)
            if pretrained_cls is None or getattr(pretrained_cls, "_tl_patched", False):
                continue
            # Only patch classes that define their own _init_weights
            if "_init_weights" not in pretrained_cls.__dict__:
                continue

            original_init_weights = pretrained_cls._init_weights

            def safe_init_weights(self, mod, _original=original_init_weights):  # type: ignore[no-untyped-def]
                first_param = next(mod.parameters(), None)
                if first_param is not None and first_param.device.type != "meta":
                    return
                _original(self, mod)

            pretrained_cls._init_weights = safe_init_weights
            pretrained_cls._tl_patched = True


class BaichuanArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for Baichuan models (v1 and v2).

    Baichuan uses combined QKV via W_pack (nn.Linear(h, 3*h)) with RoPE,
    RMSNorm, and gated MLP (SwiGLU). Per-layer rotary embeddings.

    Optional Parameters (may not exist in state_dict):
    -------------------------------------------------
    Baichuan models do NOT have biases on any projection:

    - blocks.{i}.attn.b_Q / b_K / b_V / b_O — no bias
    - blocks.{i}.mlp.b_gate / b_in / b_out — no bias
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

        # Fused W_pack prevents standard fold_ln from reaching Q/K/V separately.
        # preprocess_weights() handles it instead.
        self.supports_fold_ln = False

        self.weight_processing_conversions = {
            "blocks.{i}.attn.q.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=cfg.n_heads),
            ),
            "blocks.{i}.attn.k.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=cfg.n_heads),
            ),
            "blocks.{i}.attn.v.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=cfg.n_heads),
            ),
            "blocks.{i}.attn.o.weight": ParamProcessingConversion(
                tensor_conversion=RearrangeTensorConversion("m (n h) -> n h m", n=cfg.n_heads),
            ),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.embed_tokens"),
            "blocks": BlockBridge(
                name="model.layers",
                submodules={
                    "ln1": RMSNormalizationBridge(name="input_layernorm", config=self.cfg),
                    "ln2": RMSNormalizationBridge(name="post_attention_layernorm", config=self.cfg),
                    "attn": _BaichuanAttentionBridge(
                        name="self_attn",
                        config=self.cfg,
                        split_qkv_matrix=self._split_baichuan_w_pack,
                        submodules={
                            "qkv": LinearBridge(name="W_pack"),
                            "o": LinearBridge(name="o_proj"),
                        },
                    ),
                    "mlp": GatedMLPBridge(
                        name="mlp",
                        config=self.cfg,
                        submodules={
                            "gate": LinearBridge(name="gate_proj"),
                            "in": LinearBridge(name="up_proj"),
                            "out": LinearBridge(name="down_proj"),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.norm", config=self.cfg),
            "unembed": UnembeddingBridge(name="lm_head", config=self.cfg),
        }

    def _split_baichuan_w_pack(
        self, attention_component: Any
    ) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
        """Split Baichuan's W_pack into separate Q, K, V linear modules.

        W_pack is a simple concatenation: [Q | K | V], each of size hidden_size.
        No interleaving, no GQA — all three chunks are equal size.
        """
        w_pack = attention_component.W_pack
        weight = w_pack.weight.data
        d_model = weight.shape[1]
        hidden_size = d_model  # Q, K, V each have hidden_size output features

        q_w = weight[:hidden_size, :]
        k_w = weight[hidden_size : 2 * hidden_size, :]
        v_w = weight[2 * hidden_size :, :]

        def _make_linear(w: torch.Tensor) -> nn.Linear:
            lin = nn.Linear(d_model, hidden_size, bias=False)
            lin.weight = nn.Parameter(w)
            return lin

        return _make_linear(q_w), _make_linear(k_w), _make_linear(v_w)

    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Inject per-layer rotary embedding for component testing."""
        try:
            rotary_emb = hf_model.model.layers[0].self_attn.rotary_emb
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
        patch_dynamic_cache_v5()

        # Force-import the remote modeling module so we can patch _init_weights.
        # Baichuan2 variants ship quantizer.py which imports bitsandbytes;
        # transformers' check_imports scans every .py file in the repo and
        # raises ImportError if bitsandbytes is missing, even though quantizer
        # is not used in normal inference. Catch that case and tell the user
        # how to install the optional dependency group.
        try:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            last_exc: Exception | None = None
            # Try both class names (v1 and v2)
            for cls_name in (
                "modeling_baichuan.BaichuanForCausalLM",
                "modeling_baichuan.BaiChuanForCausalLM",
            ):
                try:
                    get_class_from_dynamic_module(cls_name, model_name)
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    continue
            if last_exc is not None and "bitsandbytes" in str(last_exc):
                if importlib.util.find_spec("bitsandbytes") is None:
                    raise ImportError(
                        "Baichuan2 variants require `bitsandbytes` for "
                        "trust_remote_code loading (their shipped quantizer.py "
                        "imports it). Install the quantization extras: "
                        "`uv sync --group quantization`."
                    ) from last_exc
        except ImportError:
            raise
        except Exception:
            pass

        _patch_init_weights_for_baichuan()

    def prepare_model(self, hf_model: Any) -> None:
        """Fix rotary caches and normalize NormHead weights before bridge creation.

        RotaryEmbedding differs between v1 and v2:
        - v1 (Baichuan-7B): `inv_freq` is a persistent buffer, loaded from the
          checkpoint as bfloat16, but `cos_cached`/`sin_cached` are non-persistent
          and materialize as garbage under meta-init.
        - v2 (Baichuan2-*): `inv_freq`, `cos_cached`, `sin_cached` are all plain
          attributes (no `register_buffer`). v5's meta-init materializes them on
          meta, and nothing in the checkpoint overwrites them.

        Both cases are resolved by computing inv_freq + caches from scratch at
        float32 using config-derived head_dim and base=10000. Recomputing v1 at
        float32 is also an upgrade over its bfloat16 checkpoint values.

        Baichuan2 Chat also uses NormHead which row-normalizes lm_head during
        forward. We apply that once here so the bridge sees the normalized
        weights directly without needing NormHead's forward path.
        """
        # Pick a real device/dtype by scanning real (non-meta) parameters.
        target_device = torch.device("cpu")
        params_fn = getattr(hf_model, "parameters", None)
        if callable(params_fn):
            for param in params_fn():
                if param.device.type != "meta":
                    target_device = param.device
                    break

        head_dim = self.cfg.d_model // self.cfg.n_heads
        base = 10000.0

        model_core = getattr(hf_model, "model", None)
        if model_core is not None:
            for layer in getattr(model_core, "layers", []):
                rotary = getattr(getattr(layer, "self_attn", None), "rotary_emb", None)
                if rotary is None:
                    continue
                max_seq = getattr(rotary, "max_seq_len_cached", self.cfg.n_ctx or 4096)
                inv_freq = 1.0 / (
                    base
                    ** (
                        torch.arange(0, head_dim, 2, device=target_device, dtype=torch.float32)
                        / head_dim
                    )
                )
                t = torch.arange(max_seq, device=target_device, dtype=torch.float32)
                freqs = torch.einsum("i,j->ij", t, inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                rotary.inv_freq = inv_freq
                rotary.cos_cached = emb.cos()[None, None, :, :]
                rotary.sin_cached = emb.sin()[None, None, :, :]

        # Normalize NormHead weights (Baichuan2 Chat)
        lm_head = getattr(hf_model, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "first_flag"):
            w = lm_head.weight.data
            lm_head.weight.data = torch.nn.functional.normalize(w, dim=-1)

    def preprocess_weights(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Split fused W_pack QKV and optionally fold layer norms."""
        fold_ln = getattr(self, "_fold_ln_requested", True)
        if not fold_ln:
            # Still need to split W_pack into Q/K/V for weight conversions
            for i in range(self.cfg.n_layers):
                qkv_key = f"blocks.{i}.attn.qkv.weight"
                if qkv_key not in state_dict:
                    continue
                w = state_dict[qkv_key]
                hidden_size = w.shape[1]
                q_w = w[:hidden_size, :]
                k_w = w[hidden_size : 2 * hidden_size, :]
                v_w = w[2 * hidden_size :, :]
                state_dict[f"blocks.{i}.attn.q.weight"] = q_w
                state_dict[f"blocks.{i}.attn.k.weight"] = k_w
                state_dict[f"blocks.{i}.attn.v.weight"] = v_w
                del state_dict[qkv_key]
            return state_dict

        for i in range(self.cfg.n_layers):
            # --- Fold ln1 into Q/K/V (split from W_pack) ---
            qkv_key = f"blocks.{i}.attn.qkv.weight"
            ln1_key = f"blocks.{i}.ln1.weight"
            if qkv_key in state_dict and ln1_key in state_dict:
                ln1_w = state_dict[ln1_key].float()
                w = state_dict[qkv_key].float()
                orig_dtype = state_dict[qkv_key].dtype
                hidden_size = w.shape[1]

                q_w = w[:hidden_size, :]
                k_w = w[hidden_size : 2 * hidden_size, :]
                v_w = w[2 * hidden_size :, :]

                state_dict[f"blocks.{i}.attn.q.weight"] = (q_w * ln1_w[None, :]).to(orig_dtype)
                state_dict[f"blocks.{i}.attn.k.weight"] = (k_w * ln1_w[None, :]).to(orig_dtype)
                state_dict[f"blocks.{i}.attn.v.weight"] = (v_w * ln1_w[None, :]).to(orig_dtype)
                del state_dict[qkv_key]
                state_dict[ln1_key] = torch.ones_like(state_dict[ln1_key])

            # --- Fold ln2 into MLP gate and up projections ---
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
