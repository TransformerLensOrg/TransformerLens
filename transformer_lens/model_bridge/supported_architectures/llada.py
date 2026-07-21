"""LLaDA architecture support for one masked-token transformer forward pass.

LLaDA is a masked discrete-diffusion model. Its transformer uses bidirectional
self-attention and returns logits for every input position; the iterative
denoising sampler lives outside the model. This adapter intentionally supports
only the transformer forward pass. TransformerBridge's autoregressive
generation APIs are disabled for this architecture.

The remote ``LLaDALlamaBlock`` owns its Q/K/V, output, and gated-MLP projections
directly instead of grouping them in attention and MLP modules. The adapter
preserves the reviewed remote block forward and replaces only its
``attention(...)`` method with a hook-aware reconstruction. This keeps the
native pre-norm residual order, dropout, RoPE implementation, and MLP math while
exposing the standard attention scores and pattern hooks.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, cast

import torch

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    AttentionBridge,
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    RMSNormalizationBridge,
    UnembeddingBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class _LLaDAAttentionBridge(AttentionBridge):
    """Reconstruct LLaDA's block-local, bidirectional attention method."""

    supports_split_qkv_fork: bool = False
    supports_attn_result: bool = True

    def set_original_component(self, original_component: torch.nn.Module) -> None:
        """Patch the owning block without registering it as a child module."""
        object.__setattr__(self, "_original_block", original_component)
        object.__setattr__(original_component, "attention", self)

    @property
    def original_component(self) -> Optional[torch.nn.Module]:
        """Return the owning LLaDA block."""
        return self.__dict__.get("_original_block")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """Run LLaDA attention with observable pre/post-softmax tensors."""
        block = self.original_component
        if block is None:
            raise RuntimeError("LLaDA attention is not attached to an owning block")
        config = self.config
        if config is None:
            raise RuntimeError("LLaDA attention is not attached to a bridge config")

        batch_size, query_len, q_width = q.shape
        n_heads = int(config.n_heads)
        n_kv_heads = int(getattr(config, "n_key_value_heads", None) or n_heads)
        head_dim = q_width // n_heads
        input_dtype = k.dtype

        q_norm = getattr(block, "q_norm", None)
        k_norm = getattr(block, "k_norm", None)
        if q_norm is not None and k_norm is not None:
            q = q_norm(q).to(dtype=input_dtype)
            k = k_norm(k).to(dtype=input_dtype)

        q = q.view(batch_size, query_len, n_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, query_len, n_kv_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, query_len, n_kv_heads, head_dim).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        present = (k, v) if use_cache else None

        if bool(getattr(block.config, "rope", False)):
            q, k = cast(Any, block.rotary_emb)(q, k)
        if hasattr(self, "hook_rot_q"):
            q = self.hook_rot_q(q)
        if hasattr(self, "hook_rot_k"):
            k = self.hook_rot_k(k)

        key_len = k.shape[-2]
        if attention_bias is not None:
            attention_bias = cast(Any, block._cast_attn_bias)(
                attention_bias[:, :, key_len - query_len : key_len, :key_len],
                input_dtype,
            )

        if n_heads != n_kv_heads:
            if n_heads % n_kv_heads != 0:
                raise ValueError(
                    f"n_heads ({n_heads}) must be divisible by n_key_value_heads " f"({n_kv_heads})"
                )
            groups = n_heads // n_kv_heads
            k = k.repeat_interleave(groups, dim=1, output_size=n_heads)
            v = v.repeat_interleave(groups, dim=1, output_size=n_heads)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn_scores = self._apply_reconstruct_attention_mask(
            attn_scores=attn_scores,
            attention_mask=attention_bias,
            seq_len=key_len,
            q_seq_len=query_len,
        )
        attn_scores = self.hook_attn_scores(attn_scores)

        pattern = torch.nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        dropout = float(getattr(block.config, "attention_dropout", 0.0))
        if block.training and dropout > 0.0:
            pattern = torch.nn.functional.dropout(pattern, p=dropout, training=True)
        pattern = self.hook_pattern(pattern)

        z = torch.matmul(pattern, v).transpose(1, 2).contiguous()
        flat_z = z.view(batch_size, query_len, n_heads * head_dim)
        if bool(getattr(self.config, "use_attn_result", False)):
            flat_z = self.o.hook_in(flat_z)
            z = flat_z.view(batch_size, query_len, n_heads, head_dim)
            output = self._compute_per_head_result(z, n_heads, head_dim)
        else:
            output = self.o(flat_z)
        output = self.hook_out(output)
        return output, present


class _LLaDABlockBridge(BlockBridge):
    """Preserve native block math while routing container-level hooks."""

    def __init__(
        self,
        name: str,
        config: Optional[Any] = None,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
    ) -> None:
        super().__init__(
            name,
            config=config,
            submodules=submodules,
            hook_alias_overrides={
                "hook_attn_in": "attn.hook_in",
                "hook_q_input": "attn.q.hook_in",
                "hook_k_input": "attn.k.hook_in",
                "hook_v_input": "attn.v.hook_in",
            },
        )
        self._llada_container_hooks_wired = False
        self._llada_container_hook_handles: list[torch.utils.hooks.RemovableHandle] = []

    def _route_attn_input(
        self, _module: torch.nn.Module, _args: tuple[Any, ...], output: torch.Tensor
    ) -> torch.Tensor:
        return self.attn.hook_in(output)

    def _route_mlp_input(
        self, _module: torch.nn.Module, _args: tuple[Any, ...], output: torch.Tensor
    ) -> torch.Tensor:
        return self.mlp.hook_in(output)

    def _route_mlp_output(
        self, _module: torch.nn.Module, _args: tuple[Any, ...], output: torch.Tensor
    ) -> torch.Tensor:
        if bool(getattr(self.mlp, "_executing_container", False)):
            return output
        return self.mlp.hook_out(output)

    def _route_pre_mlp_norm(
        self, _module: torch.nn.Module, args: tuple[Any, ...]
    ) -> Optional[tuple[Any, ...]]:
        if not self._read_use_hook_mlp_in():
            return None
        if args and isinstance(args[0], torch.Tensor):
            return (self.hook_mlp_in(args[0]),) + args[1:]
        return None

    def _maybe_wire_pre_ln_capture(self) -> None:
        """Use deepcopy-safe bound hooks for LLaDA's pre-MLP residual hook."""
        if self._pre_ln_capture_wired:
            return
        if self.ln2.original_component is not None:
            self._pre_ln_capture_handles.append(
                self.ln2.register_forward_pre_hook(self._route_pre_mlp_norm)
            )
        self._pre_ln_capture_wired = True

    def _wire_llada_container_hooks(self) -> None:
        if self._llada_container_hooks_wired:
            return

        self._llada_container_hook_handles.extend(
            (
                self.ln1.register_forward_hook(self._route_attn_input),
                self.ln2.register_forward_hook(self._route_mlp_input),
                self.mlp.out.register_forward_hook(self._route_mlp_output),
            )
        )
        self._llada_container_hooks_wired = True

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Wire LLaDA's symbolic containers, then delegate to the native block."""
        self._wire_llada_container_hooks()
        return super().forward(*args, **kwargs)


class _LLaDAGatedMLPBridge(GatedMLPBridge):
    """Executable view over LLaDA's block-local gated MLP projections."""

    def set_original_component(self, original_component: torch.nn.Module) -> None:
        """The executable view needs the block's children, not ownership of the block."""
        del original_component

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Run SiLU-gated MLP math using the live wrapped projections."""
        del kwargs
        hidden_states = self.hook_in(hidden_states)
        self._executing_container = True
        try:
            up_projection = getattr(self, "in")
            gate = self.gate(hidden_states)
            up = up_projection(hidden_states)
            activated = self.act(gate)
            gated = activated * up
            output = self.out(gated)
        finally:
            self._executing_container = False
        return self.hook_out(output)


class LLaDAArchitectureAdapter(ArchitectureAdapter):
    """Adapter for the dense ``LLaDAModelLM`` architecture.

    Support is deliberately limited to the released dense LLaDA block contract:
    Llama-style blocks, RMSNorm, separate bias-free projections, RoPE,
    bidirectional attention, an untied LM head, and no KV cache. The external
    iterative denoising/remasking loop is not a TransformerBridge generation API.
    Loading the Hugging Face checkpoint requires the caller to opt in with
    ``trust_remote_code=True``.
    """

    applicable_phases: list[int] = []
    supports_generation: bool = False
    supports_hf_output_attentions: bool = False
    supports_causal_loss: bool = False

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self._validate_config()

        self.cfg.d_vocab_out = self.cfg.d_vocab
        self.cfg.normalization_type = "RMS"
        self.cfg.uses_rms_norm = True
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.rotary_adjacent_pairs = False
        self.cfg.attention_dir = "bidirectional"
        self.cfg.final_rms = True
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False
        self.cfg.default_prepend_bos = False
        self.cfg.default_padding_side = "right"

        self.weight_processing_conversions = {
            **self._qkvo_weight_conversions(),
        }

        self.component_mapping = {
            "embed": EmbeddingBridge(name="model.transformer.wte", config=self.cfg),
            "blocks": _LLaDABlockBridge(
                name="model.transformer.blocks",
                config=self.cfg,
                submodules={
                    "ln1": RMSNormalizationBridge(name="attn_norm", config=self.cfg),
                    "attn": _LLaDAAttentionBridge(
                        name=None,
                        config=self.cfg,
                        is_causal=False,
                        submodules={
                            "q": LinearBridge(name="q_proj"),
                            "k": LinearBridge(name="k_proj"),
                            "v": LinearBridge(name="v_proj"),
                            "o": LinearBridge(name="attn_out"),
                        },
                    ),
                    "ln2": RMSNormalizationBridge(name="ff_norm", config=self.cfg),
                    "mlp": _LLaDAGatedMLPBridge(
                        name=None,
                        config=self.cfg,
                        submodules={
                            "gate": LinearBridge(name="ff_proj"),
                            "in": LinearBridge(name="up_proj"),
                            "act": GeneralizedComponent(name="act"),
                            "out": LinearBridge(name="ff_out"),
                        },
                    ),
                },
            ),
            "ln_final": RMSNormalizationBridge(name="model.transformer.ln_f", config=self.cfg),
            "unembed": UnembeddingBridge(name="model.transformer.ff_out", config=self.cfg),
        }

    @staticmethod
    def _enum_value(value: Any) -> Any:
        return getattr(value, "value", value)

    def _require_value(self, name: str, expected: Any) -> None:
        if not hasattr(self.cfg, name):
            raise ValueError(f"LLaDAModelLM config is missing required field '{name}'")
        actual = self._enum_value(getattr(self.cfg, name))
        if actual != expected:
            raise ValueError(f"LLaDAModelLM requires {name}={expected!r}; got {actual!r}")

    def _validate_config(self) -> None:
        self._require_value("block_type", "llama")
        self._require_value("block_group_size", 1)
        self._require_value("rope", True)
        self._require_value("rope_full_precision", True)
        self._require_value("alibi", False)
        self._require_value("attention_layer_norm", False)
        self._require_value("include_bias", False)
        self._require_value("include_qkv_bias", False)
        self._require_value("scale_logits", False)
        self._require_value("input_emb_norm", False)
        self._require_value("layer_norm_type", "rms")
        if self.cfg.act_fn != "silu":
            raise ValueError(
                f"LLaDAModelLM requires activation_type='silu'; got {self.cfg.act_fn!r}"
            )
        if bool(self.cfg.tie_word_embeddings):
            raise ValueError(
                "LLaDAModelLM tied embeddings are not supported by the initial dense adapter"
            )
        embedding_size = int(getattr(self.cfg, "embedding_size", self.cfg.d_vocab))
        if embedding_size != self.cfg.d_vocab:
            raise ValueError(
                "LLaDAModelLM requires embedding_size == vocab_size in the initial "
                f"dense adapter; got {embedding_size} and {self.cfg.d_vocab}"
            )

    def prepare_loading(self, model_name: str, model_kwargs: dict) -> None:
        """Disable remote branches incompatible with single-pass hook support."""
        del model_name
        config = model_kwargs.get("config")
        if config is not None:
            config.output_attentions = False
            config.use_cache = False

    def prepare_model(self, hf_model: Any) -> None:
        """Keep the wrapper and underlying model on the no-cache path."""
        for config in (
            getattr(hf_model, "config", None),
            getattr(getattr(hf_model, "model", None), "config", None),
        ):
            if config is not None:
                config.output_attentions = False
                config.use_cache = False
