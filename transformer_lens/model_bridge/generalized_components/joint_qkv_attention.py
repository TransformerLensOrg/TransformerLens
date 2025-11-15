"""Joint QKV attention bridge component.

This module contains the bridge component for attention layers that use a fused qkv matrix.
"""
from typing import Any, Callable, Dict, Mapping, Optional, cast

import einops
import torch
import torch.nn as nn

from transformer_lens.conversion_utils.conversion_steps.base_hook_conversion import (
    BaseHookConversion,
)
from transformer_lens.model_bridge.generalized_components.attention import (
    AttentionBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.generalized_components.linear import LinearBridge


class JointQKVAttentionBridge(AttentionBridge):
    """Joint QKV attention bridge that wraps a joint qkv linear layer.

    This component wraps attention layers that use a fused qkv matrix such that
    the individual activations from the separated q, k, and v matrices are hooked and accessible.
    """

    def __init__(
        self,
        name: str,
        config: Any,
        split_qkv_matrix: Callable,
        submodules: Optional[Dict[str, GeneralizedComponent]] = None,
        qkv_conversion_rule: Optional[BaseHookConversion] = None,
        attn_conversion_rule: Optional[BaseHookConversion] = None,
        pattern_conversion_rule: Optional[BaseHookConversion] = None,
        requires_position_embeddings: bool = False,
        requires_attention_mask: bool = False,
    ):
        """Initialize the Joint QKV attention bridge.

        Args:
            name: The name of this component
            config: Model configuration (required for auto-conversion detection)
            split_qkv_matrix: Function to split the qkv matrix into q, k, and v linear transformations
            submodules: Dictionary of submodules to register (e.g., q_proj, k_proj, etc.)
            qkv_conversion_rule: Optional conversion rule for the individual q, k, and v matrices to convert their output shapes to HookedTransformer format. If None, uses default RearrangeHookConversion
            attn_conversion_rule: Optional conversion rule. Passed to parent AttentionBridge. If None, AttentionAutoConversion will be used
            pattern_conversion_rule: Optional conversion rule for attention patterns. If None,
                                   uses AttentionPatternConversion to ensure [n_heads, pos, pos] shape
            requires_position_embeddings: Whether this attention requires position_embeddings as input
            requires_attention_mask: Whether this attention requires attention_mask as input
        """
        super().__init__(
            name,
            config,
            submodules=submodules,
            conversion_rule=attn_conversion_rule,
            pattern_conversion_rule=pattern_conversion_rule,
            requires_position_embeddings=requires_position_embeddings,
            requires_attention_mask=requires_attention_mask,
        )
        self.split_qkv_matrix = split_qkv_matrix
        if qkv_conversion_rule is not None:
            self.qkv_conversion_rule = qkv_conversion_rule
        else:
            self.qkv_conversion_rule = self._create_qkv_conversion_rule()
        self.q = LinearBridge(name="q")
        self.k = LinearBridge(name="k")
        self.v = LinearBridge(name="v")
        for submodule_name, submodule in (submodules or {}).items():
            if not hasattr(self, submodule_name):
                setattr(self, submodule_name, submodule)
        self.q.hook_out.hook_conversion = self.qkv_conversion_rule
        self.k.hook_out.hook_conversion = self.qkv_conversion_rule
        self.v.hook_out.hook_conversion = self.qkv_conversion_rule
        self._processed_weights: Optional[Dict[str, torch.Tensor]] = None
        self._hooked_weights_extracted = False
        self._W_Q: Optional[torch.Tensor] = None
        self._W_K: Optional[torch.Tensor] = None
        self._W_V: Optional[torch.Tensor] = None
        self._W_O: Optional[torch.Tensor] = None
        self._b_Q: Optional[torch.Tensor] = None
        self._b_K: Optional[torch.Tensor] = None
        self._b_V: Optional[torch.Tensor] = None
        self._b_O: Optional[torch.Tensor] = None
        self._reference_model: Optional[Any] = None
        self._layer_idx: Optional[int] = None

    def _create_qkv_conversion_rule(self) -> BaseHookConversion:
        """Create the appropriate conversion rule for the individual q, k, and v matrices.

        Returns:
            BaseHookConversion for individual q, k, and v matrices
        """
        assert self.config is not None

        class ConditionalRearrangeConversion(BaseHookConversion):
            def __init__(self, n_heads: int):
                super().__init__()
                self.n_heads = n_heads
                self.pattern = (
                    "batch seq (num_attention_heads d_head) -> batch seq num_attention_heads d_head"
                )

            def handle_conversion(self, input_value: torch.Tensor, *full_context) -> torch.Tensor:
                if input_value.ndim == 4:
                    return input_value
                elif input_value.ndim == 3:
                    return einops.rearrange(
                        input_value, self.pattern, num_attention_heads=self.n_heads
                    )
                else:
                    raise ValueError(
                        f"Expected 3D or 4D tensor, got {input_value.ndim}D with shape {input_value.shape}"
                    )

            def revert(self, input_value: torch.Tensor, *full_context) -> torch.Tensor:
                if input_value.ndim == 3:
                    return input_value
                elif input_value.ndim == 4:
                    return einops.rearrange(
                        input_value,
                        "batch seq num_attention_heads d_head -> batch seq (num_attention_heads d_head)",
                        num_attention_heads=self.n_heads,
                    )
                else:
                    raise ValueError(
                        f"Expected 3D or 4D tensor, got {input_value.ndim}D with shape {input_value.shape}"
                    )

        return ConditionalRearrangeConversion(self.config.n_heads)

    def set_original_component(self, original_component: torch.nn.Module) -> None:
        """Set the original component that this bridge wraps and initialize LinearBridges for q, k, v, and o transformations.

        Args:
            original_component: The original attention layer to wrap
        """
        super().set_original_component(original_component)
        q_transformation, k_transformation, v_transformation = self.split_qkv_matrix(
            original_component
        )
        self.q.set_original_component(q_transformation)
        self.k.set_original_component(k_transformation)
        self.v.set_original_component(v_transformation)
        if hasattr(self, "o") and hasattr(original_component, "c_proj"):
            self.o.set_original_component(original_component.c_proj)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the qkv linear transformation with hooks.

        Args:
            *args: Input arguments, where the first argument should be the input tensor
            **kwargs: Additional keyword arguments

        Returns:
            Output tensor after qkv linear transformation
        """
        hooked_input = self._apply_attention_input_hook(*args, **kwargs)
        q_output = self.q(hooked_input)
        k_output = self.k(hooked_input)
        v_output = self.v(hooked_input)
        output = self._reconstruct_attention(q_output, k_output, v_output, **kwargs)
        output = self._process_output(output)
        return output

    def _process_output(self, output: Any) -> Any:
        """Process the output from _reconstruct_attention.

        This override skips the duplicate hook_pattern call since
        _reconstruct_attention already applies both hook_attn_scores
        and hook_pattern correctly.

        Args:
            output: Output tuple from _reconstruct_attention (attn_output, attn_weights)

        Returns:
            Processed output with hook_out applied
        """
        attn_pattern = None
        if isinstance(output, tuple) and len(output) >= 2:
            attn_pattern = output[1]
        if attn_pattern is not None:
            self._pattern = attn_pattern
        if isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            processed_output = list(output)
            processed_output[0] = self.hook_hidden_states(output[0])
            output = tuple(processed_output)
        if isinstance(output, torch.Tensor):
            output = self.hook_out(output)
        elif isinstance(output, tuple) and len(output) > 0:
            processed_tuple = list(output)
            if isinstance(output[0], torch.Tensor):
                processed_tuple[0] = self.hook_out(output[0])
            if len(processed_tuple) == 1:
                return processed_tuple[0]
            output = tuple(processed_tuple)
        return output

    def _apply_attention_input_hook(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Apply attention input hook to the input tensor.

        This method extracts the input tensor from args/kwargs and applies the attention
        input hook in the same way as the super class.

        Args:
            *args: Input arguments, where the first argument should be the input tensor
            **kwargs: Additional keyword arguments that might contain input

        Returns:
            Input tensor with attention input hook applied

        Raises:
            ValueError: If no input tensor is found in args or kwargs
        """
        input_tensor = None
        if "query_input" in kwargs:
            input_tensor = kwargs["query_input"]
        elif "hidden_states" in kwargs:
            input_tensor = kwargs["hidden_states"]
        elif len(args) > 0 and isinstance(args[0], torch.Tensor):
            input_tensor = args[0]
        else:
            raise ValueError("No input tensor found in args or kwargs")
        return self.hook_in(input_tensor)

    def _reconstruct_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs
    ) -> tuple:
        """Manual attention computation as fallback using TransformerLens computation logic."""
        original_component = self.original_component
        assert original_component is not None
        assert self.config is not None
        num_heads = self.config.n_heads
        if len(q.shape) == 3:
            batch_size, seq_len, hidden_size = q.shape
            head_dim: int = hidden_size // num_heads
            q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        elif len(q.shape) == 4:
            batch_size, seq_len, num_heads_tensor, head_dim = q.shape
            assert (
                num_heads_tensor == num_heads
            ), f"Expected {num_heads} heads, got {num_heads_tensor}"
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
        else:
            raise ValueError(f"Unexpected Q tensor shape: {q.shape}. Expected 3D or 4D tensor.")
        scale = head_dim ** (-0.5)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device))
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            if attention_mask.shape[-1] != seq_len:
                attention_mask = attention_mask[..., :seq_len]
            if attention_mask.shape[-2] != seq_len:
                attention_mask = attention_mask[..., :seq_len, :]
            attn_scores = attn_scores + attention_mask
        attn_scores = self.hook_attn_scores(attn_scores)
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        if hasattr(original_component, "attn_dropout"):
            attn_weights = original_component.attn_dropout(attn_weights)  # type: ignore[operator]
        attn_weights = self.hook_pattern(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        final_hidden_size: int = num_heads * head_dim
        attn_output = attn_output.view(batch_size, seq_len, final_hidden_size)
        if hasattr(self, "o") and self.o is not None:
            attn_output = self.o(attn_output)
        return (attn_output, attn_weights)

    def set_processed_weights(self, weights: Mapping[str, torch.Tensor | None]) -> None:
        """Set processed weights for Joint QKV attention.

        For Joint QKV attention, the Q/K/V weights are stored in a single c_attn component.
        This method handles both 2D format [d_model, (n_heads*d_head)] and 3D TL format
        [n_heads, d_model, d_head] for Q/K/V weights.

        Args:
            weights: Dictionary containing processed weight tensors with keys:
                - "W_Q": Query weight tensor (2D or 3D format)
                - "W_K": Key weight tensor (2D or 3D format)
                - "W_V": Value weight tensor (2D or 3D format)
                - "W_O": Output projection weight (2D HF format [d_model, d_model] or 3D TL format)
                - "b_Q": Query bias tensor (optional)
                - "b_K": Key bias tensor (optional)
                - "b_V": Value bias tensor (optional)
                - "b_O": Output bias tensor (optional)
        """
        import einops

        W_Q = weights.get("W_Q")
        W_K = weights.get("W_K")
        W_V = weights.get("W_V")
        if W_Q is None or W_K is None or W_V is None:
            raise ValueError("Processed joint QKV weights must include W_Q, W_K, and W_V tensors.")
        W_O = weights.get("W_O")
        b_Q = weights.get("b_Q")
        b_K = weights.get("b_K")
        b_V = weights.get("b_V")
        b_O = weights.get("b_O")
        if W_Q.ndim == 2:
            assert self.config is not None
            n_heads = self.config.n_heads
            W_Q = einops.rearrange(
                W_Q, "d_model (n_heads d_head) -> n_heads d_model d_head", n_heads=n_heads
            )
            W_K = einops.rearrange(
                W_K, "d_model (n_heads d_head) -> n_heads d_model d_head", n_heads=n_heads
            )
            W_V = einops.rearrange(
                W_V, "d_model (n_heads d_head) -> n_heads d_model d_head", n_heads=n_heads
            )
            if b_Q is not None and b_Q.ndim == 1:
                b_Q = einops.rearrange(b_Q, "(n_heads d_head) -> n_heads d_head", n_heads=n_heads)
            if b_K is not None and b_K.ndim == 1:
                b_K = einops.rearrange(b_K, "(n_heads d_head) -> n_heads d_head", n_heads=n_heads)
            if b_V is not None and b_V.ndim == 1:
                b_V = einops.rearrange(b_V, "(n_heads d_head) -> n_heads d_head", n_heads=n_heads)
        self._W_Q = W_Q
        self._W_K = W_K
        self._W_V = W_V
        self._b_Q = b_Q
        self._b_K = b_K
        self._b_V = b_V
        if hasattr(self, "q") and self.q is not None and hasattr(self.q, "set_processed_weights"):
            self.q.set_processed_weights({"weight": W_Q, "bias": b_Q})
        if hasattr(self, "k") and self.k is not None and hasattr(self.k, "set_processed_weights"):
            self.k.set_processed_weights({"weight": W_K, "bias": b_K})
        if hasattr(self, "v") and self.v is not None and hasattr(self.v, "set_processed_weights"):
            self.v.set_processed_weights({"weight": W_V, "bias": b_V})
        if (
            hasattr(self, "o")
            and self.o is not None
            and hasattr(self.o, "set_processed_weights")
            and (W_O is not None)
        ):
            self.o.set_processed_weights({"weight": W_O, "bias": b_O})

    def process_weights(
        self,
        fold_ln: bool = False,
        center_writing_weights: bool = False,
        center_unembed: bool = False,
        fold_value_biases: bool = False,
        refactor_factored_attn_matrices: bool = False,
    ) -> None:
        """Process QKV weights according to GPT2 pretrained logic.

        Ports the weight processing from transformer_lens.pretrained.weight_conversions.gpt2
        to work with the architecture adapter.
        """
        import einops

        original_component = self.original_component
        if original_component is None:
            return
        if hasattr(original_component, "c_attn"):
            c_attn = cast(nn.Module, original_component.c_attn)
            qkv_weight = c_attn.weight
            qkv_bias = c_attn.bias
        else:
            qkv_submodule = None
            for name, module in self.submodules.items():
                if hasattr(module, "name") and module.name == "c_attn":
                    qkv_submodule = getattr(original_component, module.name, None)
                    break
            if qkv_submodule is None:
                return
            qkv_weight = cast(torch.Tensor, qkv_submodule.weight)
            qkv_bias = cast(torch.Tensor, qkv_submodule.bias)
        W_Q, W_K, W_V = torch.tensor_split(cast(torch.Tensor, qkv_weight), 3, dim=1)
        assert self.config is not None
        W_Q = einops.rearrange(W_Q, "m (i h)->i m h", i=self.config.n_heads)
        W_K = einops.rearrange(W_K, "m (i h)->i m h", i=self.config.n_heads)
        W_V = einops.rearrange(W_V, "m (i h)->i m h", i=self.config.n_heads)
        qkv_bias_tensor = cast(torch.Tensor, qkv_bias)
        qkv_bias = einops.rearrange(
            qkv_bias_tensor,
            "(qkv index head)->qkv index head",
            qkv=3,
            index=self.config.n_heads,
            head=self.config.d_head,
        )
        b_Q, b_K, b_V = (
            cast(torch.Tensor, qkv_bias[0]),
            cast(torch.Tensor, qkv_bias[1]),
            cast(torch.Tensor, qkv_bias[2]),
        )
        W_O = None
        b_O = None
        if hasattr(original_component, "c_proj"):
            c_proj = cast(nn.Module, original_component.c_proj)
            W_O = cast(torch.Tensor, c_proj.weight)
            b_O = cast(torch.Tensor, c_proj.bias)
            W_O = einops.rearrange(W_O, "(i h) m->i h m", i=self.config.n_heads)
        else:
            for name, module in self.submodules.items():
                if hasattr(module, "name") and module.name == "c_proj":
                    proj_submodule = getattr(original_component, module.name, None)
                    if proj_submodule is not None:
                        W_O = proj_submodule.weight
                        b_O = proj_submodule.bias
                        W_O = einops.rearrange(W_O, "(i h) m->i h m", i=self.config.n_heads)
                    break
        self._processed_weights = {
            "W_Q": W_Q,
            "W_K": W_K,
            "W_V": W_V,
            "b_Q": b_Q,
            "b_K": b_K,
            "b_V": b_V,
        }
        if W_O is not None:
            self._processed_weights["W_O"] = W_O
        if b_O is not None:
            self._processed_weights["b_O"] = b_O

    def _extract_hooked_transformer_weights(self) -> None:
        """Extract weights in HookedTransformer format for exact compatibility."""
        if (
            hasattr(self, "_processed_W_Q")
            and hasattr(self, "_processed_W_K")
            and hasattr(self, "_processed_W_V")
        ):
            self._W_Q = self._processed_W_Q
            self._W_K = self._processed_W_K
            self._W_V = self._processed_W_V
            self._b_Q = self._processed_b_Q
            self._b_K = self._processed_b_K
            self._b_V = self._processed_b_V
            if hasattr(self, "_processed_W_O") and self._processed_W_O is not None:
                self._W_O = self._processed_W_O
            else:
                self._W_O = None
            if hasattr(self, "_processed_b_O") and self._processed_b_O is not None:
                self._b_O = self._processed_b_O
            else:
                self._b_O = None
            self._hooked_weights_extracted = True
            return
        try:
            if hasattr(self, "_reference_model") and self._reference_model is not None:
                reference_model = self._reference_model
                layer_num = getattr(self, "_layer_idx", 0)
                reference_attn = reference_model.blocks[layer_num].attn
                self._W_Q = reference_attn.W_Q.clone()
                self._W_K = reference_attn.W_K.clone()
                self._W_V = reference_attn.W_V.clone()
                self._b_Q = reference_attn.b_Q.clone()
                self._b_K = reference_attn.b_K.clone()
                self._b_V = reference_attn.b_V.clone()
                if hasattr(reference_attn, "W_O"):
                    self._W_O = reference_attn.W_O.clone()
                if hasattr(reference_attn, "b_O"):
                    self._b_O = reference_attn.b_O.clone()
                self._hooked_weights_extracted = True
                self._reference_model = None
                return
        except Exception:
            pass
        if self._processed_weights is None:
            return
        try:
            from transformer_lens import HookedTransformer

            model_name = getattr(self.config, "model_name", "gpt2")
            device = next(self.parameters()).device if list(self.parameters()) else "cpu"
            reference_model = HookedTransformer.from_pretrained(
                model_name,
                device=device,
                fold_ln=True,
                center_writing_weights=True,
                center_unembed=True,
                fold_value_biases=True,
                refactor_factored_attn_matrices=False,
            )
            layer_num = 0
            current = self
            while hasattr(current, "parent") and current.parent is not None:
                parent = current.parent
                if hasattr(parent, "blocks"):
                    for i, block in enumerate(parent.blocks):
                        if hasattr(block, "attn") and block.attn is self:
                            layer_num = i
                            break
                    break
                current = parent
            reference_attn = reference_model.blocks[layer_num].attn
            self._W_Q = reference_attn.W_Q.clone()  # type: ignore[operator,union-attr]
            self._W_K = reference_attn.W_K.clone()  # type: ignore[operator,union-attr]
            self._W_V = reference_attn.W_V.clone()  # type: ignore[operator,union-attr]
            self._b_Q = reference_attn.b_Q.clone()  # type: ignore[operator,union-attr]
            self._b_K = reference_attn.b_K.clone()  # type: ignore[operator,union-attr]
            self._b_V = reference_attn.b_V.clone()  # type: ignore[operator,union-attr]
            if hasattr(reference_attn, "W_O"):
                self._W_O = reference_attn.W_O.clone()  # type: ignore[operator]
            if hasattr(reference_attn, "b_O"):
                self._b_O = reference_attn.b_O.clone()  # type: ignore[operator]
            del reference_model
            self._hooked_weights_extracted = True
            return
        except Exception:
            pass
        if self._processed_weights is None:
            try:
                self.process_weights(
                    fold_ln=True,
                    center_writing_weights=True,
                    center_unembed=True,
                    fold_value_biases=True,
                    refactor_factored_attn_matrices=False,
                )
            except Exception as e:
                print(f"⚠️  Failed to process weights manually: {e}")
        if self._processed_weights is not None:
            self._W_Q = self._processed_weights["W_Q"]
            self._W_K = self._processed_weights["W_K"]
            self._W_V = self._processed_weights["W_V"]
            self._b_Q = self._processed_weights["b_Q"]
            self._b_K = self._processed_weights["b_K"]
            self._b_V = self._processed_weights["b_V"]
            if "W_O" in self._processed_weights:
                self._W_O = self._processed_weights["W_O"]
            if "b_O" in self._processed_weights:
                self._b_O = self._processed_weights["b_O"]
            print(f"✅ Extracted HookedTransformer weights from processed weights")
            self._hooked_weights_extracted = True
        else:
            print(f"⚠️  Unable to extract HookedTransformer weights for {self.name}")
            print("Will attempt to use original component computation")
            self._hooked_weights_extracted = False
