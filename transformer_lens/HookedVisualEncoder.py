"""Hooked Encoder.

Contains a BERT style model. This is separate from :class:`transformer_lens.HookedTransformer`
because it has a significantly different architecture to e.g. GPT style transformers.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, cast, overload, Sequence

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from einops import repeat
from jaxtyping import Float, Int
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from typing_extensions import Literal

import transformer_lens.loading_from_pretrained as loading
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.components import (
    MLP,
    TransformerBlock,
    ClassifierHead,
    LayerNorm,
)
from transformer_lens.components.mlps.gated_mlp import GatedMLP
from transformer_lens.config.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.FactoredMatrix import FactoredMatrix
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.utilities import devices

T = TypeVar("T", bound="HookedVisualEncoder")

ImageInput = Union[Image.Image, np.ndarray, list[Image.Image], list[np.ndarray]]


class HookedVisualEncoder(HookedRootModule):
    """
    This class implements a ViT-style encoder using the components in ./components.py, with HookPoints on every interesting activation. It inherits from HookedRootModule.

    Limitations:
    - The model does not include dropouts, which may lead to inconsistent results from training or fine-tuning.

    Like HookedTransformer, it can have a pretrained Transformer's weights loaded via `.from_pretrained`. There are a few features you might know from HookedTransformer which are not yet supported:
        - There is no preprocessing (e.g. LayerNorm folding) when loading a pretrained model
    """

    blocks: nn.ModuleList[TransformerBlock]  # type: ignore[type-arg]

    def _get_blocks(self) -> list[TransformerBlock]:
        """Helper to get blocks with proper typing."""
        return [cast(TransformerBlock, block) for block in self.blocks]

    def __init__(
        self,
        cfg: Union[HookedTransformerConfig, Dict],
        move_to_device: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig(**cfg)
        elif isinstance(cfg, str):
            raise ValueError(
                "Please pass in a config dictionary or HookedTransformerConfig object. If you want to load a pretrained model, use HookedVisualEncoder.from_pretrained() instead."
            )
        self.cfg = cfg

        assert self.cfg.n_devices == 1, "Multiple devices not supported for HookedVisualEncoder"
        

        self.blocks = nn.ModuleList([TransformerBlock(self.cfg, block_index) for block_index in range(self.cfg.n_layers)])
        self.layernorm = LayerNorm(self.cfg)
        self.classifier = ClassifierHead(1000, self.cfg)

        if move_to_device:
            if self.cfg.device is None:
                raise ValueError("Cannot move to device when device is None")
            self.to(self.cfg.device)

        self.setup()

    def encoder_output(
        self,
        resid: torch.Tensor,  # (B, seq_len, d_model) = embedding_output
    ):
        """
        ViT encoder: stack of transformer blocks.
        Args:
            resid: embedding output (patch + CLS + pos embedding)
        Returns:
            resid: (B, seq_len, d_model)
        """
    
        # device safety (TL style)
        if resid.device.type != self.cfg.device:
            resid = resid.to(self.cfg.device)
    
        # No attention mask in standard ViT
        for block in self.blocks:
            resid = block(resid)
    
        return resid

    def _expected_hw(self) -> tuple[int, int]:
        image_size = 224
        if isinstance(image_size, int):
            return image_size, image_size
        if len(image_size) == 2:
            return int(image_size[0]), int(image_size[1])
        raise ValueError(f"Unsupported config.image_size={image_size}")

    def forward(
        self,
        inputs: torch.Tensor | ImageInput,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        strict_224: bool = False,
        return_type: Optional[str] = "logits",
    ):
        expected_h, expected_w = self._expected_hw()

        # Raw images -> preprocess
        if isinstance(inputs, (Image.Image, np.ndarray, list)):
            processed = self.feature_extractor(images=inputs, return_tensors="pt")
            pixel_values = processed["pixel_values"]
            # default processor should already resize to expected size
            if interpolate_pos_encoding is None:
                interpolate_pos_encoding = False

        # Precomputed pixel values -> inspect shape
        elif isinstance(inputs, torch.Tensor):
            pixel_values = inputs
            if pixel_values.ndim == 3:
                pixel_values = pixel_values.unsqueeze(0)
            if pixel_values.ndim != 4:
                raise ValueError(
                    f"Expected pixel_values with shape (B, C, H, W) or (C, H, W), got {tuple(pixel_values.shape)}"
                )

            _, _, h, w = pixel_values.shape
            matches_expected = (h == expected_h) and (w == expected_w)

            if interpolate_pos_encoding is None:
                interpolate_pos_encoding = not matches_expected

            if strict_224 and not matches_expected:
                raise ValueError(
                    f"Only {expected_h}x{expected_w} pixel_values are allowed, got {h}x{w}."
                )
        else:
            raise TypeError(f"Unsupported input type: {type(inputs)!r}")

        embedding_output = self.embeddings(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        encoder_outputs = self.encoder_output(embedding_output)
        sequence_output = self.layernorm(encoder_outputs)
        cls_token = sequence_output[:, 0, :]
        logits = self.classifier(cls_token)

        if return_type is None:
            return None
        return logits

    @overload
    def run_with_cache(
        self, *model_args: Any, return_cache_object: Literal[True] = True, **kwargs: Any
    ) -> Tuple[Float[torch.Tensor, "batch pos d_vocab"], ActivationCache]:
        ...

    @overload
    def run_with_cache(
        self, *model_args: Any, return_cache_object: Literal[False], **kwargs: Any
    ) -> Tuple[Float[torch.Tensor, "batch pos d_vocab"], Dict[str, torch.Tensor]]:
        ...

    def run_with_cache(
        self,
        *model_args: Any,
        return_cache_object: bool = True,
        remove_batch_dim: bool = False,
        **kwargs: Any,
    ) -> Tuple[
        Float[torch.Tensor, "batch pos d_vocab"],
        Union[ActivationCache, Dict[str, torch.Tensor]],
    ]:
        """
        Wrapper around run_with_cache in HookedRootModule. If return_cache_object is True, this will return an ActivationCache object, with a bunch of useful HookedTransformer specific methods, otherwise it will return a dictionary of activations as in HookedRootModule. This function was copied directly from HookedTransformer.
        """
        out, cache_dict = super().run_with_cache(
            *model_args, remove_batch_dim=remove_batch_dim, **kwargs
        )
        if return_cache_object:
            cache = ActivationCache(cache_dict, self, has_batch_dim=not remove_batch_dim)
            return out, cache
        else:
            return out, cache_dict

    def to(  # type: ignore
        self,
        device_or_dtype: Union[torch.device, str, torch.dtype],
        print_details: bool = True,
    ):
        return devices.move_to_and_update_config(self, device_or_dtype, print_details)

    def cuda(self: T, device: Optional[Union[int, torch.device]] = None) -> T:
        if isinstance(device, int):
            return self.to(f"cuda:{device}")
        elif device is None:
            return self.to("cuda")
        else:
            return self.to(device)

    def cpu(self: T) -> T:
        return self.to("cpu")

    def mps(self: T) -> T:
        """Warning: MPS may produce silently incorrect results. See #1178."""
        return self.to(torch.device("mps"))

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        checkpoint_index: Optional[int] = None,
        checkpoint_value: Optional[int] = None,
        hf_model: Optional[Any] = None,
        device: Optional[str] = None,
        move_to_device: bool = True,
        dtype: torch.dtype = torch.float32,
        **from_pretrained_kwargs: Any,
    ) -> HookedVisualEncoder:
        """Loads in the pretrained weights from huggingface. Currently supports loading weight from HuggingFace ViTForImageClassification. Unlike HookedTransformer, this does not yet do any preprocessing on the model."""
        logging.warning(
            "Support for ViT in TransformerLens is currently experimental, until such a time when it has feature "
            "parity with HookedTransformer and has been tested on real research tasks. Until then, backward "
            "compatibility is not guaranteed. Please see the docs for information on the limitations of the current "
            "implementation."
            "\n"
        )

        assert not (
            from_pretrained_kwargs.get("load_in_8bit", False)
            or from_pretrained_kwargs.get("load_in_4bit", False)
        ), "Quantization not supported"

        if "torch_dtype" in from_pretrained_kwargs:
            dtype = from_pretrained_kwargs["torch_dtype"]

        official_model_name = loading.get_official_model_name(model_name)

        cfg = loading.get_pretrained_model_config(
            official_model_name,
            checkpoint_index=checkpoint_index,
            checkpoint_value=checkpoint_value,
            fold_ln=False,
            device=device,
            n_devices=1,
            dtype=dtype,
            **from_pretrained_kwargs,
        )

        state_dict = loading.get_pretrained_state_dict(
            official_model_name, cfg, hf_model, dtype=dtype, **from_pretrained_kwargs
        )

        model = cls(cfg, move_to_device=False)

        model.load_state_dict(state_dict, strict=False)

        model.feature_extractor = AutoImageProcessor.from_pretrained(official_model_name)

        visual_model = AutoModelForImageClassification.from_pretrained(official_model_name)

        visual_model.eval()
        if "vit" in official_model_name:
            model.embeddings = visual_model.vit.embeddings
        elif "dit" in official_model_name:
            model.embeddings = visual_model.deit.embeddings
        
        del visual_model

        if move_to_device:
            if cfg.device is not None:
                model.to(cfg.device)

        print(f"Loaded pretrained model {model_name} into HookedVisualEncoder")

        return model

    @property
    def W_C(self) -> Float[torch.Tensor, "d_model num_labels"]:
        return self.classifier.W
    
    
    @property
    def b_C(self) -> Float[torch.Tensor, "num_labels"]:
        return self.classifier.b

    @property
    def W_K(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the key weights across all layers"""
        return torch.stack([block.attn.W_K for block in self._get_blocks()], dim=0)

    @property
    def W_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the query weights across all layers"""
        return torch.stack([block.attn.W_Q for block in self._get_blocks()], dim=0)

    @property
    def W_V(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the value weights across all layers"""
        return torch.stack([block.attn.W_V for block in self._get_blocks()], dim=0)

    @property
    def W_O(self) -> Float[torch.Tensor, "n_layers n_heads d_head d_model"]:
        """Stacks the attn output weights across all layers"""
        return torch.stack([block.attn.W_O for block in self._get_blocks()], dim=0)

    @property
    def W_in(self) -> Float[torch.Tensor, "n_layers d_model d_mlp"]:
        """Stacks the MLP input weights across all layers"""
        return torch.stack(
            [cast(Union[MLP, GatedMLP], block.mlp).W_in for block in self._get_blocks()], dim=0
        )

    @property
    def W_out(self) -> Float[torch.Tensor, "n_layers d_mlp d_model"]:
        """Stacks the MLP output weights across all layers"""
        return torch.stack(
            [cast(Union[MLP, GatedMLP], block.mlp).W_out for block in self._get_blocks()], dim=0
        )

    @property
    def b_K(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the key biases across all layers"""
        return torch.stack([block.attn.b_K for block in self._get_blocks()], dim=0)

    @property
    def b_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the query biases across all layers"""
        return torch.stack([block.attn.b_Q for block in self._get_blocks()], dim=0)

    @property
    def b_V(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the value biases across all layers"""
        return torch.stack([block.attn.b_V for block in self._get_blocks()], dim=0)

    @property
    def b_O(self) -> Float[torch.Tensor, "n_layers d_model"]:
        """Stacks the attn output biases across all layers"""
        return torch.stack([block.attn.b_O for block in self._get_blocks()], dim=0)

    @property
    def b_in(self) -> Float[torch.Tensor, "n_layers d_mlp"]:
        """Stacks the MLP input biases across all layers"""
        return torch.stack(
            [cast(Union[MLP, GatedMLP], block.mlp).b_in for block in self._get_blocks()], dim=0
        )

    @property
    def b_out(self) -> Float[torch.Tensor, "n_layers d_model"]:
        """Stacks the MLP output biases across all layers"""
        return torch.stack(
            [cast(Union[MLP, GatedMLP], block.mlp).b_out for block in self._get_blocks()], dim=0
        )

    @property
    def QK(self) -> FactoredMatrix:  # [n_layers, n_heads, d_model, d_model]
        """Returns a FactoredMatrix object with the product of the Q and K matrices for each layer and head.
        Useful for visualizing attention patterns."""
        return FactoredMatrix(self.W_Q, self.W_K.transpose(-2, -1))

    @property
    def OV(self) -> FactoredMatrix:  # [n_layers, n_heads, d_model, d_model]
        """Returns a FactoredMatrix object with the product of the O and V matrices for each layer and head."""
        return FactoredMatrix(self.W_V, self.W_O)

    def all_head_labels(self) -> List[str]:
        """Returns a list of strings with the format "L{l}H{h}", where l is the layer index and h is the head index."""
        return [f"L{l}H{h}" for l in range(self.cfg.n_layers) for h in range(self.cfg.n_heads)]
