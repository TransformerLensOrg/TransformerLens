"""Hooked Audio Encoder.

Contains a HuBERT style model. This is separate from :class:`transformer_lens.HookedTransformer`
because it has a significantly different architecture to e.g. GPT style transformers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, overload

import numpy as np
import torch
import torch.nn as nn
from einops import repeat
from jaxtyping import Float, Int
from transformers import (
    AutoFeatureExtractor,
    AutoProcessor,
    HubertForCTC,
    HubertModel,
    Wav2Vec2Model,
)
from typing_extensions import Literal

from transformer_lens import loading_from_pretrained as loading
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.components import MLP, Attention, BertBlock
from transformer_lens.FactoredMatrix import FactoredMatrix
from transformer_lens.hook_points import HookedRootModule
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.utilities import devices

T = TypeVar("T", bound="HookedEncoder")


class HookedAudioEncoder(HookedRootModule):
    """
    This class implements a BERT-style encoder using the components in ./components.py, with HookPoints on every interesting activation. It inherits from HookedRootModule.

    Limitations:
    - The model does not include dropouts, which may lead to inconsistent results from training or fine-tuning.

    Like HookedTransformer, it can have a pretrained Transformer's weights loaded via `.from_pretrained`. There are a few features you might know from HookedTransformer which are not yet supported:
        - There is no preprocessing (e.g. LayerNorm folding) when loading a pretrained model
    """

    def __init__(
        self,
        cfg: Union[HookedTransformerConfig, Dict],
        move_to_device: bool = True,
        model_name: str = "facebook/hubert-base-ls960",
        **kwargs: Any,
    ):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig(**cfg)
        elif isinstance(cfg, str):
            raise ValueError(
                "Please pass in a config dictionary or HookedTransformerConfig object. If you want to load a pretrained model, use HookedAudioEncoder.from_pretrained() instead."
            )
        self.cfg = cfg

        assert self.cfg.n_devices == 1, "Multiple devices not supported for HookedEncoder"

        self.blocks = nn.ModuleList([BertBlock(self.cfg) for _ in range(self.cfg.n_layers)])

        if move_to_device:
            if self.cfg.device is None:
                raise ValueError("Cannot move to device when device is None")
            self.to(self.cfg.device)

        self.setup()

    def _ensure_numpy(self, wave):
        """
        Convert torch.Tensor / np.ndarray / list -> 1D np.float32 array on CPU.
        """
        if isinstance(wave, torch.Tensor):
            arr = wave.detach().cpu().numpy()
        elif isinstance(wave, np.ndarray):
            arr = wave
        elif isinstance(wave, list):
            arr = np.asarray(wave)
        else:
            raise TypeError("wave must be torch.Tensor, np.ndarray or list of floats")

        # force 1-D (if stereo or shape (N,1) etc)
        if arr.ndim > 1:
            # if shape (n_samples, n_channels) average channels -> mono
            if arr.shape[1] <= arr.shape[0]:
                arr = arr.mean(axis=1)
            else:
                arr = arr.reshape(-1)

        return arr.astype(np.float32, copy=False)

    def to_frames(
        self,
        raw_inputs: Union[torch.Tensor, List[torch.Tensor], List[np.ndarray]],
        sampling_rate: int = 16000,
        move_to_device: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert raw audio batch -> (projected frames, frame_attention_mask)

        Args:
            raw_inputs: one of:
                - a 1D torch.Tensor or numpy array (single waveform)
                - a list of 1D torch.Tensors / numpy arrays (batch)
            self.processor: HF AutoProcessor (creates input_values + sample-level attention_mask)
            self.model: pretrained HubertModel (provides feature_extractor and feature_projection)
            sampling_rate: sample rate of the audio (default 16k)
            move_to_device: move outputs to model.device

        Returns:
            frames: torch.Tensor of shape (batch, frames, hidden_size)  <- after feature_projection
            frame_attention_mask: torch.LongTensor of shape (batch, frames) with 1 for real frames, 0 for padding
        """
        # AutoFeatureExtractor works better onnumpy array where it pads automatically. If passing in tensors, it does not pad properly, giving inhomogeneous arts error
        if isinstance(raw_inputs, (torch.Tensor, np.ndarray)):
            waves = [self._ensure_numpy(raw_inputs)]
        elif isinstance(raw_inputs, list):
            waves = [self._ensure_numpy(w) for w in raw_inputs]
        else:
            raise TypeError("Unsupported raw_inputs type")

        # Use HF processor to create input_values (padded) + sample-level attention_mask
        # Processor will do padding so we can pass a variable-length batch
        proc_out = self.processor(
            waves,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        input_values = proc_out["input_values"]  # (batch, samples), float
        sample_attention_mask = proc_out.get(
            "attention_mask"
        )  # (batch, samples), 1 for valid, 0 for padding; may be None

        # move to device
        device = self.cfg.device
        if move_to_device:
            input_values = input_values.to(device)
            if sample_attention_mask is not None:
                sample_attention_mask = sample_attention_mask.to(device)

        # 1) convolutional frontend -> (batch, conv_dim, conv_time)
        if input_values.ndim > 2:
            input_values = input_values.squeeze()
            if input_values.ndim == 1:
                input_values = input_values.unsqueeze(0)  # (1, T)
        with torch.no_grad():
            conv_feats = self.hubert_model.feature_extractor(input_values)  # (B, C, T_conv)

        # 2) transpose to (batch, T_conv, C)
        extract_features = conv_feats.transpose(1, 2)

        # 3) compute reduced frame-level attention mask (if sample mask provided)
        frame_attention_mask = None
        if sample_attention_mask is not None:
            # model should provide helper _get_feature_vector_attention_mask
            try:
                frame_attention_mask = self.hubert_model._get_feature_vector_attention_mask(
                    extract_features.shape[1], sample_attention_mask
                )
            except AttributeError:
                # fallback: compute output lengths and create mask similarly to HF implementation
                # compute output lengths (downsampled lengths) from sample attention mask (sums per example)
                input_lengths = sample_attention_mask.sum(dim=-1)  # (batch,)
                # compute output lengths through conv layers using model._get_feat_extract_output_lengths if exists
                if hasattr(model, "_get_feat_extract_output_lengths"):
                    output_lengths = self.hubert_model._get_feat_extract_output_lengths(
                        input_lengths
                    ).to(torch.long)
                else:
                    # fallback to naive downsample ratio: output_frames = extract_features.shape[1]
                    output_lengths = torch.full(
                        (sample_attention_mask.shape[0],),
                        extract_features.shape[1],
                        device=device,
                        dtype=torch.long,
                    )

                batch_size = sample_attention_mask.shape[0]
                feat_len = extract_features.shape[1]
                frame_attention_mask = torch.zeros(
                    (batch_size, feat_len), dtype=sample_attention_mask.dtype, device=device
                )
                # mark the last valid index for each example and then cumsum trick to fill ones before it
                idx = (torch.arange(batch_size, device=device), (output_lengths - 1).clamp(min=0))
                frame_attention_mask[idx] = 1
                frame_attention_mask = (
                    frame_attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool().long()
                )

        # 4) feature projection -> (batch, frames, hidden_size)
        with torch.no_grad():
            hidden_states = self.hubert_model.feature_projection(
                extract_features
            )  # typically returns (B, T, hidden)
            # In HF's hubert, feature_projection is a module that returns a tensor (not tuple). If it returns tuple, adjust.

        # convert bool mask to long (1/0) if needed
        if frame_attention_mask is not None:
            frame_attention_mask = frame_attention_mask.to(dtype=torch.long)

        return hidden_states, frame_attention_mask

    def encoder_output(
        self,
        frames: torch.Tensor,  # (batch, frames, d_model)   <-- precomputed conv features
        one_zero_attention_mask: Optional[torch.Tensor] = None,  # (batch, frames)
    ):
        # Ensure device
        if frames.device.type != self.cfg.device:
            frames = frames.to(self.cfg.device)
            if one_zero_attention_mask is not None:
                one_zero_attention_mask = one_zero_attention_mask.to(self.cfg.device)

        position_embeddings = self.hubert_model.encoder.pos_conv_embed(frames)
        resid = frames + position_embeddings
        resid = self.hubert_model.encoder.layer_norm(resid)

        large_negative_number = -torch.inf
        mask = (
            repeat(1 - one_zero_attention_mask, "batch pos -> batch 1 1 pos")
            if one_zero_attention_mask is not None
            else None
        )
        additive_attention_mask = (
            torch.where(mask == 1, large_negative_number, 0) if mask is not None else None
        )
        for block in self.blocks:
            resid = block(resid, additive_attention_mask)

        return resid

    def forward(
        self,
        inputs: Union[
            torch.Tensor,  # waveform (1D) OR precomputed frames (3D)
            List[Union[torch.Tensor, np.ndarray]],  # list of waveforms
            Tuple[torch.Tensor, torch.Tensor],  # (frames, frame_mask)
        ],
        one_zero_attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
        sampling_rate: int = 16000,
        move_to_device: bool = True,
    ) -> Optional[torch.Tensor]:
        """
        HuBERT-like forward (Transformer-Lens style).

        Args:
            input: one of:
                - 1D torch.Tensor or numpy array (single waveform) OR list of 1D waveforms -> will call self.to_frames(...)
                - 3D torch.Tensor shaped (batch, frames, d_model) -> treated as precomputed frames (skip to_frames)
                - tuple (frames, frame_mask) -> use directly
            sampling_rate: sampling rate for to_frames when converting raw audio.
            use_proj: Whether to use the final head of HubertCTC
            move_to_device: move tensors to self.cfg.device (to match your other code).

        Returns:
            Depending on return_type:
              - "hidden": (batch, frames, d_model) final encoder hidden states
        """
        # ---------- 1) Normalize input: get (frames, frame_mask) ----------
        frames = None
        frame_mask = None  # one_zero_attention_mask: 1 = valid, 0 = padding
        # print(type(inputs))
        # If user passed (frames, mask) tuple
        if isinstance(inputs, tuple) and len(inputs) == 2 and isinstance(inputs[0], torch.Tensor):
            frames, frame_mask = inputs

        # If user passed a 3D tensor -> assume (B, T, D) frames (pre-projected)
        elif isinstance(inputs, torch.Tensor) and inputs.ndim == 3:
            frames = inputs
            # frame_mask stays whatever was passed as separate argument (None here)

        # Else treat as raw waveform(s) -> call to_frames
        else:
            # allow single 1D tensor or numpy array or list of tensors/arrays
            frames, frame_mask = self.to_frames(inputs)
            # to_frames should already place tensors on device if move_to_device=True
        if isinstance(frames, tuple):
            frames = frames[0]
        frame_mask = frame_mask if one_zero_attention_mask is None else one_zero_attention_mask
        # ---------- 2) Ensure device & dtype consistency ----------
        device = self.cfg.device
        if frames.device.type != device:
            frames = frames.to(device)
            if frame_mask is not None:
                frame_mask = frame_mask.to(device)

        # ---------- 3) Run encoder (respects pos_conv_embed / layer_norm / dropout inside encoder_output) ----------
        resid = self.encoder_output(frames, frame_mask)  # (B, T, d_model)

        return resid

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
    ) -> HookedEncoder:
        """Loads in the pretrained weights from huggingface. Currently supports loading weight from HuggingFace BertForMaskedLM. Unlike HookedTransformer, this does not yet do any preprocessing on the model."""
        logging.warning(
            "Support for HuBERT in TransformerLens is currently experimental, until such a time when it has feature "
            "parity with HookedTransformer and has been tested on real research tasks. Until then, backward "
            "compatibility is not guaranteed. Please see the docs for information on the limitations of the current "
            "implementation."
            "\n"
            "If using HuBERT for interpretability research, keep in mind that HuBERT has some significant architectural "
            "differences to GPT. For example, LayerNorms are applied *after* the attention and MLP components, meaning "
            "that the last LayerNorm in a block cannot be folded."
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

        model = cls(cfg, move_to_device=False, model_name=official_model_name)
        model.load_state_dict(state_dict, strict=False)
        
        model.processor = AutoFeatureExtractor.from_pretrained(official_model_name)

        if "wav2vec2" in model_name:
            hubert_model = Wav2Vec2Model.from_pretrained(official_model_name)
        else:
            hubert_model = HubertModel.from_pretrained(official_model_name)

        if move_to_device:
            if cfg.device is None:
                raise ValueError("Cannot move to device when device is None")
            hubert_model.to(cfg.device)

        hubert_model.eval()
        model.hubert_model = hubert_model

        if move_to_device:
            model.to(cfg.device)

        print(f"Loaded pretrained model {model_name} into HookedEncoder")

        return model

    @property
    def W_K(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the key weights across all layers"""
        for block in self.blocks:
            assert isinstance(block.attn, Attention)
        return torch.stack([block.attn.W_K for block in self.blocks], dim=0)

    @property
    def W_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the query weights across all layers"""
        for block in self.blocks:
            assert isinstance(block.attn, Attention)
        return torch.stack([block.attn.W_Q for block in self.blocks], dim=0)

    @property
    def W_V(self) -> Float[torch.Tensor, "n_layers n_heads d_model d_head"]:
        """Stacks the value weights across all layers"""
        for block in self.blocks:
            assert isinstance(block.attn, Attention)
        return torch.stack([block.attn.W_V for block in self.blocks], dim=0)

    @property
    def W_O(self) -> Float[torch.Tensor, "n_layers n_heads d_head d_model"]:
        """Stacks the attn output weights across all layers"""
        for block in self.blocks:
            assert isinstance(block.attn, Attention)
        return torch.stack([block.attn.W_O for block in self.blocks], dim=0)

    @property
    def W_in(self) -> Float[torch.Tensor, "n_layers d_model d_mlp"]:
        """Stacks the MLP input weights across all layers"""
        for block in self.blocks:
            assert isinstance(block.mlp, MLP)
        return torch.stack([block.mlp.W_in for block in self.blocks], dim=0)

    @property
    def W_out(self) -> Float[torch.Tensor, "n_layers d_mlp d_model"]:
        """Stacks the MLP output weights across all layers"""
        for block in self.blocks:
            assert isinstance(block.mlp, MLP)
        return torch.stack([block.mlp.W_out for block in self.blocks], dim=0)

    @property
    def b_K(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the key biases across all layers"""
        for block in self.blocks:
            assert isinstance(block.attn, Attention)
        return torch.stack([block.attn.b_K for block in self.blocks], dim=0)

    @property
    def b_Q(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the query biases across all layers"""
        for block in self.blocks:
            assert isinstance(block.attn, Attention)
        return torch.stack([block.attn.b_Q for block in self.blocks], dim=0)

    @property
    def b_V(self) -> Float[torch.Tensor, "n_layers n_heads d_head"]:
        """Stacks the value biases across all layers"""
        for block in self.blocks:
            assert isinstance(block.attn, Attention)
        return torch.stack([block.attn.b_V for block in self.blocks], dim=0)

    @property
    def b_O(self) -> Float[torch.Tensor, "n_layers d_model"]:
        """Stacks the attn output biases across all layers"""
        for block in self.blocks:
            assert isinstance(block.attn, Attention)
        return torch.stack([block.attn.b_O for block in self.blocks], dim=0)

    @property
    def b_in(self) -> Float[torch.Tensor, "n_layers d_mlp"]:
        """Stacks the MLP input biases across all layers"""
        for block in self.blocks:
            assert isinstance(block.mlp, MLP)
        return torch.stack([block.mlp.b_in for block in self.blocks], dim=0)

    @property
    def b_out(self) -> Float[torch.Tensor, "n_layers d_model"]:
        """Stacks the MLP output biases across all layers"""
        for block in self.blocks:
            assert isinstance(block.mlp, MLP)
        return torch.stack([block.mlp.b_out for block in self.blocks], dim=0)

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
