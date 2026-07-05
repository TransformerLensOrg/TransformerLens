import torch
import torch.nn as nn
from typing import Dict, Any
from transformers import ASTConfig

from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.config.hooked_transformer_config import HookedTransformerConfig
from transformer_lens.hook_points import HookPoint

class ASTEmbed(nn.Module):
    # custom embedding layer for ViT/AST architectures
    # extracts 16x16 patches using Conv2d, flattens into sequence
    # prepends to the CLS token

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # AST uses a 16x16 patch size
        patch_size = (16, 16)
        # AST uses overlapping patches with stride of 10, this frequenly causes problems if not known
        stride = (10, 10)

        # the Conv2d layer acts as the patch extractor
        self.patch_embeddings = nn.Conv2d(
            in_channels=1,
            out_channels=cfg.d_model,
            kernel_size=patch_size,
            stride=stride
        )

        # learnable CLS token prepended to sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.distillation_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))

        # standard TransformerLens hook point
        self.hook_embed = HookPoint()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input shape: [batch, channels (1), freq, time]
        batch_size = x.shape[0]

        # mimic HF's internals: 1. add channel dim [batch, 1, time, freq], 2. transposes to image format [batch, 1, freq, time]
        x = x.unsqueeze(1).transpose(2, 3)

        # 1. extract patches -> [batch, d_model, grid_h, grid_w]
        x = self.patch_embeddings(x)

        # 2. flatten spatial dimensions -> [batch, d_model, num_patches]
        x = x.flatten(2)

        # 3. transpose to standard sequence format -> [batch, num_patches, d_model]
        x = x.transpose(1, 2)

        # 4. expand and prepend CLS token -> [batch, 1 + num_patches, d_model]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        dist_tokens = self.distillation_token.expand(batch_size, -1, -1)

        # concat cls, then distillation, then patches
        x = torch.cat((cls_tokens, dist_tokens, x), dim=1)

        return self.hook_embed(x)

class ASTAdapter(ArchitectureAdapter):
    # adapter for Audio Spectrogram Transformer (AST)
    # takes audio input as image, same core architecture as ViT-base

    @classmethod
    def get_config_map(cls, hf_config: ASTConfig) -> Dict[str, Any]:
        # maps huggingface AST config to HookedTransformerConfig
        return {
            "d_model": hf_config.hidden_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "d_vocab": hf_config.num_labels,
            "n_ctx": hf_config.max_length,
            "eps": hf_config.layer_norm_eps,
            "act_fn": "gelu",
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,

            # tell TL this is Image/Audio model not Text model
            "attention_dir": "bidirectional",
        }
    
    @classmethod
    def convert_weights(cls, hf_state_dict: Dict[str, torch.Tensor], tl_config: HookedTransformerConfig) -> Dict[str, torch.Tensor]:
        # translate the huggingface weight keys to TransformerLens
        state_dict = {}

        # 1. convolutional patch embedding
        if "audio_spectrogram_transformer.embeddings.patch_embeddings.projection.weight" in hf_state_dict:
            state_dict["embed.patch_embeddings.weight"] = hf_state_dict["audio_spectrogram_transformer.embeddings.patch_embeddings.projection.weight"]
            state_dict["embed.patch_embeddings.bias"] = hf_state_dict["audio_spectrogram_transformer.embeddings.patch_embeddings.projection.bias"]
        
        # 2. positional embeddings & CLS token
        if "audio_spectrogram_transformer.embeddings.position_embeddings" in hf_state_dict:
            state_dict["pos_embed.W_pos"] = hf_state_dict["audio_spectrogram_transformer.embeddings.position_embeddings"].squeeze(0)

        if "audio_spectrogram_transformer.embeddings.cls_token" in hf_state_dict:
            state_dict["embed.cls_token"] = hf_state_dict["audio_spectrogram_transformer.embeddings.cls_token"]
        
        if "audio_spectrogram_transformer.embeddings.distillation_token" in hf_state_dict:
            state_dict["embed.distillation_token"] = hf_state_dict["audio_spectrogram_transformer.embeddings.distillation_token"]

        # 3. dense transformer blocks
        for i in range(tl_config.n_layers):
            hf_prefix = f"audio_spectrogram_transformer.layers.{i}."
            tl_prefix = f"blocks.{i}."

            # LayerNorm 1
            state_dict[f"{tl_prefix}ln1.w"] = hf_state_dict[f"{hf_prefix}layernorm_before.weight"]
            state_dict[f"{tl_prefix}ln1.b"] = hf_state_dict[f"{hf_prefix}layernorm_before.bias"]

            state_dict[f"{tl_prefix}attn.W_Q"] = cls._reshape_weight(
                hf_state_dict[f"{hf_prefix}attention.q_proj.weight"], tl_config
            )
            state_dict[f"{tl_prefix}attn.b_Q"] = cls._reshape_bias(
                hf_state_dict[f"{hf_prefix}attention.q_proj.bias"], tl_config
            )

            state_dict[f"{tl_prefix}attn.W_K"] = cls._reshape_weight(
                hf_state_dict[f"{hf_prefix}attention.k_proj.weight"], tl_config
            )
            state_dict[f"{tl_prefix}attn.b_K"] = cls._reshape_bias(
                hf_state_dict[f"{hf_prefix}attention.k_proj.bias"], tl_config
            )

            state_dict[f"{tl_prefix}attn.W_V"] = cls._reshape_weight(
                hf_state_dict[f"{hf_prefix}attention.v_proj.weight"], tl_config
            )
            state_dict[f"{tl_prefix}attn.b_V"] = cls._reshape_bias(
                hf_state_dict[f"{hf_prefix}attention.v_proj.bias"], tl_config
            )

            # attention output
            state_dict[f"{tl_prefix}attn.W_O"] = cls._reshape_weight(
                hf_state_dict[f"{hf_prefix}attention.o_proj.weight"], tl_config, is_output=True
            )
            state_dict[f"{tl_prefix}attn.b_O"] = hf_state_dict[f"{hf_prefix}attention.o_proj.bias"]

            # LayerNorm 2
            state_dict[f"{tl_prefix}ln2.w"] = hf_state_dict[f"{hf_prefix}layernorm_after.weight"]
            state_dict[f"{tl_prefix}ln2.b"] = hf_state_dict[f"{hf_prefix}layernorm_after.bias"]

            # MLP
            state_dict[f"{tl_prefix}mlp.W_in"] = hf_state_dict[f"{hf_prefix}mlp.fc1.weight"].T
            state_dict[f"{tl_prefix}mlp.b_in"] = hf_state_dict[f"{hf_prefix}mlp.fc1.bias"]

            state_dict[f"{tl_prefix}mlp.W_out"] = hf_state_dict[f"{hf_prefix}mlp.fc2.weight"].T
            state_dict[f"{tl_prefix}mlp.b_out"] = hf_state_dict[f"{hf_prefix}mlp.fc2.bias"]
        
        # 4. final LayerNorm & Classifier Unembedding
        if "audio_spectrogram_transformer.layernorm.weight" in hf_state_dict:
            state_dict["ln_final.w"] = hf_state_dict["audio_spectrogram_transformer.layernorm.weight"]
            state_dict["ln_final.b"] = hf_state_dict["audio_spectrogram_transformer.layernorm.bias"]

        if "classifier.dense.weight" in hf_state_dict:
            state_dict["unembed.W_U"] = hf_state_dict["classifier.dense.weight"].T
            state_dict["unembed.b_U"] = hf_state_dict["classifier.dense.bias"]
        
        return state_dict
    
    @staticmethod
    def _reshape_weight(weight: torch.Tensor, cfg: HookedTransformerConfig, is_output: bool = False) -> torch.Tensor:
        # reshapes dense HF attention weights into [n_heads, d_model, d_head] for TL
        if is_output:
            # W_O requires distinct shaping: [n_heads, d_head, d_model]
            return weight.view(cfg.d_model, cfg.n_heads, cfg.d_head).permute(1, 2, 0)
        else:
            # Q, K, V
            return weight.view(cfg.n_heads, cfg.d_head, cfg.d_model).transpose(1, 2)
        
    @staticmethod
    def _reshape_bias(bias: torch.Tensor, cfg: HookedTransformerConfig) -> torch.Tensor:
        # reshapes HF attention biases into [n_heads, d_head] for TL
        return bias.view(cfg.n_heads, cfg.d_head)