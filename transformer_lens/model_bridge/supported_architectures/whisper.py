"""Whisper (``WhisperForConditionalGeneration``) adapter: a BART-family pre-norm
encoder-decoder whose encoder reads log-mel audio through two convolutions and a
fixed sinusoidal position table instead of token embeddings. The text decoder is
the standard BART-family stack with learned positions and a ``proj_out``
unembedding tied to ``decoder_embed``; there is no ``final_logits_bias``."""

from typing import Any, Dict

from transformer_lens.model_bridge.architecture_adapter import ResidualLayout
from transformer_lens.model_bridge.generalized_components import PosEmbedBridge
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.supported_architectures.bart import (
    BartFamilyArchitectureAdapter,
)


class WhisperArchitectureAdapter(BartFamilyArchitectureAdapter):
    """Architecture adapter for WhisperForConditionalGeneration models.

    Inputs: ``bridge(input_features, decoder_input=decoder_ids)``, where
    ``input_features`` is the log-mel spectrogram ``[batch, n_mels, 2 * n_frames]``
    from ``bridge.processor`` and ``decoder_ids`` are decoder token ids (e.g. the
    start-of-transcript prompt), one row per clip or one row shared by all clips.
    ``bridge.to_tokens(text)`` builds that prompt from the tokenizer's prefix tokens
    (set the language and task with ``bridge.tokenizer.set_prefix_tokens``). Returns
    decoder logits under teacher forcing.

    Transcription: ``bridge.generate(input_features, return_type="str")`` decodes from
    the same prefix (or ``decoder_input``), applying the checkpoint's
    ``suppress_tokens`` and ``begin_suppress_tokens`` as HF's generate does.

    Encoder front end (channel-first, so its hooks declare position on dim 2):

    - ``conv1`` / ``conv2``: the two ``nn.Conv1d`` layers. HF applies GELU outside
      the modules, so ``hook_out`` is pre-activation; ``conv2`` halves the frames.
    - ``pos_embed``: the fixed sinusoidal table, read whole as ``[n_frames, d_model]``
      (no batch dim) and added to the transposed ``conv2`` output.

    ``encoder_blocks.{i}.hook_out`` is ``[batch, n_frames, d_model]``: 1500 frames,
    one per 20 ms of the 30 s window, on every released checkpoint.

    Attention biases: ``q_proj``, ``v_proj`` and ``out_proj`` carry a bias but
    ``k_proj`` does not (HF builds it with ``bias=False``), so ``attn.k.bias`` is
    ``None`` on every self- and cross-attention.

    Stacks may be asymmetric (large-v3-turbo: 32 encoder / 4 decoder layers;
    distil-whisper: 2 decoder layers); ``n_layers`` follows the decoder, as for
    Blenderbot.
    """

    require_symmetric_layers = False
    n_layers_from = "decoder"
    # Checkpoints ship scale_embedding=False, and HF's forward never applies it.
    force_scale_embedding = False
    has_final_stack_norm = True
    unembed_name = "proj_out"
    audio_input_name = "input_features"
    # P1 (component + forward parity vs HF) only: P2-P4 drive the model with text prompts.
    applicable_phases = [1]

    def __init__(self, cfg: Any) -> None:
        """Set the audio seq2seq flags on top of the BART-family setup."""
        super().__init__(cfg)
        self.cfg.is_audio_model = True
        self.cfg.is_audio_seq2seq = True
        # The tokenizer's own <|startoftranscript|> prefix starts the decoder prompt; a
        # prepended BOS (<|endoftext|>) would land inside it.
        self.cfg.default_prepend_bos = False
        # Pre-LN with every residual read through a LayerNorm, so folding and centering
        # are exact; residual_layout() tells weight processing where things are.
        self.supports_fold_ln = True
        self.supports_center_writing_weights = True

    def residual_layout(self) -> ResidualLayout:
        """Norm readers, residual writers and value/output pairs of both stacks.

        Each block's norms feed its own sublayers. The encoder's final norm feeds the
        cross-attention keys and values of every decoder block, and the decoder's
        final norm feeds the unembedding. Encoder writes are the attention and MLP
        outputs plus the sinusoidal positions; the GELU'd conv output is not linear
        in any weight, so it stays uncentered (centering is exact either way, since
        every read is through a LayerNorm). Key projections have no bias.
        """
        encoder_layers = getattr(self.cfg, "encoder_layers", None) or self.cfg.n_layers
        decoder_layers = self.cfg.n_layers
        norm_readers: Dict[str, list[str]] = {}
        key_projections: set[str] = set()
        residual_writers: list[str] = []
        value_output_pairs: list[tuple[str, str]] = []
        for i in range(encoder_layers):
            block = f"encoder_blocks.{i}"
            attn = f"{block}.self_attn"
            norm_readers[f"{block}.self_attn_layer_norm"] = [f"{attn}.{p}" for p in "qkv"]
            norm_readers[f"{block}.final_layer_norm"] = [f"{block}.mlp.in"]
            key_projections.add(f"{attn}.k")
            residual_writers += [f"{attn}.o", f"{block}.mlp.out"]
            value_output_pairs.append((f"{attn}.v", f"{attn}.o"))
        cross_kv: list[str] = []
        for i in range(decoder_layers):
            block = f"decoder_blocks.{i}"
            self_attn, cross_attn = f"{block}.self_attn", f"{block}.encoder_attn"
            norm_readers[f"{block}.self_attn_layer_norm"] = [f"{self_attn}.{p}" for p in "qkv"]
            norm_readers[f"{block}.encoder_attn_layer_norm"] = [f"{cross_attn}.q"]
            norm_readers[f"{block}.final_layer_norm"] = [f"{block}.mlp.in"]
            cross_kv += [f"{cross_attn}.k", f"{cross_attn}.v"]
            key_projections.update({f"{self_attn}.k", f"{cross_attn}.k"})
            residual_writers += [f"{self_attn}.o", f"{cross_attn}.o", f"{block}.mlp.out"]
            value_output_pairs += [
                (f"{self_attn}.v", f"{self_attn}.o"),
                (f"{cross_attn}.v", f"{cross_attn}.o"),
            ]
        norm_readers["encoder_ln_final"] = cross_kv
        norm_readers["decoder_ln_final"] = ["unembed"]
        return ResidualLayout(
            norm_readers=norm_readers,
            key_projections=frozenset(key_projections),
            residual_writers=residual_writers,
            embedding_writers=["pos_embed", "decoder_embed", "decoder_pos_embed"],
            value_output_pairs=value_output_pairs,
        )

    def _encoder_embeddings(self) -> Dict[str, GeneralizedComponent]:
        conv1 = GeneralizedComponent(name="model.encoder.conv1")
        conv2 = GeneralizedComponent(name="model.encoder.conv2")
        pos_embed = PosEmbedBridge(name="model.encoder.embed_positions")
        conv1.mark_pos_dim(2)
        conv2.mark_pos_dim(2)
        pos_embed.mark_pos_dim(0)
        return {"conv1": conv1, "conv2": conv2, "pos_embed": pos_embed}
