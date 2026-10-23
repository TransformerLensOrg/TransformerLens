"""Whisper (``WhisperForConditionalGeneration``) adapter: a BART-family pre-norm
encoder-decoder whose encoder reads log-mel audio through two convolutions and a
fixed sinusoidal position table instead of token embeddings. The text decoder is
the standard BART-family stack with learned positions and a ``proj_out``
unembedding tied to ``decoder_embed``; there is no ``final_logits_bias``."""

from typing import Any, Dict

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
    start-of-transcript prompt). Returns decoder logits under teacher forcing.

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
    # generate() is a follow-up; it needs the audio-conditioned decode loop.
    supports_generation = False
    # P1 (component + forward parity vs HF) only: P2/P4 need generate() and P3
    # needs weight processing, both follow-ups.
    applicable_phases = [1]

    def __init__(self, cfg: Any) -> None:
        """Set the audio seq2seq flags on top of the BART-family setup."""
        super().__init__(cfg)
        self.cfg.is_audio_model = True
        self.cfg.is_audio_seq2seq = True

    def _encoder_embeddings(self) -> Dict[str, GeneralizedComponent]:
        conv1 = GeneralizedComponent(name="model.encoder.conv1")
        conv2 = GeneralizedComponent(name="model.encoder.conv2")
        pos_embed = PosEmbedBridge(name="model.encoder.embed_positions")
        conv1.mark_pos_dim(2)
        conv2.mark_pos_dim(2)
        pos_embed.mark_pos_dim(0)
        return {"conv1": conv1, "conv2": conv2, "pos_embed": pos_embed}
