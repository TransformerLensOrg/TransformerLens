"""FalconMamba architecture adapter.

TII's FalconMamba (``FalconMambaForCausalLM``) is Mamba-1 with one numerical
delta: a parameter-free RMS normalization applied to the B, C, and dt
projections inside the mixer (``rms_forward`` in the HF source). Module paths
are identical to Mamba, so the mapping is inherited wholesale; the RMS is
honored inside SSMMixerBridge's eager scan and S6 reconstruction via the
wrapped mixer's ``rms_eps`` attribute.
"""

from transformer_lens.model_bridge.supported_architectures.mamba import (
    MambaArchitectureAdapter,
)


class FalconMambaArchitectureAdapter(MambaArchitectureAdapter):
    """Architecture adapter for FalconMambaForCausalLM models."""
