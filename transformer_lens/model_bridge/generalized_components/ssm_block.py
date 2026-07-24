"""Block container for State Space Model (Mamba) layers: norm → mixer → residual."""

from __future__ import annotations

from transformer_lens.model_bridge.generalized_components.opaque_block import (
    OpaqueBlockBridge,
)


class SSMBlockBridge(OpaqueBlockBridge):
    """Block bridge for SSM layers (Mamba, Falcon-H1, etc.).

    Extends :class:`OpaqueBlockBridge` with SSM-specific hook aliases:

    - ``hook_mixer_in``  → ``mixer.hook_in``
    - ``hook_mixer_out`` → ``mixer.hook_out``

    These aliases are only meaningful for architectures whose blocks contain a
    ``mixer`` submodule (SSM or hybrid SSM/attention). For non-SSM architectures
    use :class:`OpaqueBlockBridge` directly.
    """

    hook_aliases = {
        **OpaqueBlockBridge.hook_aliases,
        "hook_mixer_in": "mixer.hook_in",
        "hook_mixer_out": "mixer.hook_out",
    }
