"""LiquidAI LFM2 gated short-convolution mixer bridge."""

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


class Lfm2ShortConvBridge(GeneralizedComponent):
    """Wrapper around LFM2's double-gated short-convolution mixer.

    Delegates the forward to HF's ``Lfm2ShortConv`` (preserving its fast CUDA /
    slow PyTorch dispatch and cache handling) and hooks the residual-stream
    input/output. Inner in_proj / conv / out_proj are spliced in as submodules,
    so their hooks fire during HF's own forward.

    Decode-step caveat: on stateful generation HF's conv path reads
    ``conv.weight`` directly instead of calling ``self.conv(...)``, so
    ``conv.hook_out`` fires only on prefill — see DepthwiseConv1DBridge.

    CUDA caveat: Hooks surrounding the conv1D operation only fire on the hf
    "slow path" i.e. if not on CUDA / fast path not available / torch dynamo
    compiling.
    """

    hook_aliases = {
        "hook_in_proj": "in.hook_out",
        "hook_conv": "conv.hook_out",
        "hook_gated": "out.hook_in",
    }
