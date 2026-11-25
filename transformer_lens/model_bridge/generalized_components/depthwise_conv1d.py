"""Bridge for Mamba-style depthwise causal Conv1d (distinct from GPT-2's Conv1D linear)."""
import contextlib
import inspect
import threading
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, cast

import torch

from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)

_CAUSAL_CONV1D_FN = "causal_conv1d_fn"
# Packed-sequence arguments: HF's torch path refuses them and the kernel honours them.
_PACKED_SEQUENCE_KWARGS = ("seq_idx", "cu_seqlens")


class DepthwiseConv1DBridge(GeneralizedComponent):
    """Wraps an ``nn.Conv1d`` depthwise causal convolution with input/output hooks.

    Hook shapes (channel-first, as HF's MambaMixer transposes before the call):
        hook_in:  [batch, channels, seq_len]
        hook_out: [batch, channels, seq_len + conv_kernel - 1]  (pre causal trim)

    Because ``hook_out`` is captured before the causal trim, negative ``pos_slice``
    values index the causal padding rather than real tokens — slice positive
    positions from the left, or trim the last ``conv_kernel - 1`` entries first.

    From transformers 5.15 the mixers no longer call ``self.conv1d(...)``: they
    pass ``self.conv1d.weight`` to a module-level ``causal_conv1d_fn``. The mixer
    bridges run that call through this bridge (see ``route_causal_conv1d``), so
    the hooks fire as before.

    Decode-step limitation: on stateful generation, HF's Mamba/Mamba-2 mixers
    bypass ``self.conv1d(...)`` and read ``self.conv1d.weight`` directly, so the
    forward hook never fires on decode steps — only on prefill. For per-step
    conv output during decode, compute it manually from the cached conv_states
    and ``conv1d.original_component.weight``, or run token-by-token via
    ``forward()`` instead of ``generate()``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Channel-first: the sequence axis is dim 2 (see the class docstring).
        self.hook_in.pos_dim = 2
        self.hook_out.pos_dim = 2

    def forward(self, input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. "
                "Call set_original_component() first."
            )
        input = self.hook_in(input)
        output = self.original_component(input, *args, **kwargs)
        output = self.hook_out(output)
        return output

    def get_random_inputs(
        self,
        batch_size: int = 2,
        seq_len: int = 8,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        if self.original_component is None:
            raise RuntimeError(
                f"Original component not set for {self.name}. "
                "Call set_original_component() first."
            )
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32
        conv = cast(torch.nn.Conv1d, self.original_component)
        channels = conv.in_channels  # exact, from the wrapped Conv1d
        return {"args": (torch.randn(batch_size, channels, seq_len, device=device, dtype=dtype),)}

    def __repr__(self) -> str:
        if self.original_component is not None:
            try:
                in_channels = self.original_component.in_channels
                out_channels = self.original_component.out_channels
                kernel_size = self.original_component.kernel_size
                groups = self.original_component.groups
                return (
                    f"DepthwiseConv1DBridge({in_channels} -> {out_channels}, "
                    f"kernel_size={kernel_size}, groups={groups})"
                )
            except AttributeError:
                return (
                    f"DepthwiseConv1DBridge(name={self.name}, "
                    f"original_component={type(self.original_component).__name__})"
                )
        return f"DepthwiseConv1DBridge(name={self.name}, original_component=None)"


def _is_view_of(tensor: Any, param: torch.Tensor) -> bool:
    """Whether ``tensor`` is ``param`` or a view of it (HF passes ``weight.squeeze(1)``)."""
    return tensor is param or getattr(tensor, "_base", None) is param


class _ConvRoute:
    """The conv bridge one ``route_causal_conv1d`` context runs HF's calls through."""

    def __init__(
        self, namespace: Dict[str, Any], conv: DepthwiseConv1DBridge, conv_module: torch.nn.Conv1d
    ) -> None:
        self.namespace = namespace
        self.conv = conv
        self.conv_module = conv_module

    def matches(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> bool:
        weight = args[1] if len(args) > 1 else kwargs.get("weight")
        bias = args[2] if len(args) > 2 else kwargs.get("bias")
        return (
            len(args) >= 2
            and isinstance(args[0], torch.Tensor)
            and _is_view_of(weight, self.conv_module.weight)
            and bias is self.conv_module.bias
            and all(kwargs.get(name) is None for name in _PACKED_SEQUENCE_KWARGS)
        )

    def run(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> torch.Tensor:
        hidden_states = args[0]
        activation = args[3] if len(args) > 3 else kwargs.get("activation")
        seq_len = hidden_states.shape[-1]
        out = self.conv(hidden_states.to(self.conv_module.weight.dtype))[..., :seq_len]
        if activation is not None:
            from transformers.activations import ACT2FN

            out = ACT2FN[activation](out)
        return out.to(hidden_states.dtype)


class _InstalledDispatch:
    """The stand-in for one modeling file's ``causal_conv1d_fn`` while any route uses it."""

    def __init__(self, namespace: Dict[str, Any], original: Callable[..., Any]) -> None:
        self.original = original
        self.users = 0

        def dispatch(*args: Any, **kwargs: Any) -> Any:
            # Only the calling thread's routes apply, so concurrent forwards stay apart.
            for route in reversed(_thread_routes()):
                if route.namespace is namespace and route.matches(args, kwargs):
                    return route.run(args, kwargs)
            return original(*args, **kwargs)

        self.function = dispatch


_ROUTING_LOCK = threading.Lock()
_INSTALLED: Dict[int, _InstalledDispatch] = {}  # keyed by id() of the modeling file's globals
_THREAD_STATE = threading.local()


def _thread_routes() -> List[_ConvRoute]:
    routes = getattr(_THREAD_STATE, "routes", None)
    if routes is None:
        routes = []
        _THREAD_STATE.routes = routes
    return routes


def _install_dispatch(namespace: Dict[str, Any]) -> Optional[_InstalledDispatch]:
    with _ROUTING_LOCK:
        installed = _INSTALLED.get(id(namespace))
        if installed is None:
            original = namespace.get(_CAUSAL_CONV1D_FN)
            if not callable(original):
                return None
            installed = _InstalledDispatch(namespace, original)
            namespace[_CAUSAL_CONV1D_FN] = installed.function
            _INSTALLED[id(namespace)] = installed
        installed.users += 1
        return installed


def _release_dispatch(namespace: Dict[str, Any], installed: _InstalledDispatch) -> None:
    with _ROUTING_LOCK:
        installed.users -= 1
        if installed.users == 0:
            del _INSTALLED[id(namespace)]
            # Leave alone a function that someone else has put there since.
            if namespace.get(_CAUSAL_CONV1D_FN) is installed.function:
                namespace[_CAUSAL_CONV1D_FN] = installed.original


@contextlib.contextmanager
def route_causal_conv1d(hf_module: Any, conv_attr: str) -> Iterator[None]:
    """Run HF's ``causal_conv1d_fn`` through the conv bridge while ``hf_module`` runs.

    From transformers 5.15 the Mamba-family mixers (Mamba, Mamba-2 and the hybrids
    built on them, LFM2's short conv) no longer call their conv module: the forward
    passes ``self.<conv_attr>.weight`` and ``.bias`` to a module-level
    ``causal_conv1d_fn`` of the modeling file, so the ``DepthwiseConv1DBridge``
    swapped in for the conv never runs and its hooks never fire. Inside this
    context, a call of that function on this thread with this conv's own weight and
    bias runs the bridge instead, trims the output to the input length and applies
    the activation, which is what the function's torch path computes. The bridge
    then runs the torch convolution even where a causal-conv1d kernel is installed.

    Every other call goes to the original function: calls from other threads, calls
    that carry ``seq_idx`` or ``cu_seqlens`` (packed sequences, which the torch path
    refuses and the kernel keeps apart), and the decode-step ``causal_conv1d_update``.
    The module-level function is replaced while any thread is inside such a context
    and put back when the last one leaves. Nothing changes when the conv is not a
    bridge, when the modeling file has no such function, or on transformers < 5.15,
    whose mixers call the conv module directly (their ``causal_conv1d_fn`` is None
    without the kernel package, and the kernel path calls it with keyword arguments).
    """
    conv = getattr(hf_module, conv_attr, None)
    forward = getattr(type(hf_module), "forward", None)
    namespace = getattr(inspect.unwrap(forward), "__globals__", None) if forward else None
    conv_module = getattr(conv, "original_component", None)
    if (
        namespace is None
        or not isinstance(conv, DepthwiseConv1DBridge)
        or not isinstance(conv_module, torch.nn.Conv1d)
        # The bridge reproduces the function only for a causal depthwise conv.
        or conv_module.padding != (conv_module.kernel_size[0] - 1,)
        or conv_module.groups != conv_module.in_channels
    ):
        yield
        return
    installed = _install_dispatch(namespace)
    if installed is None:
        yield
        return

    route = _ConvRoute(namespace, conv, conv_module)
    routes = _thread_routes()
    routes.append(route)
    try:
        yield
    finally:
        routes.remove(route)
        _release_dispatch(namespace, installed)
