"""Non-torch bridge: vLLM workers, Inspect remote providers."""
from __future__ import annotations

import warnings
from typing import Any, Callable, List, Optional, Tuple, Union

from transformer_lens.hook_points import HookIntrospectionMixin, HookPoint
from transformer_lens.model_bridge.bridge_core import BridgeCore
from transformer_lens.model_bridge.driver_protocol import (
    ForwardResult,
    TensorLike,
    to_torch,
    validate_driver,
)


class RemoteBridge(BridgeCore, HookIntrospectionMixin):
    """Bridge for backends with no local ``nn.Module`` (vLLM, Inspect).

    No nn.Module parentage strips the torch-only surface; driver pre-declares
    ``supported_hook_points`` (no model to walk).
    """

    def __init__(
        self,
        adapter: Any,
        tokenizer: Any,
        driver: Any,
    ) -> None:
        if not driver.supported_hook_points:
            raise ValueError(
                "RemoteBridge requires driver.supported_hook_points to be "
                "non-empty: non-torch drivers own the hook namespace because "
                "there is no local model for the bridge to walk."
            )
        BridgeCore.__init__(self, adapter, tokenizer, driver)
        # No local device; tensor.to(None) is a no-op so downstream patterns degrade cleanly.
        self.cfg.device = None
        # HookPoint is nn.Module-backed but RemoteBridge isn't an nn.Module —
        # named_modules() walks don't apply; only registry lookup matters.
        for name in driver.supported_hook_points:
            hp = HookPoint()
            hp.name = name
            self._hook_registry[name] = hp
        self._hook_registry_initialized = True
        validate_driver(self._driver, after_bridge_construction=True)

    @staticmethod
    def boot_vllm(*args: Any, **kwargs: Any) -> "RemoteBridge":
        """Boot a model via vLLM. Returns a RemoteBridge wrapping a VLLMDriver.

        Mirrors ``TransformerBridge.boot_transformers``. Lazy import so
        ``remote_bridge`` itself stays vLLM-agnostic — only callers of this
        method need vLLM installed. See :func:`sources.vllm.boot_vllm` for kwargs.
        """
        from .sources.vllm import boot_vllm as _boot_vllm

        return _boot_vllm(*args, **kwargs)

    @staticmethod
    def boot_inspect(*args: Any, **kwargs: Any) -> "RemoteBridge":
        """Boot a model via an inspect_ai provider. Returns a RemoteBridge wrapping
        an InspectDriver. Lazy import keeps remote_bridge inspect-agnostic. See
        :func:`sources.inspect.boot_inspect` for kwargs."""
        from .sources.inspect import boot_inspect as _boot_inspect

        return _boot_inspect(*args, **kwargs)

    def _scan_existing_hooks(self, module: Any, prefix: str = "") -> None:
        """No-op: registry built from driver declarations in __init__."""

    def forward(
        self,
        input: Any = None,
        *,
        return_type: str | None = "logits",
        loss_per_token: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Tokenize → driver.forward → replay captures → finalize per return_type."""
        # Early copy of _finalize_return's gate — fail before the wasted remote forward.
        self._check_loss_supported(return_type)
        self._reject_stop_at_layer(kwargs.pop("stop_at_layer", None))
        if isinstance(input, str):
            kwargs["input_ids"] = self.to_tokens(input)  # BOS-aware, matches boot_transformers
        elif isinstance(input, list) and any(isinstance(item, str) for item in input):
            # Would otherwise be treated as raw input_ids and crash deep in numpy.
            raise TypeError(
                "RemoteBridge.forward received a list of strings; batched string "
                "input is unsupported here. Pass a single str, or tokenize with "
                "to_tokens() and pass token ids."
            )
        elif input is not None:
            kwargs["input_ids"] = input

        # Only request hooks with a registered handler, so a plain forward(tokens)
        # ships logits alone instead of the full residual decomposition every call.
        if "capture" not in kwargs:
            kwargs["capture"] = tuple(
                name for name, hp in self._hook_registry.items() if hp.fwd_hooks
            )

        result: ForwardResult = self._driver.forward(**kwargs)
        if result.captured:
            self._replay_captures(result.captured)

        logits: Any = result.logits
        if logits is not None and not isinstance(logits, TensorLike):
            return logits  # weird shape — let caller handle
        if logits is not None:
            logits = to_torch(logits)

        return self._finalize_return(
            return_type,
            logits,
            kwargs.get("input_ids"),
            is_audio_model=getattr(self.cfg, "is_audio_model", False),
            loss_per_token=loss_per_token,
        )

    def run_with_hooks(
        self,
        input: Any,
        fwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        bwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
        return_type: Optional[str] = "logits",
        stop_at_layer: Optional[int] = None,
        remove_batch_dim: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Run with hooks. Remote fwd_hooks fire post-forward on captured
        activations (read-only) — they can't alter the computation, so warn; use
        ``intervene=`` specs to mutate. bwd_hooks are unsupported (no backward)."""
        if bwd_hooks:
            raise NotImplementedError(
                "RemoteBridge has no backward pass; bwd_hooks are unsupported."
            )
        self._reject_stop_at_layer(stop_at_layer)
        if fwd_hooks:
            warnings.warn(
                "RemoteBridge fwd_hooks fire on already-captured activations (read-only): "
                "a hook that returns a modified tensor does NOT change the forward "
                "computation or logits — the return is discarded. To intervene on the "
                "computation, pass intervene={hook_name: {'op': ...}} to "
                "forward()/run_with_cache().",
                UserWarning,
                stacklevel=2,
            )
        return super().run_with_hooks(
            input,
            fwd_hooks=fwd_hooks,
            bwd_hooks=bwd_hooks,
            reset_hooks_end=reset_hooks_end,
            clear_contexts=clear_contexts,
            return_type=return_type,
            remove_batch_dim=remove_batch_dim,
            **kwargs,
        )

    @staticmethod
    def _reject_stop_at_layer(stop_at_layer: Any) -> None:
        # BridgeCore's stop hooks need a local `blocks` tree; silently ignoring
        # the kwarg would lie about compute cost.
        if stop_at_layer is not None:
            raise NotImplementedError(
                "RemoteBridge does not support stop_at_layer: the remote engine "
                "always runs the full forward pass."
            )

    def run_with_cache(self, *args: Any, **kwargs: Any) -> Any:
        """Cache via driver captures; stop_at_layer rejected (full remote forward)."""
        self._reject_stop_at_layer(kwargs.get("stop_at_layer"))
        return super().run_with_cache(*args, **kwargs)

    def to_tokens(self, input: Any, prepend_bos: bool | None = None, truncate: bool = True) -> Any:
        """Tokenize a string with the same BOS handling as ``TransformerBridge``.

        Mirrors ``cfg.default_prepend_bos`` / ``tokenizer_prepends_bos`` so
        ``boot_inspect(m).run_with_cache("text")`` matches ``boot_transformers(m)``
        on the same string — a bare ``encode`` (no BOS) would silently diverge.
        """
        from transformer_lens import utils

        assert self.tokenizer is not None, "Tokenizer must be set."
        if prepend_bos is None:
            prepend_bos = getattr(self.cfg, "default_prepend_bos", True)
        tokenizer_prepends_bos = getattr(self.cfg, "tokenizer_prepends_bos", True)
        if prepend_bos and not tokenizer_prepends_bos:
            input = utils.get_input_with_manually_prepended_bos(self.tokenizer.bos_token, input)
        if isinstance(input, str):
            input = [input]
        # A single sequence never needs padding; only pad when batching (which also
        # avoids requiring a pad_token on tokenizers that lack one, e.g. raw gpt2).
        tokens = self.tokenizer(
            input,
            return_tensors="pt",
            padding=len(input) > 1,
            truncation=truncate,
            max_length=self.cfg.n_ctx if truncate else None,
        )["input_ids"]
        if not prepend_bos and tokenizer_prepends_bos:
            tokens = utils.get_tokens_with_bos_removed(self.tokenizer, tokens)
        return tokens

    def __enter__(self) -> "RemoteBridge":
        """Use as a context manager so the engine is released on exit:
        ``with RemoteBridge.boot_vllm(...) as bridge: ...``."""
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort safety net for notebooks that drop the bridge without
        # close() — repeated boot_vllm would otherwise OOM. close() is idempotent.
        try:
            self.close()
        except Exception:
            pass
