"""Shared fixtures for the JacobianLens unit-test suite."""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis import JacobianLens
from transformer_lens.utilities.activation_functions import apply_softcap

D_MODEL = 6
N_LAYERS = 4
D_VOCAB = 11
SEQ_LEN = 9
SKIP_FIRST = 2
CORPUS = "unit-test-corpus"


class _ToyBlock(nn.Module):
    def __init__(self, d_model: int, layer: int, dtype: torch.dtype):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False, dtype=dtype)
        nn.init.normal_(self.linear.weight, std=0.2)
        self.hook_out = HookPoint()
        self.hook_out.name = f"blocks.{layer}.hook_out"

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        return self.hook_out(residual + self.linear(residual))


class _CausalSumBlock(nn.Module):
    """Causal cross-position mixing with an exact triangular Jacobian."""

    def __init__(self, layer: int):
        super().__init__()
        self.hook_out = HookPoint()
        self.hook_out.name = f"blocks.{layer}.hook_out"

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        return self.hook_out(residual.cumsum(dim=1))


class _ToyTokenizer:
    def decode(self, token_ids: list[int]) -> str:
        return f"token-{token_ids[0]}"


class _ToyBridge(TransformerBridge):
    """Small real ``TransformerBridge`` subclass with Bridge-native hooks.

    The production constructor needs a Hugging Face model and architecture
    adapter. Unit tests only need its public analysis surface, so this subclass
    initializes ``nn.Module`` directly while retaining the concrete
    ``TransformerBridge`` isinstance contract.
    """

    def __init__(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        causal_final_block: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        torch.manual_seed(0)
        self.cfg = SimpleNamespace(
            n_layers=N_LAYERS,
            d_model=D_MODEL,
            d_vocab=D_VOCAB,
            d_vocab_out=D_VOCAB,
            normalization_type="LN",
            output_logits_soft_cap=None,
            model_name="toy-bridge",
            dtype=dtype,
            device="cpu",
        )
        self.adapter = SimpleNamespace(
            supports_generation=True,
            get_component_mapping=lambda: {
                "blocks": SimpleNamespace(hook_out_is_single_residual_stream=True),
                "ln_final": object(),
                "unembed": object(),
            },
            validate_output_logits_transform=lambda: None,
            apply_output_logits_transform=lambda logits: apply_softcap(
                logits, self.cfg.output_logits_soft_cap
            ),
        )
        self.compatibility_mode = False
        self._weights_processed = False
        self.tokenizer = _ToyTokenizer()
        self.embed = nn.Embedding(D_VOCAB, D_MODEL, dtype=dtype)
        blocks: list[nn.Module] = [_ToyBlock(D_MODEL, layer, dtype) for layer in range(N_LAYERS)]
        if causal_final_block:
            blocks[-1] = _CausalSumBlock(N_LAYERS - 1)
        self.blocks = nn.ModuleList(blocks)
        self.ln_final = nn.Identity()
        self.unembed = nn.Linear(D_MODEL, D_VOCAB, bias=False, dtype=dtype)

    @property
    def W_U(self) -> torch.Tensor:
        return self.unembed.weight.T

    @property
    def hook_dict(self) -> dict[str, HookPoint]:
        return {
            f"blocks.{layer}.hook_out": block.hook_out for layer, block in enumerate(self.blocks)
        }

    def parameters(self, recurse: bool = True):
        # A production bridge delegates this to its wrapped HF model. This toy
        # owns its small modules directly, so enumerate the nn.Module tree.
        return nn.Module.parameters(self, recurse=recurse)

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ):
        return nn.Module.named_parameters(
            self,
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )

    def to_tokens(self, prompt: str) -> torch.Tensor:
        ids = [(3 * index + len(prompt)) % D_VOCAB for index in range(SEQ_LEN)]
        return torch.tensor([ids], dtype=torch.long)

    def to_single_token(self, string: str) -> int:
        return len(string) % D_VOCAB

    def forward(
        self, tokens: torch.Tensor, return_type: str | None = "logits"
    ) -> torch.Tensor | None:
        residual = self.embed(tokens)
        for block in self.blocks:
            residual = block(residual)
        if return_type is None:
            return None
        return self.unembed(self.ln_final(residual))

    @contextmanager
    def hooks(
        self,
        fwd_hooks: list[tuple[str, Any]] = [],
        bwd_hooks: list[tuple[str, Any]] = [],
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
    ):
        del clear_contexts
        added: list[tuple[HookPoint, str, Any]] = []
        for direction, hook_specs in (("fwd", fwd_hooks), ("bwd", bwd_hooks)):
            for name, hook_fn in hook_specs:
                hook_point = self.hook_dict[name]
                hook_point.add_hook(hook_fn, dir=direction)
                handles = hook_point.fwd_hooks if direction == "fwd" else hook_point.bwd_hooks
                added.append((hook_point, direction, handles[-1]))
        try:
            yield self
        finally:
            if reset_hooks_end:
                for hook_point, direction, handle in added:
                    handle.hook.remove()
                    handles = hook_point.fwd_hooks if direction == "fwd" else hook_point.bwd_hooks
                    if handle in handles:
                        handles.remove(handle)

    def run_with_cache(
        self,
        input: torch.Tensor,
        return_cache_object: bool = False,
        remove_batch_dim: bool = False,
        names_filter: Any = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del return_cache_object, remove_batch_dim, kwargs

        def wanted(name: str) -> bool:
            if names_filter is None:
                return True
            if isinstance(names_filter, str):
                return name == names_filter
            if callable(names_filter):
                return bool(names_filter(name))
            return name in names_filter

        cache: dict[str, torch.Tensor] = {}

        def cache_hook(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:
            assert hook.name is not None
            cache[hook.name] = activation.detach()
            return activation

        cache_hooks = [(name, cache_hook) for name in self.hook_dict if wanted(name)]
        with self.hooks(fwd_hooks=cache_hooks):
            logits = self(input)
        assert logits is not None
        return logits, cache


class _NotABridge(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(n_layers=N_LAYERS, d_model=D_MODEL)


def _lens(
    *,
    n_prompts: int = 1,
    metadata: dict[str, Any] | None = None,
) -> JacobianLens:
    return JacobianLens(
        {0: torch.eye(D_MODEL)},
        n_prompts=n_prompts,
        d_model=D_MODEL,
        metadata=metadata,
    )


@pytest.fixture(scope="module")
def toy_model() -> _ToyBridge:
    return _ToyBridge()
