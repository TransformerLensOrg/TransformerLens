"""Regression test: every adapter's hook aliases resolve to a real HookPoint.

Catches bugs where an alias target path doesn't navigate to a HookPoint
(the complementary case to Tier 2, which catches aliases that resolve but are
bypassed by forward). Stub cfg only — no HF model load.
"""

from __future__ import annotations

from typing import Any, Iterable, Tuple

import pytest

from transformer_lens.config import TransformerBridgeConfig
from transformer_lens.factories.architecture_adapter_factory import (
    SUPPORTED_ARCHITECTURES,
)
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)


def _stub_cfg(architecture: str) -> TransformerBridgeConfig:
    """Minimal cfg for adapter instantiation; small values keep stubs cheap."""
    return TransformerBridgeConfig(
        d_model=64,
        d_head=16,
        n_layers=2,
        n_ctx=128,
        n_heads=4,
        d_vocab=1000,
        d_mlp=256,
        n_key_value_heads=4,
        default_prepend_bos=True,
        architecture=architecture,
    )


def _iter_components(
    root: Any, path: str = ""
) -> Iterable[Tuple[str, GeneralizedComponent]]:
    """Walk component_mapping recursively, yielding (dotted-path, component)."""
    if isinstance(root, dict):
        for name, comp in root.items():
            yield from _iter_components(comp, f"{path}.{name}" if path else name)
        return
    if isinstance(root, GeneralizedComponent):
        yield path, root
        for name, sub in (root.submodules or {}).items():
            yield from _iter_components(sub, f"{path}.{name}")


def _resolve(component: GeneralizedComponent, target: str) -> Any:
    """Resolve dotted alias using submodules dict — pre-model-load templates
    don't yet have submodules registered via add_module()."""
    obj: Any = component
    for part in target.split("."):
        nxt = None
        if isinstance(obj, GeneralizedComponent):
            nxt = (obj.submodules or {}).get(part)
        if nxt is None:
            nxt = getattr(obj, part, None)
        if nxt is None:
            raise AttributeError(part)
        obj = nxt
    return obj


# xfail(strict=True) so future fixes XPASS and force the marker to be removed.
# Each entry maps to a specific audit finding deferred from the C1+C15 PR.
_KNOWN_DEAD_ALIASES = {
    "GPT2LMHeadCustomModel": "audit H27 — stale adapter, delete candidate",
    "NanoGPTForCausalLM": "audit H28 — broken weight conversion, delete candidate",
    "NeelSoluOldForCausalLM": "audit H28 — orphan weight conversion, delete candidate",
    "LlavaForConditionalGeneration": "audit H15 — vision-encoder layer submodules unwired",
    "LlavaNextForConditionalGeneration": "audit H15 + M24 — vision encoder + tiling opaque",
    "LlavaOnevisionForConditionalGeneration": "audit H15 + M25 — vision encoder + video frames opaque",
    "Gemma3ForConditionalGeneration": "audit H15 — multimodal vision encoder opaque",
    "OpenELMForCausalLM": "audit H23 — per-layer head counts break uniform q/k/v shape",
    "GraniteMoeHybridForCausalLM": "new finding — MoE+shared-MLP block lacks proper submodule aliases",
}


def _architecture_params():
    """Parametrize list with xfail markers for known-dead-alias adapters."""
    params = []
    for arch in sorted(SUPPORTED_ARCHITECTURES):
        reason = _KNOWN_DEAD_ALIASES.get(arch)
        if reason is not None:
            params.append(pytest.param(arch, marks=pytest.mark.xfail(strict=True, reason=reason)))
        else:
            params.append(arch)
    return params


@pytest.mark.parametrize("architecture", _architecture_params())
def test_every_hook_alias_resolves_to_hookpoint(architecture: str) -> None:
    """Every declared hook_aliases entry must resolve to a HookPoint."""
    adapter_cls = SUPPORTED_ARCHITECTURES[architecture]
    try:
        adapter = adapter_cls(_stub_cfg(architecture))
    except Exception as exc:
        pytest.skip(f"Adapter {architecture} cannot instantiate with stub cfg: {exc}")

    mapping = adapter.component_mapping
    if mapping is None:
        pytest.skip(f"Adapter {architecture} has no component_mapping")

    failures: list[str] = []
    for path, component in _iter_components(mapping):
        for alias_name, target in component.hook_aliases.items():
            targets = target if isinstance(target, list) else [target]
            resolved = False
            for single in targets:
                try:
                    obj = _resolve(component, single)
                except AttributeError:
                    continue
                if isinstance(obj, HookPoint):
                    resolved = True
                    break
            if not resolved:
                failures.append(
                    f"{path}.{alias_name} -> {target} (type at path: unresolved)"
                )
    assert not failures, (
        f"Architecture {architecture}: {len(failures)} dead hook aliases:\n  "
        + "\n  ".join(failures)
    )
