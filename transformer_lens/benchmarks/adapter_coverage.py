"""Adapter coverage checks: whether an adapter maps and exposes the whole architecture.

``verify_models`` compares logits, which a bridge that delegates its forward to
HuggingFace reproduces even when its hooks describe a different computation. These
checks ask the complementary question on a path the logits cannot hide behind.
"""

from __future__ import annotations

import ast
import copy
import fnmatch
import inspect
import re
import textwrap
from collections import Counter
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd
from transformers import AutoConfig, PreTrainedConfig
from transformers.pytorch_utils import Conv1D

from transformer_lens.benchmarks.hook_registration import (
    benchmark_critical_forward_hooks,
    benchmark_forward_hooks,
    benchmark_gated_hooks_fire,
    benchmark_hook_functionality,
)
from transformer_lens.benchmarks.utils import (
    BenchmarkResult,
    BenchmarkSeverity,
    deterministic_rng,
)
from transformer_lens.hook_points import HookPoint
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    AltUpBlockBridge,
    AttentionBridge,
    AudioFeatureExtractorBridge,
    BertPoolerBridge,
    BlockBridge,
    CLIPVisionEncoderBridge,
    CLIPVisionEncoderLayerBridge,
    Conv1DBridge,
    ConvPosEmbedBridge,
    DepthwiseConv1DBridge,
    EmbeddingBridge,
    GatedDeltaNetBridge,
    GatedRMSNormBridge,
    JointGateUpMLPBridge,
    JointQKVAttentionBridge,
    Lfm2ShortConvBridge,
    LinearBridge,
    MLAAttentionBridge,
    MLPBridge,
    MoEBridge,
    MoERouterBridge,
    NormalizationBridge,
    OpaqueBlockBridge,
    PosEmbedBridge,
    Qwen3_5VisionBlockBridge,
    Qwen3_5VisionEncoderBridge,
    RotaryEmbeddingBridge,
    SiglipVisionEncoderBridge,
    SiglipVisionEncoderLayerBridge,
    SSM2MixerBridge,
    SSMMixerBridge,
    SymbolicBridge,
    T5BlockBridge,
    T5Gemma2DecoderBlockBridge,
    T5GemmaDecoderBlockBridge,
    UnembeddingBridge,
    VisionClassifierHeadBridge,
    VisionEmbeddingsBridge,
    VisionProjectionBridge,
)
from transformer_lens.model_bridge.generalized_components.base import (
    GeneralizedComponent,
)
from transformer_lens.model_bridge.sources import build_bridge_from_module
from transformer_lens.tools.analysis.logit_lens import _READOUT_TRANSFORM_COMPONENTS

DEFAULT_TEST_TEXT = "The capital of France is Paris, and the capital of Italy is Rome."

# A prompt as text, or as token ids for a bridge that has no tokenizer.
PromptInput = Union[str, torch.Tensor]

# HF module kind -> bridge components that may wrap a module of that kind directly.
# Keyed by kind, not HF class name, so an architecture nobody has seen still has a row.
MODULE_KIND_OWNERS: Dict[str, Tuple[type, ...]] = {
    "linear": (LinearBridge, Conv1DBridge, UnembeddingBridge, VisionClassifierHeadBridge),
    "embedding": (EmbeddingBridge, PosEmbedBridge, UnembeddingBridge),
    "norm": (NormalizationBridge, GatedRMSNormBridge),
    "conv": (DepthwiseConv1DBridge, ConvPosEmbedBridge),
    # Modules holding bare nn.Parameters (router weights, SSM dynamics, learned positions).
    "parameters": (
        AttentionBridge,
        MLPBridge,
        MoEBridge,
        MoERouterBridge,
        SSMMixerBridge,
        SSM2MixerBridge,
        Lfm2ShortConvBridge,
        GatedDeltaNetBridge,
        EmbeddingBridge,
        PosEmbedBridge,
        UnembeddingBridge,
        NormalizationBridge,
        VisionEmbeddingsBridge,
    ),
}

# Components that model the internals of the module they wrap (fused projections,
# expert banks, recurrences), so every parameter beneath them is theirs.
SUBTREE_OWNERS: Tuple[type, ...] = (
    JointQKVAttentionBridge,
    MLAAttentionBridge,
    JointGateUpMLPBridge,
    MoEBridge,
    GatedDeltaNetBridge,
    VisionProjectionBridge,
    VisionEmbeddingsBridge,
    AudioFeatureExtractorBridge,
    ConvPosEmbedBridge,
    BertPoolerBridge,
)

# Residual-stream blocks: a parameter with only one of these above it is unmapped,
# and one behind a generic wrapper inside them is a hole in the stack the logits come from.
RESIDUAL_BLOCK_COMPONENTS: Tuple[type, ...] = (
    BlockBridge,
    OpaqueBlockBridge,
    AltUpBlockBridge,
    T5BlockBridge,
    T5GemmaDecoderBlockBridge,
    T5Gemma2DecoderBlockBridge,
)

# Components that only hold other components and own nothing themselves.
CONTAINER_COMPONENTS: Tuple[type, ...] = RESIDUAL_BLOCK_COMPONENTS + (
    CLIPVisionEncoderLayerBridge,
    SiglipVisionEncoderLayerBridge,
    Qwen3_5VisionBlockBridge,
    SymbolicBridge,
)

# Components that expose only their own input and output for anything they do not wrap directly.
BOUNDARY_COMPONENTS: Tuple[type, ...] = (
    CLIPVisionEncoderBridge,
    SiglipVisionEncoderBridge,
    Qwen3_5VisionEncoderBridge,
    RotaryEmbeddingBridge,
)

_COMPONENT_PACKAGE = "transformer_lens.model_bridge.generalized_components"

# Correct adapters measure 0 to 6e-6 of the reference magnitude in fp32; the faults
# these checks exist for measure 1e-2 and up.
_RELATIVE_TOLERANCE: Dict[torch.dtype, float] = {
    torch.float64: 1e-6,
    torch.float32: 1e-4,
    torch.float16: 2e-2,
    torch.bfloat16: 2e-2,
}

# Substitutes tried for a multiplier sitting at its identity value.
_PERTURBED_VALUES: Tuple[float, ...] = (0.5, 2.0)


def _blocking(part: str, ok: bool, message: str = "") -> Dict[str, str]:
    """One blocking part that ran: a wrong answer here makes the adapter misleading."""
    return {"part": part, "status": "pass" if ok else "fail", "message": message}


def _gap(kind: str, where: str) -> Dict[str, str]:
    """Something the adapter does not expose; it lowers coverage and blocks nothing."""
    return {"kind": kind, "where": where}


def _outcome(
    name: str,
    summary: str,
    details: Optional[Dict[str, Any]] = None,
    blocking: Sequence[Dict[str, str]] = (),
    gaps: Sequence[Dict[str, str]] = (),
) -> BenchmarkResult:
    """Fail on a blocking finding and note gaps otherwise: misleading blocks, missing does not."""
    merged = dict(details or {})
    merged["blocking"] = list(blocking)
    merged["gaps"] = list(gaps)
    failed = [entry for entry in blocking if entry["status"] == "fail"]
    if failed:
        message = "; ".join(
            f"{e['part']}: {e['message']}" if e["message"] else e["part"] for e in failed
        )
        return BenchmarkResult(
            name=name,
            severity=BenchmarkSeverity.DANGER,
            message=message,
            details=merged,
            passed=False,
        )
    severity = BenchmarkSeverity.WARNING if gaps else BenchmarkSeverity.INFO
    note = f" ({len(gaps)} gap(s))" if gaps else ""
    return BenchmarkResult(name=name, severity=severity, message=summary + note, details=merged)


def _not_applicable(
    name: str, reason: str, details: Optional[Dict[str, Any]] = None
) -> BenchmarkResult:
    return BenchmarkResult(
        name=name, severity=BenchmarkSeverity.SKIPPED, message=reason, details=details
    )


def _could_not_run(name: str, error: BaseException | str) -> BenchmarkResult:
    """A check that cannot run certifies nothing, so it blocks instead of skipping."""
    text = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
    message = f"could not run: {text[:500]}"
    return BenchmarkResult(
        name=name,
        severity=BenchmarkSeverity.ERROR,
        message=message,
        details={"blocking": [_blocking("run", False, message)], "gaps": []},
        passed=False,
    )


def _tolerance(tensor: torch.Tensor) -> float:
    return _RELATIVE_TOLERANCE.get(tensor.dtype, 1e-4)


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> Tuple[float, float]:
    """Worst absolute difference and the magnitude it is judged against."""
    if actual.shape != expected.shape:
        return float("inf"), 0.0
    expected = expected.to(actual.device)
    scale = expected.abs().max().item()
    return (actual - expected).abs().max().item(), scale


def _within(error: float, scale: float, rtol: float) -> bool:
    return error <= rtol * max(scale, 1e-8)


@dataclass
class _Prompt:
    """One text prompt as token ids, with the extra inputs each side's forward needs."""

    tokens: torch.Tensor
    in_vocabulary: bool
    bridge_kwargs: Dict[str, Any]
    reference_kwargs: Dict[str, Any]


def _prompt(bridge: TransformerBridge, test_input: PromptInput) -> Optional[_Prompt]:
    """Token inputs for a text model, or None when the model takes no text."""
    takes_text = not any(
        getattr(bridge.cfg, flag, False)
        for flag in ("is_audio_model", "is_audio_seq2seq", "is_visual_model")
    )
    if not takes_text:
        return None
    config = bridge.original_model.config
    if isinstance(test_input, torch.Tensor):
        tokens = test_input
    elif getattr(bridge, "tokenizer", None) is None:
        return None
    else:
        tokens = bridge.to_tokens(test_input)
    in_vocabulary = True
    vocab_size = getattr(config, "vocab_size", None)
    # Tiny test checkpoints often pair a full tokenizer with a truncated embedding table.
    if isinstance(vocab_size, int) and vocab_size > 0 and int(tokens.max()) >= vocab_size:
        tokens, in_vocabulary = tokens.remainder(vocab_size), False
    if getattr(config, "is_encoder_decoder", False):
        return _Prompt(
            tokens, in_vocabulary, {"decoder_input": tokens}, {"decoder_input_ids": tokens}
        )
    return _Prompt(tokens, in_vocabulary, {}, {})


def _refit(prompt: _Prompt, model: nn.Module) -> _Prompt:
    """The same prompt with its token ids folded into ``model``'s embedding table."""
    tokens = _fit_vocabulary(prompt.tokens, model)
    return _Prompt(
        tokens,
        prompt.in_vocabulary,
        {key: tokens for key in prompt.bridge_kwargs},
        {key: tokens for key in prompt.reference_kwargs},
    )


_NEEDS_TEXT = "the model takes no text prompt"
_NON_FINITE = "HuggingFace itself returns non-finite logits on this checkpoint"


# --------------------------------------------------------------------------------------
# Walking the wrapped model
# --------------------------------------------------------------------------------------


def _iter_native_modules(
    root: nn.Module,
) -> Iterator[Tuple[str, nn.Module, Tuple[GeneralizedComponent, ...]]]:
    """Yield (HF path, module, bridge components above it) over the wrapped HF tree."""
    stack: List[Tuple[str, nn.Module, Tuple[GeneralizedComponent, ...]]] = [("", root, ())]
    while stack:
        path, module, owners = stack.pop()
        if isinstance(module, GeneralizedComponent):
            # Other children of a bridge component are bridge-side, not HF modules.
            inner = module._modules.get("_original_component")
            if inner is not None:
                stack.append((path, inner, owners + (module,)))
            continue
        yield path, module, owners
        for name, child in module._modules.items():
            if child is not None:
                stack.append((f"{path}.{name}" if path else name, child, owners))


def _iter_bridged_components(root: nn.Module) -> Iterator[Tuple[str, GeneralizedComponent]]:
    """Yield (HF path, bridge component) for every component spliced into the HF tree."""
    stack: List[Tuple[str, nn.Module]] = [("", root)]
    while stack:
        path, module = stack.pop()
        if isinstance(module, GeneralizedComponent):
            yield path, module
            inner = module._modules.get("_original_component")
            if inner is not None:
                stack.append((path, inner))
            continue
        for name, child in module._modules.items():
            if child is not None:
                stack.append((f"{path}.{name}" if path else name, child))


def module_kind(module: nn.Module) -> str:
    """Row of ``MODULE_KIND_OWNERS`` a parameter-bearing HF module belongs to."""
    if isinstance(module, (nn.Linear, Conv1D)):
        return "linear"
    if isinstance(module, nn.Embedding):
        return "embedding"
    # HF defines its RMSNorm variants per architecture; they share only the name.
    if isinstance(module, (nn.LayerNorm, nn.GroupNorm, _BatchNorm)) or (
        "Norm" in type(module).__name__
    ):
        return "norm"
    if isinstance(module, _ConvNd):
        return "conv"
    return "parameters"


def _library_class(component: GeneralizedComponent) -> type:
    """Nearest library component class, so an adapter-local subclass keeps its parent's role."""
    for cls in type(component).__mro__:
        if cls.__module__.startswith(_COMPONENT_PACKAGE):
            return cls
    return GeneralizedComponent


def _ownership(module: nn.Module, owners: Tuple[GeneralizedComponent, ...]) -> str:
    """``specific``, ``generic`` or ``unmapped`` for a parameter-bearing module."""
    if not owners:
        return "unmapped"
    if any(issubclass(_library_class(o), SUBTREE_OWNERS) for o in owners):
        return "specific"
    owner = owners[-1]
    owner_class = _library_class(owner)
    if issubclass(owner_class, CONTAINER_COMPONENTS):
        return "unmapped"
    wraps_directly = owner._modules.get("_original_component") is module
    if wraps_directly and issubclass(owner_class, MODULE_KIND_OWNERS[module_kind(module)]):
        return "specific"
    return "generic"


def _collapse_indices(path: str) -> str:
    return re.sub(r"(?<=\.)\d+(?=\.|$)", "{i}", path)


# --------------------------------------------------------------------------------------
# A. Structural coverage
# --------------------------------------------------------------------------------------

_OWNERSHIP_RANK = {"unmapped": 0, "generic": 1, "specific": 2}


def _graded_modules(bridge: TransformerBridge) -> List[Dict[str, Any]]:
    """Every parameter-bearing HF module with its ownership grade and where it sits."""
    located = []
    best: Dict[int, int] = {}
    for path, module, owners in _iter_native_modules(bridge.original_model):
        parameters = [p for p in module._parameters.values() if p is not None]
        if not parameters:
            continue
        grade = _ownership(module, owners)
        located.append((path, module, owners, parameters, grade))
        for parameter in parameters:
            best[id(parameter)] = max(best.get(id(parameter), 0), _OWNERSHIP_RANK[grade])
    names = {rank: grade for grade, rank in _OWNERSHIP_RANK.items()}
    graded = []
    for path, module, owners, parameters, _ in located:
        # A tied or shared tensor is reachable by several paths; its best owner counts.
        rank = min(best[id(parameter)] for parameter in parameters)
        block = next((o for o in owners if isinstance(o, RESIDUAL_BLOCK_COMPONENTS)), None)
        graded.append(
            {
                "path": path,
                "grade": names[rank],
                "block": block,
                "kind": module_kind(module),
                "owner": type(owners[-1]).__name__ if owners else "none",
                "parameters": parameters,
            }
        )
    return graded


def benchmark_structural_coverage(bridge: TransformerBridge) -> BenchmarkResult:
    """Which HF parameters the bridge component for their module kind owns, and which it does not."""
    name = "structural_coverage"
    try:
        graded = _graded_modules(bridge)
        sizes: Dict[int, Tuple[int, str]] = {}
        for entry in graded:
            for parameter in entry["parameters"]:
                sizes[id(parameter)] = (parameter.numel(), entry["grade"])
        total = sum(size for size, _ in sizes.values()) or 1
        share = {
            grade: sum(size for size, own in sizes.values() if own == grade) / total
            for grade in _OWNERSHIP_RANK
        }
        # A pattern is as well owned as its worst instance (a layer may lack what others map).
        patterns: Dict[str, int] = {}
        in_block: Dict[str, bool] = {}
        for entry in graded:
            pattern = _collapse_indices(entry["path"])
            rank = _OWNERSHIP_RANK[entry["grade"]]
            patterns[pattern] = min(patterns.get(pattern, rank), rank)
            in_block[pattern] = in_block.get(pattern, False) or entry["block"] is not None
        specific = sum(1 for rank in patterns.values() if rank == _OWNERSHIP_RANK["specific"])
        unmapped = sorted(p for p, rank in patterns.items() if rank == _OWNERSHIP_RANK["unmapped"])
        generic = sorted(p for p, rank in patterns.items() if rank == _OWNERSHIP_RANK["generic"])
        details: Dict[str, Any] = {
            "pattern_coverage": {
                "specific": specific,
                "total": len(patterns),
                "share": round(specific / len(patterns), 4) if patterns else None,
            },
            "parameter_share": {k: round(v, 4) for k, v in share.items()},
            "unmapped_modules": unmapped,
            "generic_modules_in_blocks": [p for p in generic if in_block[p]],
            "generic_modules_elsewhere": [p for p in generic if not in_block[p]],
            "ownership": dict(
                Counter(
                    f"{e['kind']} under {e['owner']}: {e['grade']}" for e in graded
                ).most_common()
            ),
        }
        gaps = [_gap("unmapped", p) for p in unmapped] + [_gap("generic", p) for p in generic]
        summary = f"{specific} of {len(patterns)} module path patterns are owned by the component for their kind"
        return _outcome(name, summary, details, gaps=gaps)
    except Exception as error:
        return _could_not_run(name, error)


# --------------------------------------------------------------------------------------
# B. Expected hooks exist and fire
# --------------------------------------------------------------------------------------


def _resolve(component: nn.Module, target: str) -> Optional[HookPoint]:
    current: Any = component
    for part in target.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current if isinstance(current, HookPoint) else None


def _resolve_alias(component: nn.Module, target: Any) -> Optional[HookPoint]:
    for candidate in target if isinstance(target, list) else [target]:
        hook = _resolve(component, candidate)
        if hook is not None:
            return hook
    return None


def _named_components(
    root: nn.Module, prefix: str = ""
) -> Iterator[Tuple[str, GeneralizedComponent]]:
    """Every bridge component beneath ``root``, once, under its TransformerLens name.

    A component is registered twice, under its TransformerLens name and spliced into the
    wrapped HF module; ``named_modules`` reports whichever it meets first, which is the
    HF-side path, so the TransformerLens-side children are walked first here.
    """
    seen: set = set()

    def walk(module: nn.Module, path: str) -> Iterator[Tuple[str, GeneralizedComponent]]:
        children = sorted(
            module._modules.items(), key=lambda item: item[0] == "_original_component"
        )
        for child_name, child in children:
            if child is None or id(child) in seen:
                continue
            seen.add(id(child))
            child_path = f"{path}.{child_name}" if path else child_name
            if isinstance(child, GeneralizedComponent) and "_original_component" not in child_path:
                yield child_path, child
            yield from walk(child, child_path)

    yield from walk(root, prefix)


def _bridge_components(bridge: TransformerBridge) -> Iterator[Tuple[str, GeneralizedComponent]]:
    return _named_components(bridge)


def _unresolved_aliases(bridge: TransformerBridge) -> List[str]:
    """Aliases a component's class promises that point at nothing on this model."""
    return [
        f"{module_name}.{alias} -> {target}"
        for module_name, module in _bridge_components(bridge)
        for alias, target in (module.hook_aliases or {}).items()
        if _resolve_alias(module, target) is None
    ]


def _fired_hook_ids(bridge: TransformerBridge, prompt: _Prompt) -> set:
    """Identities of the HookPoints that fire on one forward pass."""
    fired: set = set()
    points = {id(point): point for point in bridge.hook_dict.values()}
    try:
        for key, point in points.items():
            point.add_hook(lambda _tensor, hook, key=key: fired.add(key))
        with torch.no_grad():
            bridge(prompt.tokens, **prompt.bridge_kwargs)
    finally:
        for point in points.values():
            point.remove_hooks()
    return fired


# Config flags that switch on hooks which are silent by default, and their setters.
_HOOK_FLAGS: Dict[str, Tuple[str, ...]] = {
    "use_attn_in": ("hook_attn_in",),
    "use_split_qkv_input": ("hook_q_input", "hook_k_input", "hook_v_input"),
    "use_hook_mlp_in": ("hook_mlp_in",),
    "use_attn_result": ("hook_result",),
}


def _fired_under_flags(bridge: TransformerBridge, prompt: _Prompt) -> Tuple[set, List[str]]:
    """Hooks that fire with each gating flag on, and the flags this bridge refuses."""
    fired: set = set()
    refused: List[str] = []
    for flag in _HOOK_FLAGS:
        setter = getattr(bridge, f"set_{flag}", None)
        if setter is None:
            refused.append(flag)
            continue
        try:
            setter(True)
            fired |= _fired_hook_ids(bridge, prompt)
        except (NotImplementedError, ValueError):
            # The attention bridge in use has no such fork; nothing to promise.
            refused.append(flag)
        finally:
            try:
                setter(False)
            except (NotImplementedError, ValueError):
                pass
    return fired, refused


def _collapse_set(names: Any) -> set:
    return {_collapse_indices(n) for n in names}


def _classify_reused(
    result: BenchmarkResult,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], str]:
    """Sort one reused hook benchmark's outcome into blocking parts and gaps, with the reason."""
    if result.severity == BenchmarkSeverity.ERROR:
        return (
            [_blocking(result.name, False, result.message)],
            [],
            "blocking: the benchmark could not run on this bridge",
        )
    if result.name == "gated_hooks_fire" and not result.passed:
        failed = (result.details or {}).get("failed", [])
        gaps = [_gap("silent_hook", f"{stem} (with {flag} on)") for flag, stem in failed]
        return [_blocking(result.name, True)], gaps, "gap: the hooks exist and stay silent"
    if not result.passed:
        return (
            [_blocking(result.name, False, result.message)],
            [],
            "blocking: a hook that is present gave a wrong result",
        )
    return [_blocking(result.name, True)], [], "ran without a blocking finding"


def benchmark_expected_hooks(
    bridge: TransformerBridge, test_input: PromptInput = DEFAULT_TEST_TEXT
) -> BenchmarkResult:
    """The hooks the bridge classes in use promise exist, and fire on a forward pass."""
    name = "expected_hooks"
    try:
        prompt = _prompt(bridge, test_input)
        if prompt is None:
            return _not_applicable(name, _NEEDS_TEXT)
        unresolved = _unresolved_aliases(bridge)
        fired = _fired_hook_ids(bridge, prompt)
        fired_flagged, refused_flags = _fired_under_flags(bridge, prompt)
        refused_stems = {stem for flag in refused_flags for stem in _HOOK_FLAGS[flag]}

        # A block class promises its residual-stream aliases; one that resolves but
        # never fires on a block that ran is a hook an intervention silently misses.
        silent_promised: List[str] = []
        idle_blocks = 0
        for module_name, module in _bridge_components(bridge):
            if not isinstance(module, RESIDUAL_BLOCK_COMPONENTS):
                continue
            if id(module.hook_in) not in fired:
                idle_blocks += 1
                continue
            for alias, target in (module.hook_aliases or {}).items():
                hook = _resolve_alias(module, target)
                if hook is None or alias in refused_stems:
                    continue
                if id(hook) not in fired and id(hook) not in fired_flagged:
                    silent_promised.append(f"{module_name}.{alias}")

        blocking: List[Dict[str, str]] = []
        gaps = [
            _gap("absent_hook", alias.split(" -> ")[0])
            for alias in sorted(_collapse_set(unresolved))
        ]
        gaps += [_gap("silent_hook", hook) for hook in sorted(_collapse_set(silent_promised))]
        reused: Dict[str, Dict[str, str]] = {}
        # The reused benchmarks tokenize the text themselves, which a truncated vocabulary rejects.
        if isinstance(test_input, str) and prompt.in_vocabulary and not prompt.bridge_kwargs:
            results = [
                benchmark_forward_hooks(bridge, test_input),
                benchmark_critical_forward_hooks(bridge, test_input),
                benchmark_gated_hooks_fire(bridge, test_input),
            ]
            # The ablation benchmark targets block 0's attention value hook, which no block
            # class promises: a state-space block has none, and that is not a gap.
            if "blocks.0.attn.hook_v" in bridge.hook_dict:
                results.append(benchmark_hook_functionality(bridge, test_input))
            else:
                reused["hook_functionality"] = {
                    "message": "not run",
                    "classified": "not applicable: block 0 has no attention value hook",
                }
            for result in results:
                parts, found, reason = _classify_reused(result)
                blocking += parts
                # The block-level aliases above already name these hooks layer by layer.
                gaps += [
                    gap
                    for gap in found
                    if not any(
                        known["where"].endswith("." + gap["where"].split(" ")[0]) for known in gaps
                    )
                ]
                reused[result.name] = {"message": result.message, "classified": reason}

        hook_names = bridge.hook_dict
        silent_other = {
            hook_name
            for hook_name, point in hook_names.items()
            if id(point) not in fired and id(point) not in fired_flagged
        }
        details: Dict[str, Any] = {
            "registered": len(hook_names),
            "fired": sum(1 for point in hook_names.values() if id(point) in fired),
            "unresolved_aliases": sorted(_collapse_set(unresolved)),
            "silent_promised_hooks": sorted(_collapse_set(silent_promised)),
            "refused_flags": refused_flags,
            "blocks_not_run_by_a_text_prompt": idle_blocks,
            "other_silent_hooks": sorted(_collapse_set(silent_other))[:60],
            "reused_benchmarks": reused,
        }
        summary = "every promised block hook resolves and fires on the blocks that ran"
        if gaps:
            summary = "some promised hooks are absent or silent"
        return _outcome(name, summary, details, blocking, gaps)
    except Exception as error:
        return _could_not_run(name, error)


# --------------------------------------------------------------------------------------
# D. Hook consistency
# --------------------------------------------------------------------------------------


def build_reference_model(bridge: TransformerBridge) -> nn.Module:
    """A pristine HF twin of the model the bridge wraps, with the same weights."""
    wrapped = bridge.original_model
    state: Dict[str, torch.Tensor] = {}
    for path, module, _ in _iter_native_modules(wrapped):
        prefix = f"{path}." if path else ""
        for param_name, parameter in module._parameters.items():
            if parameter is not None:
                state[prefix + param_name] = parameter.detach()
        for buffer_name, buffer in module._buffers.items():
            if buffer is not None and buffer_name not in module._non_persistent_buffers_set:
                state[prefix + buffer_name] = buffer.detach()
    with deterministic_rng(0):
        try:
            reference = type(wrapped)(copy.deepcopy(wrapped.config))
        except Exception:
            # Some models consume their config while building (timm towers pop block
            # arguments), so the live config cannot build a second model; reload it.
            config = AutoConfig.from_pretrained(
                bridge.cfg.model_name,
                trust_remote_code=bool(getattr(bridge.cfg, "trust_remote_code", False)),
            )
            reference = type(wrapped)(config)
    missing, _ = reference.load_state_dict(state, strict=False)
    parameter_names = {n for n, _ in reference.named_parameters(remove_duplicate=False)}
    lost = sorted(parameter_names.intersection(missing))
    if lost:
        raise RuntimeError(f"reference model is missing weights the bridge holds: {lost[:5]}")
    first = next(wrapped.parameters())
    return reference.to(device=first.device, dtype=first.dtype).eval()


def _reference_logits(reference: nn.Module, prompt: _Prompt) -> torch.Tensor:
    output = reference(prompt.tokens, **prompt.reference_kwargs)
    logits = getattr(output, "logits", None)
    return logits if isinstance(logits, torch.Tensor) else output[0]


def _check_parity(logits: torch.Tensor, reference_logits: torch.Tensor) -> Dict[str, Any]:
    error, scale = _relative_error(logits, reference_logits)
    return {
        "status": "pass" if _within(error, scale, _tolerance(logits)) else "fail",
        "max_abs_error": error,
        "reference_scale": scale,
    }


def _close(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    error, scale = _relative_error(actual, expected)
    return _within(error, scale, _tolerance(actual))


def _mlp_term_is_derived(bridge: TransformerBridge, cache: Any, layer: int) -> bool:
    """Whether the library reads this block's MLP term off the stream instead of a hook.

    It does so for a block with an MLP-side write and no ``hook_mlp_out``. A block whose
    MLP is its one passthrough mixer has no ``hook_mlp_out`` either, but its term is the
    mixer's hooked output.
    """
    return f"blocks.{layer}.hook_mlp_out" not in bridge.hook_dict and bool(cache._has_mlp(layer))


def _normalised_after_add(
    bridge: TransformerBridge, cache: Any, layer: int, terms: List[torch.Tensor]
) -> bool:
    """Whether the block norms each sum (post-norm), leaving no additive decomposition."""
    prefix = f"blocks.{layer}"
    norms = []
    for sub_name, sub in _named_components(bridge.blocks[layer], prefix):
        if issubclass(_library_class(sub), NormalizationBridge):
            entered = cache.cache_dict.get(f"{sub_name}.hook_in")
            left = cache.cache_dict.get(f"{sub_name}.hook_out")
            if entered is not None and left is not None:
                norms.append((entered, left))
    stream = cache[f"{prefix}.hook_resid_pre"]
    normed = False
    for term in terms:
        total = stream + term
        after = next((left for entered, left in norms if _close(total, entered)), None)
        stream, normed = (after, True) if after is not None else (total, normed)
    return normed and _close(stream, cache[f"{prefix}.hook_resid_post"])


def _check_residual_sum(bridge: TransformerBridge, cache: Any) -> Dict[str, Any]:
    """The library's own residual decomposition adds up to the stream at every layer."""
    n_layers = int(getattr(bridge.cfg, "n_layers", 0) or 0)
    if not hasattr(bridge, "blocks") or n_layers == 0:
        return {"status": "skip", "reason": "the bridge has no `blocks` residual stream"}
    if f"blocks.{n_layers - 1}.hook_resid_post" not in cache:
        return {"status": "fail", "reason": "the last block exposes no hook_resid_post"}
    try:
        components, labels = cache.decompose_resid(layer=-1, return_labels=True)
    except KeyError as error:
        return {"status": "fail", "reason": f"decomposition needs a hook that is absent: {error}"}
    by_layer: Dict[int, List[torch.Tensor]] = {}
    running = torch.zeros_like(components[0])
    for index, label in enumerate(labels):
        layer = label.split("_", 1)[0]
        if layer.isdigit():
            by_layer.setdefault(int(layer), []).append(components[index])
        else:
            running = running + components[index]
    worst_error, worst_scale = 0.0, 0.0
    failing: List[int] = []
    for layer in sorted(by_layer):
        key = f"blocks.{layer}.hook_resid_post"
        if key not in cache:
            continue
        for term in by_layer[layer]:
            running = running + term
        gap, scale = _relative_error(running, cache[key])
        if not _within(gap, scale, _tolerance(running)):
            failing.append(layer)
        if gap / max(scale, 1e-8) >= worst_error / max(worst_scale, 1e-8):
            worst_error, worst_scale = gap, scale
    if failing and all(
        _normalised_after_add(bridge, cache, layer, by_layer[layer]) for layer in sorted(by_layer)
    ):
        return {
            "status": "skip",
            "reason": "each block norms the stream after adding to it, so it has no additive decomposition",
        }
    # A derived term is the stream's own difference, so it cannot disagree with the stream.
    derived = sum(1 for layer in range(n_layers) if _mlp_term_is_derived(bridge, cache, layer))
    return {
        "status": "fail" if failing else "pass",
        "max_abs_error": worst_error,
        "reference_scale": worst_scale,
        "first_failing_layer": failing[0] if failing else None,
        "terms": len(labels),
        "layers_with_derived_mlp_term": derived,
    }


def _check_logit_readout(
    bridge: TransformerBridge, cache: Any, logits: torch.Tensor
) -> Dict[str, Any]:
    """The final residual, through the final norm and unembedding, gives the logits."""
    n_layers = int(getattr(bridge.cfg, "n_layers", 0) or 0)
    key = f"blocks.{n_layers - 1}.hook_resid_post"
    unembed = getattr(bridge, "unembed", None)
    if key not in cache or unembed is None:
        return {"status": "skip", "reason": "no final residual hook or no unembedding"}
    # Heads the library's own logit lens refuses to read through, for the same reason.
    mounted = [name for name in _READOUT_TRANSFORM_COMPONENTS if hasattr(bridge, name)]
    if mounted:
        return {"status": "skip", "reason": f"the readout passes through {mounted}"}
    residual = cache[key]
    ln_final = getattr(bridge, "ln_final", None)
    with torch.no_grad():
        normalized = ln_final(residual) if ln_final is not None else residual
        rebuilt = bridge.adapter.apply_output_logits_transform(unembed(normalized))
    if rebuilt.shape != logits.shape:
        return {"status": "fail", "reason": f"rebuilt logits have shape {tuple(rebuilt.shape)}"}
    error = (rebuilt - logits).abs().max().item()
    spread = (logits - logits.mean(-1, keepdim=True)).abs().max().item()
    return {
        "status": "pass" if _within(error, spread, _tolerance(logits)) else "fail",
        "max_abs_error": error,
        "reference_scale": spread,
    }


def _first_tensor(output: Any) -> Optional[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    hidden = getattr(output, "last_hidden_state", None)
    return hidden if isinstance(hidden, torch.Tensor) else None


def _halve_output(_module: nn.Module, _args: Any, output: Any) -> Any:
    if isinstance(output, torch.Tensor):
        return output * 0.5
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return (output[0] * 0.5,) + output[1:]
    if isinstance(getattr(output, "last_hidden_state", None), torch.Tensor):
        output.last_hidden_state = output.last_hidden_state * 0.5
    return output


def _intervention_targets(bridge: TransformerBridge, limit: int) -> List[Tuple[str, str]]:
    """(bridge hook name, HF module path) pairs, sampled across the distinct layer layouts."""
    names: Dict[int, str] = {}
    for hook_name, hook in bridge._hook_registry.items():
        names.setdefault(id(hook), hook_name)
    # Keyed by the layer's HF path ("model.layers.3"); "" holds components outside any layer.
    by_layer: Dict[str, List[Tuple[str, str]]] = {}
    layouts: Dict[str, Tuple[str, ...]] = {}
    for path, component in _iter_bridged_components(bridge.original_model):
        output_hook = names.get(id(component.hook_out))
        if output_hook is None or not path:
            continue
        match = re.match(r"^(.*?\.\d+)(?:\.|$)", path)
        layer = match.group(1) if match else ""
        by_layer.setdefault(layer, []).append((output_hook, path))
        layouts[layer] = layouts.get(layer, ()) + (type(component).__name__,)
    chosen: List[Tuple[str, str]] = list(by_layer.pop("", []))
    seen_layouts: set = set()
    layers = list(by_layer)
    for position, layer in enumerate(layers):
        prefix = re.sub(r"\.\d+$", "", layer)
        layout = (prefix, layouts[layer])
        is_last = all(not other.startswith(prefix + ".") for other in layers[position + 1 :])
        if layout in seen_layouts and not is_last:
            continue
        seen_layouts.add(layout)
        chosen.extend(by_layer[layer])
    if len(chosen) > limit:
        step = len(chosen) / limit
        chosen = [chosen[int(i * step)] for i in range(limit)]
    return chosen


def _check_interventions(
    bridge: TransformerBridge,
    reference: nn.Module,
    prompt: _Prompt,
    cache: Any,
    reference_logits: torch.Tensor,
    limit: int,
) -> Dict[str, Any]:
    """Editing a component's output hook matches the same edit on the HF module."""
    targets = _intervention_targets(bridge, limit)
    reference_modules = dict(reference.named_modules())
    clean: Dict[str, torch.Tensor] = {}
    handles = []
    for _, path in targets:
        module = reference_modules.get(path)
        if module is None:
            continue

        def capture(_module: nn.Module, _args: Any, output: Any, path: str = path) -> None:
            tensor = _first_tensor(output)
            if tensor is not None:
                clean[path] = tensor.detach()

        handles.append(module.register_forward_hook(capture))
    try:
        with torch.no_grad():
            _reference_logits(reference, prompt)
    finally:
        for handle in handles:
            handle.remove()

    rtol = _tolerance(reference_logits)
    scale = reference_logits.abs().max().item()
    compared, inert, mismatched = 0, 0, []
    for hook_name, path in targets:
        hooked = cache.cache_dict.get(hook_name)
        raw = clean.get(path)
        # Only a hook that carries the module's own output can be edited on both sides alike.
        if hooked is None or raw is None or not raw.is_floating_point():
            continue
        if hooked.numel() != raw.numel() or not torch.allclose(
            hooked.reshape(-1).float(), raw.reshape(-1).float().to(hooked.device), atol=1e-6
        ):
            continue
        handle = reference_modules[path].register_forward_hook(_halve_output)
        try:
            with torch.no_grad():
                expected = _reference_logits(reference, prompt)
                actual = bridge.run_with_hooks(
                    prompt.tokens,
                    fwd_hooks=[(hook_name, lambda tensor, hook: tensor * 0.5)],
                    **prompt.bridge_kwargs,
                )
        finally:
            handle.remove()
        compared += 1
        effect = (expected - reference_logits).abs().max().item()
        if effect <= 1e-6 * max(scale, 1e-8):
            inert += 1
        error, _ = _relative_error(actual, expected)
        if not _within(error, scale, rtol):
            mismatched.append({"hook": hook_name, "hf_module": path, "max_abs_error": error})
    if compared == 0:
        return {"status": "skip", "reason": "no hook carries a module's own output"}
    # Halving every sampled output and moving nothing means the hooks are off the logit path.
    status = "fail" if mismatched or inert == compared else "pass"
    return {
        "status": status,
        "compared": compared,
        "without_effect": inert,
        "mismatched": mismatched[:10],
        "reference_scale": scale,
    }


def _consistency_parts(
    bridge: TransformerBridge,
    reference: nn.Module,
    prompt: _Prompt,
    max_interventions: int,
) -> Dict[str, Dict[str, Any]]:
    with torch.no_grad():
        reference_logits = _reference_logits(reference, prompt)
        if not torch.isfinite(reference_logits).all():
            raise RuntimeError(_NON_FINITE)
        logits, cache = bridge.run_with_cache(prompt.tokens, **prompt.bridge_kwargs)
    return {
        "logit_parity": _check_parity(logits, reference_logits),
        "residual_sum": _check_residual_sum(bridge, cache),
        "logit_readout": _check_logit_readout(bridge, cache, logits),
        "interventions": _check_interventions(
            bridge, reference, prompt, cache, reference_logits, max_interventions
        ),
    }


def _failed_parts(parts: Dict[str, Dict[str, Any]]) -> List[str]:
    return [part for part, outcome in parts.items() if outcome["status"] == "fail"]


def _part_message(part: str, outcome: Dict[str, Any]) -> str:
    if outcome["status"] != "fail":
        return ""
    if "reason" in outcome:
        return str(outcome["reason"])
    if part == "interventions":
        mismatched = outcome.get("mismatched", [])
        if not mismatched:
            return "no sampled hook edit moves the logits"
        return f"{len(mismatched)} hook edit(s) differ from the same edit on the HuggingFace module, e.g. {mismatched[0]['hook']}"
    return f"off by {outcome.get('max_abs_error', float('nan')):.3g} on a scale of {outcome.get('reference_scale', float('nan')):.3g}"


def _blocking_parts(parts: Dict[str, Dict[str, Any]], prefix: str = "") -> List[Dict[str, str]]:
    """The consistency parts that ran, as blocking entries."""
    return [
        _blocking(prefix + part, outcome["status"] == "pass", _part_message(part, outcome))
        for part, outcome in parts.items()
        if outcome["status"] != "skip"
    ]


def _variant_blocking(label: str, outcome: Dict[str, Any]) -> List[Dict[str, str]]:
    """Blocking entries for one rebuilt variant; a bridge that will not boot on it blocks too."""
    if "parts" not in outcome:
        return [_blocking(f"{label}: boot", False, outcome.get("error", ""))]
    return _blocking_parts(outcome["parts"], f"{label}: ")


def benchmark_hook_consistency(
    bridge: TransformerBridge,
    reference_model: nn.Module,
    test_input: PromptInput = DEFAULT_TEST_TEXT,
    max_interventions: int = 32,
) -> BenchmarkResult:
    """The hooks describe the computation that produces the logits."""
    name = "hook_consistency"
    try:
        prompt = _prompt(bridge, test_input)
        if prompt is None:
            return _not_applicable(name, _NEEDS_TEXT)
        parts = _consistency_parts(bridge, reference_model, prompt, max_interventions)
        blocking = _blocking_parts(parts)
        if not blocking:
            return _not_applicable(name, "no consistency identity applies to this bridge", parts)
        ran = ", ".join(entry["part"] for entry in blocking)
        return _outcome(name, f"holds: {ran}", dict(parts), blocking)
    except Exception as error:
        return _could_not_run(name, error)


# --------------------------------------------------------------------------------------
# Residual floor
# --------------------------------------------------------------------------------------


def benchmark_residual_floor(
    bridge: TransformerBridge, test_input: PromptInput = DEFAULT_TEST_TEXT
) -> BenchmarkResult:
    """Every block's change to the residual stream is accounted for by hooked outputs."""
    name = "residual_floor"
    try:
        prompt = _prompt(bridge, test_input)
        if prompt is None:
            return _not_applicable(name, _NEEDS_TEXT)
        n_layers = int(getattr(bridge.cfg, "n_layers", 0) or 0)
        if not hasattr(bridge, "blocks") or n_layers == 0:
            return _not_applicable(name, "the bridge has no `blocks` residual stream")
        with torch.no_grad():
            logits, cache = bridge.run_with_cache(prompt.tokens, **prompt.bridge_kwargs)
        if isinstance(logits, torch.Tensor) and not torch.isfinite(logits).all():
            return _could_not_run(name, "the model returns non-finite logits on this checkpoint")
        by_layer: Dict[int, List[torch.Tensor]] = {}
        derived: set = set()
        absent: Dict[int, str] = {}
        # The library's own plan, read group by group so one block's missing hook does not
        # hide the blocks that do account for themselves.
        for labels, read in cache._resid_component_plan(n_layers):
            layers = {int(label.partition("_")[0]) for label in labels if label[:1].isdigit()}
            try:
                tensors = read()
            except KeyError as error:
                for layer in layers:
                    absent[layer] = f"the decomposition needs a hook that is absent: {error}"
                continue
            for label, tensor in zip(labels, tensors):
                layer_text, _, term = label.partition("_")
                if not layer_text.isdigit():
                    continue
                by_layer.setdefault(int(layer_text), []).append(tensor)
                if term == "mlp_out" and _mlp_term_is_derived(bridge, cache, int(layer_text)):
                    derived.add(int(layer_text))

        reasons: Dict[str, List[int]] = {}
        for layer in range(n_layers):
            prefix = f"blocks.{layer}"
            if f"{prefix}.hook_resid_pre" not in cache or f"{prefix}.hook_resid_post" not in cache:
                reasons.setdefault(
                    "hook_resid_pre or hook_resid_post is absent or silent", []
                ).append(layer)
            elif layer in absent:
                reasons.setdefault(absent[layer], []).append(layer)
            elif layer in derived:
                reasons.setdefault(
                    "the MLP term is inferred from the stream, not read from a hook", []
                ).append(layer)
            else:
                terms = by_layer.get(layer, [])
                total = cache[f"{prefix}.hook_resid_pre"]
                for term in terms:
                    total = total + term
                if not _close(total, cache[f"{prefix}.hook_resid_post"]) and not (
                    _normalised_after_add(bridge, cache, layer, terms)
                ):
                    reasons.setdefault(
                        "the change in the stream is not accounted for by hooked outputs", []
                    ).append(layer)
        failing = sorted({layer for layers in reasons.values() for layer in layers})
        details: Dict[str, Any] = {"blocks": n_layers, "failing_blocks": reasons}
        if not failing:
            return _outcome(
                name,
                f"all {n_layers} blocks account for their change in the stream",
                details,
                [_blocking("floor", True)],
            )
        # Name what to wrap: modules in the failing blocks with no bridge component of their own.
        failing_blocks = {id(bridge.blocks[layer]) for layer in failing}
        block_paths = {
            id(component): path
            for path, component in _iter_bridged_components(bridge.original_model)
        }
        unhooked: set = set()
        to_wrap: set = set()
        for entry in _graded_modules(bridge):
            block = entry["block"]
            if block is None or id(block) not in failing_blocks or entry["grade"] != "unmapped":
                continue
            unhooked.add(_collapse_indices(entry["path"]))
            relative = entry["path"][len(block_paths.get(id(block), "")) :].strip(".")
            child = relative.split(".", 1)[0]
            if child:
                to_wrap.add(_collapse_indices(f"{block_paths.get(id(block), '')}.{child}"))
        details["modules_without_output_hook"] = sorted(unhooked)
        details["wrap"] = sorted(to_wrap)
        reason = next(iter(reasons))
        message = f"{len(failing)} of {n_layers} blocks: {reason}"
        if to_wrap:
            message += f"; no output hook on {', '.join(sorted(to_wrap)[:6])}"
        return _outcome(name, "", details, [_blocking("floor", False, message)])
    except Exception as error:
        return _could_not_run(name, error)


# --------------------------------------------------------------------------------------
# Variant models for C and E
# --------------------------------------------------------------------------------------

BootVariant = Callable[[nn.Module], TransformerBridge]


@dataclass
class _ConfigField:
    """A config attribute, possibly on a nested sub-config such as ``text_config``."""

    path: str
    value: Any
    owner: Any

    @property
    def name(self) -> str:
        return self.path.rsplit(".", 1)[-1]


# Attributes every HF config carries (architectures, label maps, ...), never architecture knobs.
_GENERIC_CONFIG_FIELDS = frozenset(vars(PreTrainedConfig()))


def _config_fields(config: Any) -> Iterator[_ConfigField]:
    for name, value in vars(config).items():
        if name.startswith("_") or name in _GENERIC_CONFIG_FIELDS:
            continue
        if hasattr(value, "to_dict") and hasattr(value, "__dict__"):
            for sub_name, sub_value in vars(value).items():
                if not sub_name.startswith("_") and sub_name not in _GENERIC_CONFIG_FIELDS:
                    yield _ConfigField(f"{name}.{sub_name}", sub_value, value)
        else:
            yield _ConfigField(name, value, config)


def _edited_config(config: Any, edits: Dict[str, Any]) -> Any:
    edited = copy.deepcopy(config)
    for path, value in edits.items():
        target = edited
        *parents, leaf = path.split(".")
        for parent in parents:
            target = getattr(target, parent)
        setattr(target, leaf, value)
    return edited


def _build_variant(reference: nn.Module, edits: Dict[str, Any]) -> nn.Module:
    """The reference architecture rebuilt from an edited config, keeping every weight that still fits."""
    config = _edited_config(reference.config, edits)
    with deterministic_rng(0):
        variant = type(reference)(config)
    own = variant.state_dict()
    kept = {
        key: value
        for key, value in reference.state_dict().items()
        if key in own and own[key].shape == value.shape
    }
    variant.load_state_dict(kept, strict=False)
    first = next(reference.parameters())
    return variant.to(device=first.device, dtype=first.dtype).eval()


def default_variant_booter(bridge: TransformerBridge, **boot_kwargs: Any) -> BootVariant:
    """Boot variants the way the bridge under test was built, reusing its tokenizer."""

    def boot(hf_model: nn.Module) -> TransformerBridge:
        architecture = bridge.cfg.architecture
        if getattr(bridge, "tokenizer", None) is None and architecture is not None:
            # A bridge built around a local module has no checkpoint name to boot from.
            return build_bridge_from_module(
                hf_model, architecture, hf_config=copy.deepcopy(hf_model.config)
            ).eval()
        return TransformerBridge.boot_transformers(
            bridge.cfg.model_name, hf_model=hf_model, tokenizer=bridge.tokenizer, **boot_kwargs
        )

    return boot


def _run_variant(
    reference: nn.Module,
    edits: Dict[str, Any],
    boot_variant: BootVariant,
    prompt: _Prompt,
    max_interventions: int,
) -> Dict[str, Any]:
    """Consistency results for the adapter on a variant, or why the variant is unusable."""
    try:
        variant = _build_variant(reference, edits)
        twin = copy.deepcopy(variant)
        with torch.no_grad():
            twin_logits = _reference_logits(twin, prompt)
        if not torch.isfinite(twin_logits).all():
            return {"usable": False, "reason": "HuggingFace returns non-finite logits"}
    except Exception as error:
        # HF itself rejects the config, so no adapter could be expected to handle it.
        return {"usable": False, "reason": f"{type(error).__name__}: {str(error)[:200]}"}
    try:
        variant_bridge = boot_variant(variant)
        parts = _consistency_parts(variant_bridge, twin, prompt, max_interventions)
    except Exception as error:
        return {
            "usable": True,
            "failed": ["boot"],
            "error": f"{type(error).__name__}: {str(error)[:300]}",
        }
    return {"usable": True, "failed": _failed_parts(parts), "parts": parts}


# --------------------------------------------------------------------------------------
# C. Layer-type coverage
# --------------------------------------------------------------------------------------


def _layer_count(owner: Any) -> Optional[int]:
    for attribute in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        value = getattr(owner, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _is_per_layer_field(field: _ConfigField) -> bool:
    """A list with one hashable entry per layer that names a kind, not a size."""
    n_layers = _layer_count(field.owner)
    value = field.value
    if n_layers is None or not isinstance(value, (list, tuple)) or len(value) != n_layers:
        return False
    if field.name.endswith(("token_id", "token_ids")):
        return False
    if all(isinstance(v, str) for v in value):
        return True
    return all(isinstance(v, (bool, int)) and v in (0, 1) for v in value)


def _source_trees(*objects: Any) -> List[ast.AST]:
    """Parsed source of the modules that define the given classes, where it can be read."""
    trees: List[ast.AST] = []
    for obj in objects:
        module = inspect.getmodule(obj)
        if module is None:
            continue
        try:
            trees.append(ast.parse(textwrap.dedent(inspect.getsource(module))))
        except (OSError, TypeError, SyntaxError):
            continue
    return trees


def _mentions(node: ast.AST, names: set) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and child.attr in names:
            return True
        if isinstance(child, ast.Name) and child.id in names:
            return True
    return False


def _strings(node: ast.AST) -> set:
    return {
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }


def _literal_kinds(field_name: str, trees: Sequence[ast.AST]) -> set:
    """String kinds the architecture's own source compares or assigns to this field."""
    kinds: set = set()
    for tree in trees:
        aliases = {field_name}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and _mentions(node.value, {field_name}):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        aliases.add(target.id)
                    elif isinstance(target, ast.Attribute):
                        aliases.add(target.attr)
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare) and _mentions(node, aliases):
                kinds |= _strings(node)
            elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Attribute) and t.attr == field_name for t in node.targets
            ):
                kinds |= _strings(node.value)
    return kinds


def _candidate_kinds(field: _ConfigField, trees: Sequence[ast.AST]) -> List[Any]:
    present = list(dict.fromkeys(field.value))
    if not all(isinstance(v, str) for v in field.value):
        return present + [v for v in (0, 1) if v not in present]
    kinds = _literal_kinds(field.name, trees)
    try:
        default = getattr(type(field.owner)(), field.name, None)
    except Exception:
        default = None
    if isinstance(default, (list, tuple)):
        kinds |= {v for v in default if isinstance(v, str)}
    return present + sorted(kinds - set(present))


def benchmark_layer_type_coverage(
    bridge: TransformerBridge,
    reference_model: nn.Module,
    boot_variant: Optional[BootVariant] = None,
    test_input: PromptInput = DEFAULT_TEST_TEXT,
    max_interventions: int = 8,
    max_variant_params: Optional[float] = None,
) -> BenchmarkResult:
    """The adapter holds on every layer kind the config can describe, not only the checkpoint's."""
    name = "layer_type_coverage"
    try:
        boot = boot_variant or default_variant_booter(bridge)
        prompt = _prompt(bridge, test_input)
        if prompt is None:
            return _not_applicable(name, _NEEDS_TEXT)
        if not any(_is_per_layer_field(f) for f in _config_fields(reference_model.config)):
            return _not_applicable(name, "the config has no per-layer type field")
        reference_model, base_note = _variant_base(reference_model, max_variant_params)
        # The model variants start from may have a smaller embedding table than the bridge's.
        prompt = _refit(prompt, reference_model)
        config = reference_model.config
        fields = [f for f in _config_fields(config) if _is_per_layer_field(f)]
        trees = _source_trees(type(config), type(reference_model))
        with torch.no_grad():
            if not torch.isfinite(_reference_logits(reference_model, prompt)).all():
                return _could_not_run(name, _NON_FINITE)
        details: Dict[str, Any] = {"fields": {}}
        blocking: List[Dict[str, str]] = []
        exercised = 0
        if base_note is not None:
            details["variant_base"] = base_note
            # The rebuilt base has to agree with HuggingFace before any kind is changed on it.
            unchanged = _run_variant(reference_model, {}, boot, prompt, max_interventions)
            blocking += _variant_blocking("shrunken config", unchanged)
        for config_field in fields:
            present = list(dict.fromkeys(config_field.value))
            record: Dict[str, Any] = {"present": present, "exercised": {}, "rejected_by_hf": {}}
            details["fields"][config_field.path] = record
            for kind in _candidate_kinds(config_field, trees):
                if kind in present:
                    continue
                # Earlier layers stay as the checkpoint has them so the kinds sit side by side.
                layers = list(config_field.value)
                layers[-1] = kind
                outcome = _run_variant(
                    reference_model, {config_field.path: layers}, boot, prompt, max_interventions
                )
                if not outcome["usable"]:
                    record["rejected_by_hf"][str(kind)] = outcome["reason"]
                    continue
                exercised += 1
                record["exercised"][str(kind)] = outcome
                blocking += _variant_blocking(f"{config_field.path}={kind}", outcome)
        if exercised == 0:
            return _outcome(
                name,
                "the checkpoint already contains every layer kind HuggingFace accepts",
                details,
                blocking,
            )
        return _outcome(
            name, f"holds on {exercised} layer kind(s) the checkpoint lacks", details, blocking
        )
    except Exception as error:
        return _could_not_run(name, error)


# --------------------------------------------------------------------------------------
# Shrunken variants
# --------------------------------------------------------------------------------------

# Integer config fields that count or index things; shrinking them is not a width change.
_NOT_A_WIDTH = re.compile(
    r"vocab|token|position|ctx|context|window|max_|num_|^n_(?!embd|inner)|seq|length|theta|interval"
    r"|period|rank|group|kernel|stride|patch|image|channel|sample|seed|layers?$|heads?$|_id$|_ids$"
)
_LAYER_COUNT_FIELDS = ("num_hidden_layers", "n_layer", "num_layers", "n_layers")
_SMALL_VOCABULARY = 1024


# The width names HuggingFace configs share; the fallback when shrinking every wide field breaks one.
_CORE_WIDTH = re.compile(
    r"^(hidden_size|intermediate_size|moe_intermediate_size|d_model|n_embd|n_inner|d_ff|d_kv"
    r"|head_dim|ffn_dim|encoder_ffn_dim|decoder_ffn_dim)$"
)
_EXPERT_COUNT = re.compile(r"^(num|n)_.*experts?$")
_FEW_EXPERTS = 8


def _shrink_edits(
    config: Any,
    width_divisor: int,
    core_widths_only: bool,
    shrink_vocabulary: bool,
    shrink_depth: bool,
) -> Dict[str, Any]:
    """Config edits that make the architecture small without changing what it computes per layer."""
    edits: Dict[str, Any] = {}
    fields = list(_config_fields(config))
    if shrink_depth:
        by_owner: Dict[int, List[_ConfigField]] = {}
        for config_field in fields:
            by_owner.setdefault(id(config_field.owner), []).append(config_field)
        for owned in by_owner.values():
            count = next((f for f in owned if f.name in _LAYER_COUNT_FIELDS), None)
            if count is None or not isinstance(count.value, int) or count.value <= 2:
                continue
            per_layer = [f for f in owned if _is_per_layer_field(f)]
            # Keep the shortest prefix that still holds every layer kind the checkpoint has.
            kinds = [tuple(f.value[i] for f in per_layer) for i in range(count.value)]
            keep = max(2, max(kinds.index(kind) for kind in set(kinds)) + 1)
            edits[count.path] = keep
            for config_field in per_layer:
                edits[config_field.path] = list(config_field.value[:keep])
    for config_field in fields:
        value, field_name = config_field.value, config_field.name
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if "vocab_size" in field_name:
            if shrink_vocabulary and value > _SMALL_VOCABULARY:
                edits[config_field.path] = _SMALL_VOCABULARY
        elif field_name.endswith("token_id"):
            # Special ids must stay inside the shrunken table.
            if shrink_vocabulary and value >= _SMALL_VOCABULARY:
                edits[config_field.path] = value % 3
        elif width_divisor > 1 and _EXPERT_COUNT.match(field_name):
            if value > _FEW_EXPERTS:
                edits[config_field.path] = _FEW_EXPERTS
        elif (
            width_divisor > 1
            and value >= 64
            and value % width_divisor == 0
            and not _NOT_A_WIDTH.search(field_name)
            and (not core_widths_only or _CORE_WIDTH.match(field_name))
        ):
            edits[config_field.path] = value // width_divisor
    return edits


def _shrink_plans(config: Any) -> Iterator[Dict[str, Any]]:
    """Shrinks to try, most aggressive first, without repeating one."""
    seen: List[Dict[str, Any]] = []
    for width_divisor in (16, 8, 4, 2, 1):
        for core_widths_only in (False, True):
            for shrink_vocabulary in (True, False):
                for shrink_depth in (True, False):
                    edits = _shrink_edits(
                        config, width_divisor, core_widths_only, shrink_vocabulary, shrink_depth
                    )
                    if edits and edits not in seen:
                        seen.append(edits)
                        yield edits


def build_shrunken_model(
    model_class: type,
    config: Any,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    max_parameters: float = 5e7,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """A small random-weight model of this architecture, and the config edits that made it small.

    Tries the most aggressive shrink first and backs off until HuggingFace builds the
    model and runs a forward pass, so no per-architecture list of size fields is needed.
    """
    first_error = ""
    for edits in _shrink_plans(config):
        try:
            # Sized on the meta device first: a shrink that did not take would
            # otherwise allocate the full model just to be thrown away.
            with init_empty_weights():
                outline = model_class(_edited_config(config, edits))
            size = sum(p.numel() for p in outline.parameters())
            if size > max_parameters:
                raise ValueError(f"still {size / 1e6:.0f}M parameters")
            with deterministic_rng(0):
                small = model_class(_edited_config(config, edits))
            small = small.to(device=device, dtype=dtype).eval()
            tokens = _fit_vocabulary(torch.arange(1, 9, device=device)[None], small)
            kwargs = (
                {"decoder_input_ids": tokens}
                if getattr(small.config, "is_encoder_decoder", False)
                else {}
            )
            with torch.no_grad():
                output = small(tokens, **kwargs)
            logits = getattr(output, "logits", None)
            logits = logits if isinstance(logits, torch.Tensor) else output[0]
            if not torch.isfinite(logits).all():
                raise ValueError("non-finite logits")
        except Exception as error:
            # HuggingFace rejects this combination; try a gentler one. The most
            # aggressive attempt's error is the informative one.
            first_error = first_error or f"{type(error).__name__}: {str(error)[:120]}"
            continue
        return small, {
            "edits": edits,
            "parameters": sum(p.numel() for p in small.parameters()),
            "weights": "random",
        }
    raise RuntimeError(f"no shrunken config builds and runs in HuggingFace ({first_error})")


def shrunken_reference(reference: nn.Module) -> Tuple[nn.Module, Dict[str, Any]]:
    """The reference architecture rebuilt small with random weights, and how it was shrunk.

    A variant is rebuilt whole, so a large checkpoint has to be rebuilt small first.
    Random weights are fine there: the question is whether the bridge and HuggingFace
    compute the same thing, not what they compute.
    """
    first = next(reference.parameters())
    return build_shrunken_model(type(reference), reference.config, first.device, first.dtype)


# Above this a model is not rebuilt at full size even when no shrunken config works.
_FULL_SIZE_LIMIT = 2e9


def _variant_base(
    reference: nn.Module, max_variant_params: Optional[float]
) -> Tuple[nn.Module, Optional[Dict[str, Any]]]:
    """The model variants are rebuilt from: the reference, or a small random rebuild of a large one."""
    n_params = sum(p.numel() for p in reference.parameters())
    if max_variant_params is None or n_params <= max_variant_params:
        return reference, None
    try:
        return shrunken_reference(reference)
    except RuntimeError as error:
        if n_params > _FULL_SIZE_LIMIT:
            raise
        return reference, {"edits": {}, "parameters": n_params, "weights": f"checkpoint ({error})"}


def _fit_vocabulary(tokens: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Token ids folded into the model's embedding table."""
    embeddings = cast(Any, model).get_input_embeddings()
    size = getattr(embeddings, "num_embeddings", None)
    if isinstance(size, int) and size > 0 and int(tokens.max()) >= size:
        return tokens.remainder(size)
    return tokens


# --------------------------------------------------------------------------------------
# E. Scaling-factor coverage
# --------------------------------------------------------------------------------------


def _declared_float(owner: Any, field_name: str) -> bool:
    for cls in type(owner).__mro__:
        annotation = getattr(cls, "__annotations__", {}).get(field_name)
        if annotation is not None:
            return "float" in str(annotation)
    return False


def _identity_valued_fields(config: Any) -> List[_ConfigField]:
    """Float config fields sitting at 1.0, where a multiplier would be invisible."""
    fields = []
    for config_field in _config_fields(config):
        value = config_field.value
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != 1:
            continue
        if isinstance(value, float) or _declared_float(config_field.owner, config_field.name):
            fields.append(config_field)
    return fields


def _same_shapes(a: nn.Module, b: nn.Module) -> bool:
    a_state, b_state = a.state_dict(), b.state_dict()
    return a_state.keys() == b_state.keys() and all(
        a_state[key].shape == b_state[key].shape for key in a_state
    )


def benchmark_scaling_factor_coverage(
    bridge: TransformerBridge,
    reference_model: nn.Module,
    boot_variant: Optional[BootVariant] = None,
    test_input: PromptInput = DEFAULT_TEST_TEXT,
    max_interventions: int = 8,
    max_variant_params: Optional[float] = None,
) -> BenchmarkResult:
    """The adapter holds when a multiplier the checkpoint leaves at 1.0 takes another value."""
    name = "scaling_factor_coverage"
    try:
        boot = boot_variant or default_variant_booter(bridge)
        prompt = _prompt(bridge, test_input)
        if prompt is None:
            return _not_applicable(name, _NEEDS_TEXT)
        if not _identity_valued_fields(reference_model.config):
            return _not_applicable(name, "no float config field sits at 1.0")
        reference_model, base_note = _variant_base(reference_model, max_variant_params)
        # The model variants start from may have a smaller embedding table than the bridge's.
        prompt = _refit(prompt, reference_model)
        fields = _identity_valued_fields(reference_model.config)
        with torch.no_grad():
            baseline = _reference_logits(reference_model, prompt)
        if not torch.isfinite(baseline).all():
            return _could_not_run(name, _NON_FINITE)
        scale = baseline.abs().max().item()
        details: Dict[str, Any] = {"exercised": {}, "not_multipliers": {}}
        blocking: List[Dict[str, str]] = []
        if base_note is not None:
            details["variant_base"] = base_note
            # The rebuilt base has to agree with HuggingFace before any multiplier is changed on it.
            unchanged = _run_variant(reference_model, {}, boot, prompt, max_interventions)
            blocking += _variant_blocking("shrunken config", unchanged)
        for config_field in fields:
            verdict = "no value HuggingFace accepts changes the logits"
            for value in _PERTURBED_VALUES:
                edits = {config_field.path: value}
                try:
                    probe = _build_variant(reference_model, edits)
                    with torch.no_grad():
                        probe_logits = _reference_logits(probe, prompt)
                except Exception as error:
                    verdict = f"HuggingFace rejects {value}: {type(error).__name__}"
                    continue
                if not _same_shapes(probe, reference_model):
                    verdict = "changes parameter shapes, so it is a size and not a multiplier"
                    break
                if not torch.isfinite(probe_logits).all():
                    verdict = f"HuggingFace returns non-finite logits at {value}"
                    continue
                moved, _ = _relative_error(probe_logits, baseline)
                if moved <= 1e-6 * max(scale, 1e-8):
                    continue
                outcome = _run_variant(reference_model, edits, boot, prompt, max_interventions)
                if not outcome["usable"]:
                    verdict = outcome["reason"]
                    continue
                outcome["value"] = value
                details["exercised"][config_field.path] = outcome
                blocking += _variant_blocking(f"{config_field.path}={value}", outcome)
                verdict = ""
                break
            if verdict:
                details["not_multipliers"][config_field.path] = verdict
        if not details["exercised"]:
            return _not_applicable(
                name, "no field at 1.0 acts as a multiplier on the logits", details
            )
        fields_run = ", ".join(details["exercised"])
        return _outcome(
            name, f"holds with non-identity values for: {fields_run}", details, blocking
        )
    except Exception as error:
        return _could_not_run(name, error)


# --------------------------------------------------------------------------------------
# F. Reference comparison
# --------------------------------------------------------------------------------------


def adapter_surface(bridge: TransformerBridge) -> Dict[str, set]:
    """The hook names and weight names an adapter exposes on a checkpoint."""
    return {
        "hooks": set(bridge.hook_dict),
        "weights": set(bridge.state_dict()),
    }


def reference_coverage(surface: Dict[str, set], reference_surface: Dict[str, set]) -> float:
    """Share of the reference's hook and weight names the adapter also has, pooled."""
    wanted = sum(len(reference_surface[key]) for key in ("hooks", "weights"))
    shared = sum(len(reference_surface[key] & surface[key]) for key in ("hooks", "weights"))
    return shared / wanted if wanted else 1.0


def benchmark_reference_comparison(
    surface: Dict[str, set], reference_surface: Dict[str, set]
) -> BenchmarkResult:
    """Which of a reference adapter's hook names and weight names the adapter also exposes."""
    name = "reference_comparison"
    try:
        details: Dict[str, Any] = {
            "coverage": round(reference_coverage(surface, reference_surface), 4)
        }
        gaps: List[Dict[str, str]] = []
        for key in ("hooks", "weights"):
            missing = sorted(_collapse_set(reference_surface[key] - surface[key]))
            # Names only the adapter under test has are listed and cost nothing.
            extra = sorted(_collapse_set(surface[key] - reference_surface[key]))
            details[key] = {
                "adapter": len(surface[key]),
                "reference": len(reference_surface[key]),
                "missing_from_adapter": missing,
                "only_in_adapter": extra,
            }
            gaps += [_gap("missing_from_reference", pattern) for pattern in missing]
        summary = (
            f"the adapter has {details['coverage']:.1%} of the reference's hook and weight names"
        )
        return _outcome(name, summary, details, gaps=gaps)
    except Exception as error:
        return _could_not_run(name, error)


# --------------------------------------------------------------------------------------
# Declared gaps
# --------------------------------------------------------------------------------------

# ArchitectureAdapter attribute: HF module path pattern or hook name -> one-line reason.
DECLARED_GAPS_ATTRIBUTE = "declared_coverage_gaps"


def declared_gaps(adapter: Any) -> Dict[str, str]:
    """An adapter's own account of what it leaves unexposed, or {} when it gives none."""
    declared = getattr(adapter, DECLARED_GAPS_ATTRIBUTE, None)
    if not isinstance(declared, dict):
        return {}
    return {str(pattern): str(reason) for pattern, reason in declared.items()}


def _declaration_covers(pattern: str, where: str) -> bool:
    pattern, where = _collapse_indices(pattern), _collapse_indices(where)
    return (
        where == pattern
        or where.startswith(pattern + ".")
        or where.endswith("." + pattern)
        or fnmatch.fnmatchcase(where, pattern)
    )


def annotate_gaps(
    gaps: Sequence[Dict[str, str]], declarations: Dict[str, str]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Attach each gap's declared reason; also return declarations that match no gap.

    A declaration explains a gap and never removes it, so it cannot change a verdict.
    """
    annotated: List[Dict[str, Any]] = []
    used: set = set()
    for gap in gaps:
        match = next((p for p in declarations if _declaration_covers(p, gap["where"])), None)
        if match is not None:
            used.add(match)
        annotated.append(
            {
                "kind": gap["kind"],
                "where": gap["where"],
                "declared": match is not None,
                "reason": declarations[match] if match is not None else None,
            }
        )
    return annotated, sorted(set(declarations) - used)
