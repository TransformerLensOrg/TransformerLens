"""Run the adapter coverage checks on one adapter, as a command or a function.

    python -m transformer_lens.benchmarks.adapter_coverage_runner \\
        --adapter my_adapter.py --checkpoint org/tiny-foo --output report.json

Omit ``--adapter`` to check the adapter already registered for the checkpoint. An adapter
file is registered for the run only; no repo file or registry entry is written.
``run_coverage_checks`` runs the same checks on a bridge that is already built.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from transformers import AutoConfig

from transformer_lens.benchmarks.adapter_coverage import (
    DEFAULT_TEST_TEXT,
    BootVariant,
    PromptInput,
    adapter_surface,
    annotate_gaps,
    benchmark_expected_hooks,
    benchmark_hook_consistency,
    benchmark_layer_type_coverage,
    benchmark_reference_comparison,
    benchmark_residual_floor,
    benchmark_scaling_factor_coverage,
    benchmark_structural_coverage,
    build_reference_model,
    declared_gaps,
    default_variant_booter,
)
from transformer_lens.benchmarks.utils import BenchmarkResult, BenchmarkSeverity
from transformer_lens.factories import architecture_adapter_factory
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.sources.transformers import (
    determine_architecture_from_hf_config,
)

_DTYPES = {"float32": torch.float32, "float64": torch.float64, "bfloat16": torch.bfloat16}
_CHECK_ORDER = (
    "structural_coverage",
    "expected_hooks",
    "residual_floor",
    "layer_type_coverage",
    "hook_consistency",
    "scaling_factor_coverage",
    "reference_comparison",
)
_VARIANT_CHECKS = ("layer_type_coverage", "scaling_factor_coverage")
_ADAPTER_MODULE = "coverage_adapter_under_test"
_REFERENCE_MODULE = "coverage_reference_adapter"


def load_adapter_class(source: str, module_name: str = _ADAPTER_MODULE) -> type:
    """The one adapter class a file or importable module defines."""
    if source.endswith(".py"):
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import an adapter from {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(source)
    defined = [
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and issubclass(value, ArchitectureAdapter)
        and value.__module__ == module.__name__
    ]
    # A file may define a private base for its adapter; the adapter is the leaf.
    leaves = [c for c in defined if not any(c is not o and issubclass(o, c) for o in defined)]
    if len(leaves) != 1:
        names = [c.__name__ for c in leaves]
        raise ValueError(f"expected exactly one adapter class in {source}, found {names}")
    return leaves[0]


def register_adapter(hf_class: str, adapter_class: type) -> None:
    """Point the adapter factory at ``adapter_class`` for this process."""
    factory = architecture_adapter_factory.ArchitectureAdapterFactory
    factory.register_adapter(hf_class, adapter_class)
    # The module-level table answers "is this architecture supported"; the factory selects
    # from its own copy, so both have to name the adapter under test.
    architecture_adapter_factory.SUPPORTED_ARCHITECTURES[hf_class] = adapter_class
    if factory._adapters.get(hf_class) is not adapter_class:
        raise RuntimeError(f"the adapter factory did not take {adapter_class.__name__}")


def _status(result: BenchmarkResult) -> str:
    if result.severity == BenchmarkSeverity.SKIPPED:
        return "skip"
    if not result.passed:
        return "fail"
    return "gaps" if result.severity == BenchmarkSeverity.WARNING else "pass"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_jsonable(v) for v in value]
        return sorted(items, key=str) if isinstance(value, (set, frozenset)) else items
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def summarize(
    results: Sequence[BenchmarkResult],
    declarations: Optional[Dict[str, str]] = None,
    **header: Any,
) -> Dict[str, Any]:
    """The report: blocking findings decide ``passed``; gaps only lower ``coverage``."""
    order = {name: index for index, name in enumerate(_CHECK_ORDER)}
    ordered = sorted(results, key=lambda r: order.get(r.name, len(order)))
    blocking_findings: List[Dict[str, str]] = []
    gaps: List[Dict[str, str]] = []
    n_blocking_parts = 0
    coverage: Dict[str, Optional[float]] = {}
    for result in ordered:
        details = result.details or {}
        for entry in details.get("blocking", []):
            n_blocking_parts += 1
            if entry["status"] == "fail":
                blocking_findings.append(
                    {"check": result.name, "part": entry["part"], "message": entry["message"]}
                )
        gaps += [gap for gap in details.get("gaps", []) if gap not in gaps]
        if result.passed and result.severity != BenchmarkSeverity.SKIPPED:
            if result.name == "reference_comparison":
                coverage["reference"] = details.get("coverage")
            elif result.name == "structural_coverage":
                coverage["ownership"] = (details.get("pattern_coverage") or {}).get("share")
    # A reference adapter is the better yardstick when there is one.
    basis = next((b for b in ("reference", "ownership") if coverage.get(b) is not None), None)
    annotated, stale = annotate_gaps(gaps, declarations or {})
    return {
        **header,
        "passed": not blocking_findings,
        "blocking_findings": blocking_findings,
        "n_blocking_parts": n_blocking_parts,
        "n_blocking_failed": len(blocking_findings),
        "coverage": coverage[basis] if basis else None,
        "coverage_basis": basis,
        "gaps": annotated,
        "stale_declarations": stale,
        "checks": [
            {
                "name": result.name,
                "status": _status(result),
                "message": result.message,
                "details": _jsonable(result.details or {}),
            }
            for result in ordered
        ],
    }


def _blocked(name: str, message: str, part: str = "run") -> BenchmarkResult:
    return BenchmarkResult(
        name=name,
        severity=BenchmarkSeverity.ERROR,
        message=message,
        details={"blocking": [{"part": part, "status": "fail", "message": message}], "gaps": []},
        passed=False,
    )


def _skipped(name: str, message: str) -> BenchmarkResult:
    return BenchmarkResult(name=name, severity=BenchmarkSeverity.SKIPPED, message=message)


def run_coverage_checks(
    bridge: TransformerBridge,
    reference_model: Optional[nn.Module] = None,
    boot_variant: Optional[BootVariant] = None,
    test_input: PromptInput = DEFAULT_TEST_TEXT,
    reference_surface: Optional[Dict[str, set]] = None,
    variant_bridge: Optional[TransformerBridge] = None,
    max_variant_params: float = 2e7,
    **header: Any,
) -> Dict[str, Any]:
    """Run every check on a bridge that is already built and return the report.

    Needs no checkpoint name, adapter file or subprocess: any bridge will do.

    Args:
        bridge: The bridge whose adapter is under test.
        reference_model: A pristine HF twin with the same weights; rebuilt from the
            bridge when omitted.
        boot_variant: Builds a bridge around a rebuilt HF model, for the layer-type and
            scaling checks; defaults to the way ``bridge`` itself was built.
        test_input: Prompt text, or token ids for a bridge with no tokenizer.
        reference_surface: ``adapter_surface`` of a reference adapter's bridge on the
            same checkpoint; the reference comparison is skipped without it.
        variant_bridge: A smaller bridge of the same class to rebuild variants from.
        max_variant_params: Above this size, variants are rebuilt from a shrunken config.
        **header: Extra fields copied into the report.
    """
    results = _check_bridge(
        bridge,
        reference_model,
        boot_variant,
        test_input,
        reference_surface,
        variant_bridge,
        max_variant_params,
    )
    return summarize(results, declared_gaps(bridge.adapter), **header)


def _check_bridge(
    bridge: TransformerBridge,
    reference_model: Optional[nn.Module],
    boot_variant: Optional[BootVariant],
    test_input: PromptInput,
    reference_surface: Optional[Dict[str, set]],
    variant_bridge: Optional[TransformerBridge],
    max_variant_params: float,
) -> List[BenchmarkResult]:
    results = [
        benchmark_structural_coverage(bridge),
        benchmark_expected_hooks(bridge, test_input),
        benchmark_residual_floor(bridge, test_input),
    ]
    reference = reference_model
    try:
        if reference is None:
            reference = build_reference_model(bridge)
        results.append(benchmark_hook_consistency(bridge, reference, test_input))
    except Exception as error:
        reason = f"could not run: no HuggingFace reference: {type(error).__name__}: {error}"
        results += [_blocked(name, reason) for name in ("hook_consistency", *_VARIANT_CHECKS)]

    if reference is not None:
        variant_checks: List[Callable[..., BenchmarkResult]] = [
            benchmark_layer_type_coverage,
            benchmark_scaling_factor_coverage,
        ]
        try:
            base_bridge, base_reference = bridge, reference
            if variant_bridge is not None:
                base_bridge, base_reference = variant_bridge, build_reference_model(variant_bridge)
            booter = boot_variant or default_variant_booter(base_bridge)
            results += [
                check(
                    base_bridge,
                    base_reference,
                    booter,
                    test_input,
                    max_variant_params=max_variant_params,
                )
                for check in variant_checks
            ]
        except Exception as error:
            reason = f"could not run: {type(error).__name__}: {error}"
            results += [_blocked(name, reason) for name in _VARIANT_CHECKS]

    if reference_surface is None:
        results.append(_skipped("reference_comparison", "no reference adapter was given"))
    else:
        results.append(benchmark_reference_comparison(adapter_surface(bridge), reference_surface))
    return results


def _checkpoint_class(checkpoint: str, trust_remote_code: bool) -> str:
    """The HuggingFace class a checkpoint's own config names."""
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=trust_remote_code)
    architectures = getattr(config, "architectures", None) or []
    if architectures:
        return str(architectures[0])
    try:
        # Some checkpoints name no class; the library maps their model_type where it knows it.
        return str(determine_architecture_from_hf_config(config))
    except ValueError:
        raise ValueError(
            f"{checkpoint} does not name its architecture; pass the HuggingFace class explicitly"
        ) from None


def run_checks_in_process(
    checkpoint: str,
    adapter_path: Optional[str] = None,
    hf_class: Optional[str] = None,
    reference_adapter: Optional[str] = None,
    device: str = "cpu",
    dtype: str = "float32",
    trust_remote_code: bool = False,
    variant_checkpoint: Optional[str] = None,
    max_variant_params: float = 2e7,
) -> Dict[str, Any]:
    """Boot a checkpoint and run every check in this process.

    Prefer :func:`run_adapter_coverage`, which isolates the run in a subprocess: an
    adapter file given here stays registered for the rest of the process.
    """
    started = time.time()
    header: Dict[str, Any] = {
        "adapter": adapter_path,
        "hf_class": hf_class,
        "checkpoint": checkpoint,
        "device": device,
        "dtype": dtype,
    }
    boot_kwargs: Dict[str, Any] = dict(
        device=device, dtype=_DTYPES[dtype], trust_remote_code=trust_remote_code
    )

    def boot(model_name: str) -> TransformerBridge:
        return TransformerBridge.boot_transformers(model_name, **boot_kwargs)

    try:
        if adapter_path is not None or reference_adapter is not None:
            hf_class = hf_class or _checkpoint_class(checkpoint, trust_remote_code)
            header["hf_class"] = hf_class
        if adapter_path is not None and hf_class is not None:
            adapter_class = load_adapter_class(adapter_path)
            header["adapter_class"] = adapter_class.__name__
            register_adapter(hf_class, adapter_class)
        bridge = boot(checkpoint)
    except Exception as error:
        # Nothing else can be said about an adapter whose bridge does not boot.
        reason = f"the bridge did not boot: {type(error).__name__}: {error}"
        return summarize(
            [_blocked("boot", reason, part="boot")],
            **header,
            elapsed_s=round(time.time() - started, 1),
        )

    surface = adapter_surface(bridge)
    variant_bridge = None
    variant_failure = ""
    if variant_checkpoint is not None:
        try:
            variant_bridge = boot(variant_checkpoint)
        except Exception as error:
            variant_failure = f"could not run: {type(error).__name__}: {error}"
    results = _check_bridge(
        bridge,
        None,
        default_variant_booter(variant_bridge or bridge, **boot_kwargs),
        DEFAULT_TEST_TEXT,
        None,
        variant_bridge,
        max_variant_params,
    )
    if variant_failure:
        # The requested variant base never booted, so its two checks certify nothing.
        results = [r for r in results if r.name not in _VARIANT_CHECKS]
        results += [_blocked(name, variant_failure) for name in _VARIANT_CHECKS]
    if reference_adapter is not None and hf_class is not None:
        results = [r for r in results if r.name != "reference_comparison"]
        try:
            # Registered last: every check above ran against the adapter under test.
            register_adapter(hf_class, load_adapter_class(reference_adapter, _REFERENCE_MODULE))
            results.append(
                benchmark_reference_comparison(surface, adapter_surface(boot(checkpoint)))
            )
        except Exception as error:
            results.append(
                _blocked("reference_comparison", f"could not run: {type(error).__name__}: {error}")
            )
    return summarize(
        results,
        declared_gaps(bridge.adapter),
        **header,
        elapsed_s=round(time.time() - started, 1),
    )


def run_adapter_coverage(
    checkpoint: str,
    adapter_path: Optional[str] = None,
    hf_class: Optional[str] = None,
    reference_adapter: Optional[str] = None,
    device: str = "cpu",
    dtype: str = "float32",
    output_path: Optional[str] = None,
    trust_remote_code: bool = False,
    variant_checkpoint: Optional[str] = None,
    max_variant_params: float = 2e7,
    timeout: float = 600.0,
) -> Dict[str, Any]:
    """Run the coverage checks in a fresh subprocess and return the report."""
    with tempfile.TemporaryDirectory() as scratch:
        target = output_path or str(Path(scratch) / "coverage.json")
        command = [
            sys.executable,
            "-m",
            "transformer_lens.benchmarks.adapter_coverage_runner",
            "--in-process",
            "--checkpoint",
            checkpoint,
            "--device",
            device,
            "--dtype",
            dtype,
            "--max-variant-params",
            str(max_variant_params),
            "--output",
            target,
        ]
        optional = {
            "--adapter": adapter_path,
            "--hf-class": hf_class,
            "--reference-adapter": reference_adapter,
            "--variant-checkpoint": variant_checkpoint,
        }
        for flag, value in optional.items():
            if value is not None:
                command += [flag, value]
        if trust_remote_code:
            command.append("--trust-remote-code")
        header: Dict[str, Any] = {
            "adapter": adapter_path,
            "hf_class": hf_class,
            "checkpoint": checkpoint,
            "device": device,
            "dtype": dtype,
        }
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            report = summarize([_blocked("run", f"timed out after {timeout:.0f}s")], **header)
        else:
            if Path(target).exists():
                report = json.loads(Path(target).read_text())
            else:
                tail = (completed.stderr or completed.stdout)[-1500:]
                report = summarize([_blocked("run", f"the run wrote no report: {tail}")], **header)
        if output_path is not None:
            Path(output_path).write_text(json.dumps(report, indent=2))
        return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--checkpoint", required=True, help="HuggingFace checkpoint id")
    parser.add_argument("--adapter", help="adapter file to test; omit to test the registered one")
    parser.add_argument("--hf-class", help="HuggingFace class the adapter file is registered for")
    parser.add_argument(
        "--reference-adapter", help="reference adapter module or file to compare with"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=sorted(_DTYPES))
    parser.add_argument("--output", help="path for the JSON report")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--variant-checkpoint",
        help="checkpoint of the same class to rebuild layer-type and scaling variants from",
    )
    parser.add_argument(
        "--max-variant-params",
        type=float,
        default=2e7,
        help="above this size, variants are rebuilt from a shrunken config with random weights",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--in-process", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    shared = dict(
        checkpoint=args.checkpoint,
        adapter_path=args.adapter,
        hf_class=args.hf_class,
        reference_adapter=args.reference_adapter,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        variant_checkpoint=args.variant_checkpoint,
        max_variant_params=args.max_variant_params,
    )
    if args.in_process:
        report = run_checks_in_process(**shared)
        if args.output:
            Path(args.output).write_text(json.dumps(report, indent=2))
    else:
        report = run_adapter_coverage(**shared, output_path=args.output, timeout=args.timeout)
    for check in report["checks"]:
        print(f"[{check['status'].upper():4}] {check['name']}: {check['message']}")
    coverage = report["coverage"]
    declared = sum(1 for gap in report["gaps"] if gap["declared"])
    print(
        f"{'PASS' if report['passed'] else 'FAIL'}: {report['n_blocking_failed']} of "
        f"{report['n_blocking_parts']} blocking parts failed; "
        + (
            f"coverage {coverage:.2f} ({report['coverage_basis']}); "
            if coverage is not None
            else "coverage not measured; "
        )
        + f"{len(report['gaps'])} gap(s), {declared} declared"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
