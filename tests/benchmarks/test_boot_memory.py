#!/usr/bin/env python3
"""Memory benchmark: TransformerBridge.boot_transformers vs HookedTransformer.from_pretrained.

Run with: python -m pytest tests/benchmarks/test_boot_memory.py -v -s
Or directly: python tests/benchmarks/test_boot_memory.py [model_name]
"""

import gc
import os
import subprocess
import sys

import pytest


def get_rss_mb():
    """Get current process RSS in MB."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024
        except FileNotFoundError:
            pass
        try:
            result = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(os.getpid())],
                capture_output=True,
                text=True,
            )
            return int(result.stdout.strip()) / 1024
        except Exception:
            return 0.0


def profile_hooked_transformer(
    model_name, fold_ln=False, fold_value_biases=False, center_writing_weights=False
):
    """Profile HookedTransformer.from_pretrained RSS at each stage."""
    import torch

    _ = torch.set_grad_enabled(False)
    checkpoints = []

    gc.collect()
    checkpoints.append(("baseline", get_rss_mb()))

    from transformer_lens import HookedTransformer

    gc.collect()
    checkpoints.append(("after import", get_rss_mb()))

    model = HookedTransformer.from_pretrained(
        model_name,
        fold_ln=fold_ln,
        fold_value_biases=fold_value_biases,
        center_writing_weights=center_writing_weights,
    )
    gc.collect()
    checkpoints.append(("after from_pretrained", get_rss_mb()))

    param_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1024 / 1024
    checkpoints.append(("param_size_mb", param_mb))

    del model
    gc.collect()
    checkpoints.append(("after del model", get_rss_mb()))

    return checkpoints


def profile_transformer_bridge(
    model_name, fold_ln=False, fold_value_biases=False, center_writing_weights=False
):
    """Profile TransformerBridge.boot_transformers RSS at each stage."""
    import torch

    _ = torch.set_grad_enabled(False)
    checkpoints = []

    gc.collect()
    checkpoints.append(("baseline", get_rss_mb()))

    from transformer_lens.model_bridge import TransformerBridge

    gc.collect()
    checkpoints.append(("after import", get_rss_mb()))

    bridge = TransformerBridge.boot_transformers(model_name)
    gc.collect()
    checkpoints.append(("after boot_transformers", get_rss_mb()))

    bridge.enable_compatibility_mode(
        fold_ln=fold_ln,
        fold_value_biases=fold_value_biases,
        center_writing_weights=center_writing_weights,
    )
    gc.collect()
    checkpoints.append(("after enable_compatibility_mode", get_rss_mb()))

    param_mb = sum(p.nelement() * p.element_size() for p in bridge.parameters()) / 1024 / 1024
    checkpoints.append(("param_size_mb", param_mb))

    del bridge
    gc.collect()
    checkpoints.append(("after del bridge", get_rss_mb()))

    return checkpoints


def run_in_subprocess(func_name, model_name, **kwargs):
    """Run a profiling function in a fresh subprocess for clean RSS readings."""
    kwargs_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    script = f"""
import sys
sys.path.insert(0, '.')
from tests.benchmarks.test_boot_memory import {func_name}
results = {func_name}({model_name!r}, {kwargs_str})
for name, val in results:
    print(f"{{name}}\\t{{val:.1f}}")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    )
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"{func_name} subprocess failed (exit {result.returncode})")

    checkpoints = {}
    for line in result.stdout.strip().split("\n"):
        if "\t" in line:
            name, val = line.split("\t", 1)
            checkpoints[name] = float(val)
    return checkpoints


MEMORY_BENCHMARK_MODELS = ["gpt2"]
_BENCH_KWARGS = dict(fold_ln=False, fold_value_biases=False, center_writing_weights=False)


class TestBootMemory:
    """Ensure TransformerBridge memory stays within bounds relative to HookedTransformer."""

    @pytest.mark.parametrize("model_name", MEMORY_BENCHMARK_MODELS)
    def test_bridge_memory_within_bounds(self, model_name):
        """TransformerBridge RSS must not exceed 4x parameter size."""
        results = run_in_subprocess("profile_transformer_bridge", model_name, **_BENCH_KWARGS)

        param_mb = results["param_size_mb"]
        net_rss = results["after enable_compatibility_mode"] - results["baseline"]
        max_allowed = param_mb * 4

        print(f"\n  TransformerBridge({model_name}):")
        print(f"    Param size:    {param_mb:>8.1f} MB")
        print(f"    Net RSS:       {net_rss:>8.1f} MB ({net_rss / param_mb:.1f}x params)")
        print(f"    Max allowed:   {max_allowed:>8.1f} MB (4x params)")

        assert net_rss < max_allowed, (
            f"TransformerBridge RSS ({net_rss:.0f} MB) exceeds 4x param size "
            f"({max_allowed:.0f} MB) for {model_name}. Ratio: {net_rss / param_mb:.1f}x"
        )

    @pytest.mark.parametrize("model_name", MEMORY_BENCHMARK_MODELS)
    def test_bridge_vs_hooked_transformer_ratio(self, model_name):
        """TransformerBridge must use no more than 2x the RSS of HookedTransformer."""
        ht_results = run_in_subprocess("profile_hooked_transformer", model_name, **_BENCH_KWARGS)
        bridge_results = run_in_subprocess(
            "profile_transformer_bridge", model_name, **_BENCH_KWARGS
        )

        ht_net = ht_results["after from_pretrained"] - ht_results["baseline"]
        bridge_net = bridge_results["after enable_compatibility_mode"] - bridge_results["baseline"]
        ratio = bridge_net / ht_net if ht_net > 0 else float("inf")

        print(f"\n  Memory comparison ({model_name}):")
        print(f"    HookedTransformer:  {ht_net:>8.1f} MB")
        print(f"    TransformerBridge:  {bridge_net:>8.1f} MB")
        print(f"    Ratio:              {ratio:>8.1f}x")

        assert ratio < 2.0, (
            f"TransformerBridge uses {ratio:.1f}x more memory than HookedTransformer "
            f"for {model_name} (Bridge: {bridge_net:.0f} MB, HT: {ht_net:.0f} MB). Expected < 2.0x."
        )


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    print(f"Memory benchmark for: {model_name}")
    print("=" * 60)

    print("\nHookedTransformer.from_pretrained:")
    ht = run_in_subprocess("profile_hooked_transformer", model_name, **_BENCH_KWARGS)
    for name, val in ht.items():
        print(f"  {name:<35s} {val:>8.1f} MB")

    print("\nTransformerBridge.boot_transformers:")
    bridge = run_in_subprocess("profile_transformer_bridge", model_name, **_BENCH_KWARGS)
    for name, val in bridge.items():
        print(f"  {name:<35s} {val:>8.1f} MB")

    print("\n" + "=" * 60)
    ht_net = ht["after from_pretrained"] - ht["baseline"]
    bridge_net = bridge["after enable_compatibility_mode"] - bridge["baseline"]
    print(f"HookedTransformer net:   {ht_net:>8.1f} MB")
    print(f"TransformerBridge net:   {bridge_net:>8.1f} MB")
    print(f"Ratio:                   {bridge_net / ht_net:>8.1f}x")
    print(f"Param size:              {bridge['param_size_mb']:>8.1f} MB")
