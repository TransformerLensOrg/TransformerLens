"""MPS regression test for the SSM2 eager-scan intervention path (device correctness).

The eager scan builds its recurrent ``ssm_state`` accumulator explicitly; if that
tensor is created on the default CPU device instead of the input's device, the
recurrence ``decay * ssm_state + writes`` crashes on any non-CPU model. This test
guards that on Metal. Skips on non-MPS runners; runs in the `mps-checks` CI job.
"""

import gc

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available on this runner — skipping Apple Silicon SSM tests",
)


class _Tok:
    pass


def _tiny_mamba2_bridge():
    """Tiny synthetic Mamba-2 bridge (float32, no Hub download), built on CPU."""
    from transformers import AutoModelForCausalLM
    from transformers.models.mamba2 import Mamba2Config

    from transformer_lens.model_bridge.bridge import TransformerBridge
    from transformer_lens.model_bridge.sources._bridge_builder import (
        build_bridge_config_from_hf,
    )
    from transformer_lens.model_bridge.supported_architectures.mamba2 import (
        Mamba2ArchitectureAdapter,
    )

    torch.manual_seed(0)
    cfg = Mamba2Config(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        state_size=16,
        expand=2,
        head_dim=16,
        num_heads=4,
        n_groups=2,
        conv_kernel=4,
        chunk_size=8,
    )
    cfg.architectures = ["Mamba2ForCausalLM"]
    hf = AutoModelForCausalLM.from_config(cfg).to(torch.float32).eval()
    bridge_cfg = build_bridge_config_from_hf(hf.config, "Mamba2ForCausalLM", "x", torch.float32)
    return TransformerBridge(hf, Mamba2ArchitectureAdapter(bridge_cfg), tokenizer=_Tok())


def test_mps_eager_scan_runs_on_metal():
    """eager_scan=True must run on MPS (ssm_state created on the input's device) and
    match the CPU fused kernel, with the intervention hooks firing on Metal."""
    bridge = _tiny_mamba2_bridge()
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        cpu_fused = bridge(tokens)

    bridge.original_model.to("mps")
    for block in bridge.blocks:
        block.mixer.eager_scan = True
    tokens_mps = tokens.to("mps")

    with torch.no_grad():
        out = bridge(tokens_mps, use_cache=False)  # would crash pre-fix (CPU ssm_state)
        _, cache = bridge.run_with_cache(tokens_mps, use_cache=False)

    assert out.device.type == "mps", f"eager-scan output not on MPS: {out.device}"
    assert cache["blocks.0.mixer.hook_ssm_write"].device.type == "mps"
    assert cache["blocks.0.mixer.hook_ssm_state"].device.type == "mps"

    rel = (out.cpu() - cpu_fused).abs().max().item() / max(cpu_fused.abs().max().item(), 1e-8)
    assert rel < 1e-3, f"MPS eager scan diverges from CPU fused kernel: rel {rel:.2e}"

    del bridge
    torch.mps.empty_cache()
    gc.collect()
