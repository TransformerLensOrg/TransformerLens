"""The unembed component test must account for a trained final_logits_bias.

Marian/Bart apply a learned ``final_logits_bias`` AFTER lm_head inside the model
forward. The bridge folds it into the unembed bias (b_U), so an isolated
comparison against HF's bare lm_head diverges by exactly the bias — a false
component failure even though the assembled forward matches HF. The benchmark
adds the bias to the HF side so the comparison is fair. Marian's opus-mt bias
is trained (max ~7.4); most Bart models leave it zero (hence they never tripped).
"""

import pytest

pytest.importorskip("transformers")


def test_marian_unembed_passes_with_final_logits_bias():
    """opus-mt has a non-zero final_logits_bias; the unembed component must
    still match HF once the bias is applied to both sides."""
    import torch

    try:
        from transformers import AutoModelForSeq2SeqLM

        from transformer_lens.benchmarks.component_outputs import ComponentBenchmarker
        from transformer_lens.model_bridge import TransformerBridge

        bridge = TransformerBridge.boot_transformers("Helsinki-NLP/opus-mt-nl-en", device="cpu")
        hf = AutoModelForSeq2SeqLM.from_pretrained(
            "Helsinki-NLP/opus-mt-nl-en", dtype=torch.float32
        ).eval()
    except (OSError, ConnectionError, TimeoutError) as exc:
        pytest.skip(f"marian unavailable offline: {exc}")

    # Precondition: the bias is genuinely non-zero (else the test proves nothing).
    flb = getattr(hf, "final_logits_bias", None)
    assert flb is not None and flb.abs().max().item() > 1.0

    bridge.adapter.setup_component_testing(hf, bridge_model=bridge)
    bench = ComponentBenchmarker(bridge, hf, bridge.adapter, bridge.cfg)
    report = bench.benchmark_all_components()

    unembed = [r for r in report.component_results if r.component_path.endswith("unembed")]
    assert unembed, "unembed component was not tested"
    assert all(r.passed for r in unembed), [
        (r.component_path, r.max_diff) for r in unembed if not r.passed
    ]
