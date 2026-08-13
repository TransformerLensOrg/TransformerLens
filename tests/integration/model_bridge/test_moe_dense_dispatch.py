"""Dense-prefix MoE dispatch on a real mixed checkpoint (#1645).

katuni4ka/tiny-random-deepseek-v3 has first_k_dense_replace=1 over two layers:
layer 0 is a plain gated MLP, layer 1 a sparse MoE block — one boot exercises
both bindings of the DenseOrMoEBridge template.
"""

import torch
from transformers import AutoModelForCausalLM

from tests.tiny_checkpoints import MIXED_DENSE_SPARSE_MOE, assert_tiny_parity
from transformer_lens.model_bridge import TransformerBridge

MODEL_NAME = MIXED_DENSE_SPARSE_MOE


def test_mixed_checkpoint_dense_and_sparse_mlp_hooks() -> None:
    bridge = TransformerBridge.boot_transformers(MODEL_NAME, device="cpu", dtype=torch.float32)
    tokens = bridge.to_tokens("The quick brown fox jumps")
    d_model = bridge.cfg.d_model
    d_mlp = bridge.original_model.config.intermediate_size

    with torch.no_grad():
        logits, cache = bridge.run_with_cache(tokens)

    # Dense layer 0: neuron-basis hooks with gated-MLP semantics.
    assert cache["blocks.0.mlp.hook_pre"].shape[-1] == d_mlp
    torch.testing.assert_close(
        cache["blocks.0.mlp.hook_pre"], cache["blocks.0.mlp.dense_gate.hook_out"]
    )
    torch.testing.assert_close(
        cache["blocks.0.mlp.hook_pre_linear"], cache["blocks.0.mlp.dense_in.hook_out"]
    )
    torch.testing.assert_close(
        cache["blocks.0.mlp.hook_post"], cache["blocks.0.mlp.dense_out.hook_in"]
    )

    # Sparse layer 1: MoE boundary semantics unchanged.
    assert cache["blocks.1.mlp.hook_pre"].shape[-1] == d_model
    torch.testing.assert_close(cache["blocks.1.mlp.hook_pre"], cache["blocks.1.mlp.hook_in"])

    # The dispatch must not perturb the forward itself.
    hf_eager = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    with torch.no_grad():
        hf_logits = hf_eager(tokens).logits
    assert_tiny_parity(logits, hf_logits, MODEL_NAME)


def test_dense_layer_neuron_intervention_targets_gate_output() -> None:
    """A write on the dense layer's hook_pre must be a write on the gate
    projection's output — the #1645 complaint was precisely that no
    neuron-basis intervention point existed on dense layers. (Pre-fix,
    hook_pre aliased the MLP *input*, so zeroing it also moved logits —
    equivalence with the gate target is what pins the semantics.)"""
    bridge = TransformerBridge.boot_transformers(MODEL_NAME, device="cpu", dtype=torch.float32)
    tokens = bridge.to_tokens("The quick brown fox jumps")

    def zero(tensor: torch.Tensor, hook) -> torch.Tensor:
        return tensor * 0.0

    with torch.no_grad():
        base_logits = bridge(tokens)
        via_alias = bridge.run_with_hooks(tokens, fwd_hooks=[("blocks.0.mlp.hook_pre", zero)])
        via_target = bridge.run_with_hooks(
            tokens, fwd_hooks=[("blocks.0.mlp.dense_gate.hook_out", zero)]
        )
    torch.testing.assert_close(via_alias, via_target)
    assert not torch.allclose(base_logits, via_alias)


def test_dense_layer_weights_reach_get_params() -> None:
    """The neuron hooks advertise a neuron basis; the weight API must back it.
    get_params() keyed on submodules named `in`/`out`, so dense MoE layers were
    silently zero-filled — real weights on the component, zeros in the dict."""
    bridge = TransformerBridge.boot_transformers(MODEL_NAME, device="cpu", dtype=torch.float32)
    params = bridge.get_params()

    dense = bridge.blocks[0].mlp
    assert dense.bound_dense is True
    # Anchor on the wrapped HF module, NOT on dense.W_in/W_gate/W_out:
    # get_params_util builds these entries by reading those very properties, so
    # comparing the two would be the same expression evaluated twice and could
    # not detect a projection being read from the wrong submodule.
    hf_mlp = dense.original_component
    torch.testing.assert_close(params["blocks.0.mlp.W_in"], hf_mlp.up_proj.weight.T)
    torch.testing.assert_close(params["blocks.0.mlp.W_gate"], hf_mlp.gate_proj.weight.T)
    torch.testing.assert_close(params["blocks.0.mlp.W_out"], hf_mlp.down_proj.weight.T)
    assert (params["blocks.0.mlp.W_in"] != 0).any()
    # Negative control: gate and up are same-shaped, so this checkpoint can tell
    # them apart only if their weights actually differ.
    assert not torch.equal(hf_mlp.gate_proj.weight, hf_mlp.up_proj.weight)

    # Sparse layers legitimately have no single W_* and keep the placeholder.
    assert bridge.blocks[1].mlp.bound_dense is False
    assert (params["blocks.1.mlp.W_in"] == 0).all()


def test_hook_dict_follows_a_dense_to_sparse_rebind() -> None:
    """The alias cache must not outlive the binding it described: a rebind can
    leave the hook registry exactly the same size while changing what the
    aliases point at."""
    bridge = TransformerBridge.boot_transformers(MODEL_NAME, device="cpu", dtype=torch.float32)
    _ = bridge.hook_dict  # populate the cache while layer 0 is dense
    assert bridge._collect_block_instance_aliases()["blocks.0.mlp.hook_pre"].endswith(
        "dense_gate.hook_out"
    )

    bridge.blocks[0].mlp.set_original_component(bridge.blocks[1].mlp.original_component)

    assert bridge.blocks[0].mlp.bound_dense is False
    assert bridge._collect_block_instance_aliases()["blocks.0.mlp.hook_pre"].endswith("mlp.hook_in")
