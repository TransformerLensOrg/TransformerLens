"""Residual-decomposition APIs on SSM / hybrid bridges.

Blocks without attention (gated-delta-net, Mamba) expose their sequence-mixer
output instead of ``hook_attn_out``; blocks without an MLP (Mamba, NemotronH)
write only through their mixer. Every decomposition must still sum to the
residual stream it decomposes.
"""
from unittest.mock import MagicMock

import pytest
import torch

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
)

try:
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    _QWEN3_5_AVAILABLE = True
except ImportError:
    _QWEN3_5_AVAILABLE = False

ATOL = 1e-5
TOKENS = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])


def _boot(hf_model, arch):
    from transformer_lens.factories.architecture_adapter_factory import (
        ArchitectureAdapterFactory,
    )

    cfg = build_bridge_config_from_hf(hf_model.config, arch, "tiny", torch.float32)
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(cfg)
    return TransformerBridge(model=hf_model, adapter=adapter, tokenizer=MagicMock())


def _qwen35():
    from transformer_lens.config.transformer_bridge_config import (
        TransformerBridgeConfig,
    )
    from transformer_lens.model_bridge.supported_architectures.qwen3_5 import (
        Qwen3_5ArchitectureAdapter,
    )

    torch.manual_seed(0)
    c = Qwen3_5TextConfig(
        hidden_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=256,
        vocab_size=512,
        full_attention_interval=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        rope_parameters={
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "rope_type": "default",
        },
    )
    m = Qwen3_5ForCausalLM(c).eval()
    bcfg = TransformerBridgeConfig(
        d_model=128,
        d_head=32,
        n_heads=4,
        n_layers=8,
        n_ctx=2048,
        d_vocab=512,
        d_mlp=256,
        n_key_value_heads=2,
        architecture="Qwen3_5ForCausalLM",
    )
    return TransformerBridge(m, Qwen3_5ArchitectureAdapter(bcfg), tokenizer=MagicMock())


def _mamba2(use_bias: bool = False):
    from transformers import AutoModelForCausalLM
    from transformers.models.mamba2 import Mamba2Config

    torch.manual_seed(0)
    c = Mamba2Config(
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
        use_bias=use_bias,
    )
    c.architectures = ["Mamba2ForCausalLM"]
    model = AutoModelForCausalLM.from_config(c).eval()
    if use_bias:
        # Non-zero output biases, so a double-counted bias row would show.
        with torch.no_grad():
            for layer in model.backbone.layers:
                layer.mixer.out_proj.bias.normal_()
    return _boot(model, "Mamba2ForCausalLM")


def _mamba2_bias():
    return _mamba2(use_bias=True)


def _nemotronh():
    from transformers import AutoModelForCausalLM
    from transformers.models.nemotron_h import NemotronHConfig

    torch.manual_seed(0)
    c = NemotronHConfig(
        vocab_size=256,
        hidden_size=64,
        layers_block_type=["mamba", "attention", "mamba", "mlp"],
        num_attention_heads=4,
        num_key_value_heads=2,
        ssm_state_size=16,
        mamba_num_heads=4,
        mamba_head_dim=16,
        n_groups=2,
        conv_kernel=4,
        expand=2,
        intermediate_size=128,
        chunk_size=8,
    )
    c.architectures = ["NemotronHForCausalLM"]
    return _boot(AutoModelForCausalLM.from_config(c).eval(), "NemotronHForCausalLM")


def _granite_dense():
    """Dense GraniteMoeHybrid (shared MLP, no experts) with a non-unit residual_multiplier:
    every sublayer output is scaled after its hook_out before the residual add."""
    from transformers import AutoModelForCausalLM
    from transformers.models.granitemoehybrid import GraniteMoeHybridConfig

    torch.manual_seed(0)
    c = GraniteMoeHybridConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=32,
        shared_intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_local_experts=0,
        max_position_embeddings=64,
        layer_types=["mamba", "attention", "mamba"],
        mamba_n_heads=8,
        mamba_n_groups=2,
        mamba_d_state=16,
        mamba_d_head=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_chunk_size=16,
        position_embedding_type="rope",
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        residual_multiplier=0.22,
    )
    c.architectures = ["GraniteMoeHybridForCausalLM"]
    return _boot(AutoModelForCausalLM.from_config(c).eval(), "GraniteMoeHybridForCausalLM")


def _zamba2():
    """Zamba2 with hybrid layers: the shared transformer feeds the Mamba-2 input of layers 1
    and 3 but does not write to the residual stream."""
    from transformers import AutoModelForCausalLM
    from transformers.models.zamba2 import Zamba2Config

    torch.manual_seed(0)
    c = Zamba2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        attention_hidden_size=128,
        layers_block_type=["mamba", "hybrid", "mamba", "hybrid"],
        hybrid_layer_ids=[1, 3],
        num_mem_blocks=1,
        use_mem_rope=False,
        mamba_d_state=16,
        mamba_ngroups=1,
        mamba_headdim=16,
        n_mamba_heads=8,
        chunk_size=8,
    )
    c.architectures = ["Zamba2ForCausalLM"]
    return _boot(AutoModelForCausalLM.from_config(c).eval(), "Zamba2ForCausalLM")


def _recurrent_gemma():
    """RecurrentGemma: RG-LRU and local-attention temporal blocks, each followed by a gated
    MLP. hidden_size 80 makes HF's bfloat16 embedding normalizer (8.9375) differ from
    sqrt(80)."""
    from transformers import AutoModelForCausalLM
    from transformers.models.recurrent_gemma import RecurrentGemmaConfig

    torch.manual_seed(0)
    c = RecurrentGemmaConfig(
        vocab_size=256,
        hidden_size=80,
        lru_width=80,
        intermediate_size=160,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=20,
        attention_window_size=8,
        block_types=["recurrent", "recurrent", "attention"],
    )
    c.architectures = ["RecurrentGemmaForCausalLM"]
    return _boot(AutoModelForCausalLM.from_config(c).eval(), "RecurrentGemmaForCausalLM")


_FACTORIES = {
    "granite_dense": _granite_dense,
    "mamba2": _mamba2,
    "mamba2_bias": _mamba2_bias,
    "nemotron_h": _nemotronh,
    "recurrent_gemma": _recurrent_gemma,
    "zamba2": _zamba2,
}
if _QWEN3_5_AVAILABLE:
    _FACTORIES["qwen3_5"] = _qwen35


@pytest.fixture(scope="module", params=sorted(_FACTORIES))
def bridge_and_cache(request):
    bridge = _FACTORIES[request.param]()
    with torch.no_grad():
        _, cache = bridge.run_with_cache(TOKENS)
    return request.param, bridge, cache


def _final_resid(bridge, cache):
    return cache[f"blocks.{bridge.cfg.n_layers - 1}.hook_resid_post"]


def _attn_layers(bridge):
    return [i for i, _ in bridge.blocks_with("attn")]


class TestDecomposeResid:
    def test_sums_to_final_residual(self, bridge_and_cache):
        _, bridge, cache = bridge_and_cache
        comps = cache.decompose_resid()
        torch.testing.assert_close(comps.sum(0), _final_resid(bridge, cache), atol=ATOL, rtol=0)

    @pytest.mark.parametrize("layer", [1, 2, 3])
    def test_sums_to_resid_pre(self, bridge_and_cache, layer):
        _, bridge, cache = bridge_and_cache
        if layer >= bridge.cfg.n_layers:
            pytest.skip("model has fewer layers")
        comps = cache.decompose_resid(layer=layer)
        torch.testing.assert_close(
            comps.sum(0), cache[f"blocks.{layer}.hook_resid_pre"], atol=ATOL, rtol=0
        )

    def test_attn_and_mlp_modes_partition_the_stream(self, bridge_and_cache):
        _, bridge, cache = bridge_and_cache
        attn = cache.decompose_resid(mode="attn", incl_embeds=False)
        mlp = cache.decompose_resid(mode="mlp", incl_embeds=False)
        embed = cache["hook_embed"]
        torch.testing.assert_close(
            attn.sum(0) + mlp.sum(0) + embed, _final_resid(bridge, cache), atol=ATOL, rtol=0
        )

    def test_labels_name_the_mixer(self, bridge_and_cache):
        name, bridge, cache = bridge_and_cache
        _, labels = cache.decompose_resid(return_labels=True)
        expected = {
            "qwen3_5": ["0_linear_attn_out", "0_mlp_out", "3_attn_out", "3_mlp_out"],
            "granite_dense": ["0_mixer_out", "0_mlp_out", "1_attn_out", "1_mlp_out"],
            "mamba2": ["0_mixer_out", "1_mixer_out"],
            "mamba2_bias": ["0_mixer_out", "1_mixer_out"],
            # Passthrough mixers are labelled by cfg.layers_block_type.
            "nemotron_h": ["0_mixer_out", "1_attn_out", "2_mixer_out", "3_mlp_out"],
            # Hybrid layers 1 and 3 write only through their Mamba-2 mixer.
            "zamba2": ["0_mixer_out", "1_mixer_out", "2_mixer_out", "3_mixer_out"],
            # Layer 2's temporal block is local attention.
            "recurrent_gemma": [
                "0_mixer_out",
                "0_mlp_out",
                "1_mixer_out",
                "1_mlp_out",
                "2_attn_out",
                "2_mlp_out",
                "3_mixer_out",
                "3_mlp_out",
            ],
        }[name]
        if name in ("nemotron_h", "zamba2", "recurrent_gemma"):
            assert labels[1:] == expected
        else:
            assert set(expected) <= set(labels)
        if name in ("mamba2", "mamba2_bias"):
            assert not any(label.endswith("_mlp_out") for label in labels)

    def test_mlp_type_mixers_fill_the_mlp_slot(self, bridge_and_cache):
        name, bridge, cache = bridge_and_cache
        if name != "nemotron_h":
            pytest.skip("only NemotronH has MLP-type passthrough mixers")
        _, mlp_labels = cache.decompose_resid(mode="mlp", incl_embeds=False, return_labels=True)
        _, attn_labels = cache.decompose_resid(mode="attn", incl_embeds=False, return_labels=True)
        assert mlp_labels == ["3_mlp_out"]
        assert attn_labels == ["0_mixer_out", "1_attn_out", "2_mixer_out"]

    def test_mlp_input_sums_to_resid_mid(self, bridge_and_cache):
        name, bridge, cache = bridge_and_cache
        if name not in ("qwen3_5", "recurrent_gemma"):
            pytest.skip("only Qwen3.5 and RecurrentGemma blocks have a separate MLP input")
        for layer in (0, 2, 3):
            comps = cache.decompose_resid(layer=layer, mlp_input=True)
            torch.testing.assert_close(
                comps.sum(0), cache[f"blocks.{layer}.hook_resid_mid"], atol=ATOL, rtol=0
            )

    def test_logit_attrs_runs(self, bridge_and_cache):
        _, bridge, cache = bridge_and_cache
        attrs = cache.logit_attrs(cache.decompose_resid(), tokens=torch.tensor([3, 3]))
        assert attrs.shape[0] == cache.decompose_resid().shape[0]


class TestHeadResults:
    def test_heads_only_on_attention_layers(self, bridge_and_cache):
        _, bridge, cache = bridge_and_cache
        _, labels = cache.stack_head_results(incl_remainder=False, return_labels=True)
        expected = [f"L{l}H{h}" for l in _attn_layers(bridge) for h in range(bridge.cfg.n_heads)]
        assert labels == expected

    def test_heads_plus_remainder_sum_to_final_residual(self, bridge_and_cache):
        _, bridge, cache = bridge_and_cache
        stack = cache.stack_head_results(incl_remainder=True)
        torch.testing.assert_close(stack.sum(0), _final_resid(bridge, cache), atol=ATOL, rtol=0)

    def test_head_results_sum_to_attn_out(self, bridge_and_cache):
        _, bridge, cache = bridge_and_cache
        cache.compute_head_results()
        for layer in _attn_layers(bridge):
            heads = cache[f"blocks.{layer}.attn.hook_result"].sum(-2)
            b_O = bridge.blocks[layer].attn.b_O
            if b_O is not None:
                heads = heads + b_O
            torch.testing.assert_close(
                heads, cache[f"blocks.{layer}.hook_attn_out"], atol=ATOL, rtol=0
            )


class TestFullResidDecomposition:
    @pytest.mark.parametrize("expand_neurons", [False, True])
    def test_sums_to_final_residual(self, bridge_and_cache, expand_neurons):
        name, bridge, cache = bridge_and_cache
        if name == "granite_dense" and expand_neurons:
            # Its MLP is `shared_mlp`, outside the `mlp` slot neuron expansion reads.
            with pytest.raises(NotImplementedError, match="shared_mlp"):
                cache.get_full_resid_decomposition(expand_neurons=True)
            return
        stack, labels = cache.get_full_resid_decomposition(
            expand_neurons=expand_neurons, return_labels=True
        )
        assert stack.shape[0] == len(labels)
        torch.testing.assert_close(stack.sum(0), _final_resid(bridge, cache), atol=ATOL, rtol=0)

    @pytest.mark.parametrize("expand_neurons", [False, True])
    def test_ln_and_projection_sum_to_projected_scaled_residual(
        self, bridge_and_cache, expand_neurons
    ):
        name, bridge, cache = bridge_and_cache
        if name == "granite_dense" and expand_neurons:
            # Its MLP is `shared_mlp`, outside the `mlp` slot neuron expansion reads.
            with pytest.raises(NotImplementedError, match="shared_mlp"):
                cache.get_full_resid_decomposition(expand_neurons=True)
            return
        torch.manual_seed(1)
        directions = torch.randn(bridge.cfg.d_model, 3)
        stack = cache.get_full_resid_decomposition(
            expand_neurons=expand_neurons, apply_ln=True, project_output_onto=directions
        )
        final = _final_resid(bridge, cache)
        expected = cache.apply_ln_to_stack(final[None], bridge.cfg.n_layers)[0] @ directions
        torch.testing.assert_close(stack.sum(0), expected, atol=1e-4, rtol=0)


def test_zamba2_hybrid_layers_write_only_their_mamba_output():
    bridge = _zamba2()
    with torch.no_grad():
        _, cache = bridge.run_with_cache(TOKENS)
    for layer in range(bridge.cfg.n_layers):
        write = cache[f"blocks.{layer}.hook_resid_post"] - cache[f"blocks.{layer}.hook_resid_pre"]
        torch.testing.assert_close(
            write, cache[f"blocks.{layer}.mixer.hook_out"], atol=ATOL, rtol=0
        )
    # On a hybrid layer the norm reads the residual plus the shared transformer's output.
    assert not torch.allclose(cache["blocks.1.norm.hook_in"], cache["blocks.1.hook_resid_pre"])


def test_recurrent_gemma_hook_embed_is_the_scaled_embedding():
    """HF scales the embedding by its bfloat16 normalizer after embed_tokens."""
    bridge = _recurrent_gemma()
    with torch.no_grad():
        _, cache = bridge.run_with_cache(TOKENS)
    torch.testing.assert_close(
        cache["hook_embed"], cache["blocks.0.hook_resid_pre"], atol=0, rtol=0
    )
    for layer in range(bridge.cfg.n_layers):
        resid_mid = (
            cache[f"blocks.{layer}.hook_resid_pre"] + cache[f"blocks.{layer}.hook_mixer_out"]
        )
        torch.testing.assert_close(
            resid_mid, cache[f"blocks.{layer}.hook_resid_mid"], atol=ATOL, rtol=0
        )
