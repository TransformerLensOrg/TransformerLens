"""fold_ln reaches the batched expert weights transformers 5.13 hides from the state dict.

Expert stacks are stored as single 3-D Parameters (``experts.gate_up_proj``), which are
not ``weight``/``bias`` leaves of a declared bridge submodule, so ``state_dict()`` drops
them and the state-dict fold silently skipped every sparse layer's ln2. Logits stayed
right — nothing had been scaled — but attention and dense layers ran in a folded basis
while sparse MLP layers kept their learned gains, and that mix shows up only in DLA,
logit lens and factored-matrix reads.
"""

from typing import Any, Dict, List

import pytest
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components.moe import (
    fold_scale_into_moe_block,
    has_batched_experts,
    unwrap_bridge,
)
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_config_from_hf,
)

TOKENS = torch.tensor([[1, 5, 9, 13, 17, 21]])

# moe_intermediate_size differs from hidden_size everywhere so a fold landing on the
# wrong axis is a shape error rather than a silently plausible multiply.
SPECS: Dict[str, Dict[str, Any]] = {
    "mixtral": dict(
        architecture="MixtralForCausalLM",
        sparse_layers=[0, 1],
        config=dict(
            model_type="mixtral",
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=48,
            num_local_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=64,
        ),
    ),
    "qwen2_moe": dict(
        architecture="Qwen2MoeForCausalLM",
        sparse_layers=[0, 1],
        config=dict(
            model_type="qwen2_moe",
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=48,
            moe_intermediate_size=24,
            shared_expert_intermediate_size=40,
            num_experts=4,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
            max_position_embeddings=64,
        ),
    ),
    "gpt_oss": dict(
        architecture="GptOssForCausalLM",
        sparse_layers=[0, 1],
        config=dict(
            model_type="gpt_oss",
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            intermediate_size=24,
            num_local_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=64,
        ),
    ),
    "glm4_moe": dict(
        architecture="Glm4MoeForCausalLM",
        sparse_layers=[1],
        config=dict(
            model_type="glm4_moe",
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            intermediate_size=48,
            moe_intermediate_size=24,
            n_shared_experts=1,
            n_routed_experts=4,
            num_experts_per_tok=2,
            first_k_dense_replace=1,
            max_position_embeddings=64,
        ),
    ),
}


class _Tok:
    pass


def _randomize(hf_model: nn.Module) -> None:
    """Non-unit norm gains and O(1) logits.

    from_config leaves every norm at 1.0, which makes folding a vacuous multiply, and
    leaves the projections near zero, which puts the logits in a range where even a
    badly misplaced fold moves them less than the tolerance.
    """
    with torch.no_grad():
        for name, param in hf_model.named_parameters():
            if param.ndim > 1:
                param.normal_(0.0, 0.2)
            elif name.endswith("weight"):
                param.copy_(torch.rand_like(param) + 0.5)  # a norm gain, never 1.0
            else:
                param.normal_(0.0, 0.05)  # biases, attention sinks


def _build(tag: str) -> TransformerBridge:
    spec = SPECS[tag]
    torch.manual_seed(0)
    hf_config = AutoConfig.for_model(**spec["config"])
    hf_config.architectures = [spec["architecture"]]
    hf_model = AutoModelForCausalLM.from_config(hf_config).to(torch.float32).eval()
    _randomize(hf_model)
    bridge_config = build_bridge_config_from_hf(
        hf_model.config, spec["architecture"], f"{tag}-tiny", torch.float32
    )
    adapter = ArchitectureAdapterFactory.select_architecture_adapter(bridge_config)
    return TransformerBridge(hf_model, adapter, tokenizer=_Tok())


@pytest.fixture(params=sorted(SPECS), ids=sorted(SPECS))
def bridge(request: Any) -> TransformerBridge:
    return _build(request.param)


@pytest.fixture
def sparse_layers(request: Any) -> List[int]:
    return list(SPECS[request.node.callspec.params["bridge"]]["sparse_layers"])


def _fold(bridge: TransformerBridge) -> None:
    """Fold layer norms only, so any logit movement is the fold's own doing."""
    bridge.process_weights(
        fold_ln=True,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
    )


def _norms(bridge: TransformerBridge) -> Dict[str, torch.Tensor]:
    return {
        f"blocks.{index}.{name}": unwrap_bridge(getattr(block, name)).weight.detach().clone()
        for index, block in enumerate(bridge.blocks)
        for name in ("ln1", "ln2")
    }


def _experts(bridge: TransformerBridge, layer: int) -> nn.Module:
    return unwrap_bridge(unwrap_bridge(bridge.blocks[layer].mlp).experts)


def test_every_norm_reaches_identity(bridge: TransformerBridge) -> None:
    _fold(bridge)
    unfolded = {
        key: (float(weight.min()), float(weight.max()))
        for key, weight in _norms(bridge).items()
        if not torch.allclose(weight, torch.ones_like(weight), atol=1e-6)
    }
    assert not unfolded, f"norms left unfolded: {unfolded}"


def test_logits_invariant(bridge: TransformerBridge) -> None:
    with torch.no_grad():
        before = bridge(TOKENS).clone()
    assert before.abs().max() > 1.0, "logit range too narrow to catch a misplaced fold"
    _fold(bridge)
    with torch.no_grad():
        after = bridge(TOKENS)
    torch.testing.assert_close(after, before, atol=1e-4, rtol=1e-4)


def test_input_projections_absorb_the_gain(
    bridge: TransformerBridge, sparse_layers: List[int]
) -> None:
    gains = _norms(bridge)
    before = {
        layer: _experts(bridge, layer).gate_up_proj.detach().clone() for layer in sparse_layers
    }
    _fold(bridge)
    for layer in sparse_layers:
        gain = gains[f"blocks.{layer}.ln2"]
        experts = _experts(bridge, layer)
        axis = 1 if getattr(experts, "is_transposed", False) else -1
        shape = [1] * before[layer].ndim
        shape[axis] = gain.shape[0]
        torch.testing.assert_close(
            experts.gate_up_proj, before[layer] * gain.reshape(shape), atol=1e-5, rtol=1e-5
        )


def test_down_projections_are_untouched(
    bridge: TransformerBridge, sparse_layers: List[int]
) -> None:
    """down_proj reads the expert intermediate; scaling it would double-count the gain."""
    before = {layer: _experts(bridge, layer).down_proj.detach().clone() for layer in sparse_layers}
    _fold(bridge)
    for layer in sparse_layers:
        torch.testing.assert_close(
            _experts(bridge, layer).down_proj, before[layer], atol=0.0, rtol=0.0
        )


def test_expert_biases_are_untouched(bridge: TransformerBridge, sparse_layers: List[int]) -> None:
    """A norm gain multiplies the input, so downstream biases keep their values."""
    experts = _experts(bridge, sparse_layers[0])
    before = {
        name: param.detach().clone()
        for name, param in experts.named_parameters()
        if name.endswith("_bias")
    }
    if not before:
        pytest.skip("this architecture's batched experts have no biases")
    _fold(bridge)
    for name, original in before.items():
        torch.testing.assert_close(
            dict(experts.named_parameters())[name], original, atol=0.0, rtol=0.0
        )


def test_refolding_is_a_no_op(bridge: TransformerBridge) -> None:
    """The norm is at identity after the first pass, so a second one multiplies by 1."""
    _fold(bridge)
    folded = {key: value.clone() for key, value in bridge.state_dict().items()}
    experts = {
        index: {
            name: param.detach().clone()
            for name, param in unwrap_bridge(block.mlp).named_parameters()
        }
        for index, block in enumerate(bridge.blocks)
    }

    bridge._fold_layer_norms_into_batched_experts()

    for key, value in folded.items():
        torch.testing.assert_close(bridge.state_dict()[key], value, atol=0.0, rtol=0.0, msg=key)
    for index, block in enumerate(bridge.blocks):
        for name, param in unwrap_bridge(block.mlp).named_parameters():
            torch.testing.assert_close(
                param, experts[index][name], atol=0.0, rtol=0.0, msg=f"blocks.{index}.mlp.{name}"
            )


def test_fold_ln_false_leaves_the_norms_alone(bridge: TransformerBridge) -> None:
    before = _norms(bridge)
    bridge.process_weights(
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
    )
    for key, weight in before.items():
        torch.testing.assert_close(_norms(bridge)[key], weight, atol=0.0, rtol=0.0, msg=key)


class _UnrecognizedExperts(nn.Module):
    """A batched-expert block whose input projection this code has never seen."""

    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.ones(2, 8, 4))
        self.down_proj = nn.Parameter(torch.ones(2, 4, 4))
        self.mystery_proj = nn.Parameter(torch.ones(2, 4, 4))


class _NormalizingExperts(nn.Module):
    """A batched-expert block that re-normalizes its own input."""

    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.ones(2, 8, 4))
        self.down_proj = nn.Parameter(torch.ones(2, 4, 4))
        self.inner_norm = nn.RMSNorm(4)


@pytest.mark.parametrize("block_class", [_UnrecognizedExperts, _NormalizingExperts])
def test_unclassifiable_parameter_declines_the_whole_fold(block_class: Any, caplog: Any) -> None:
    """A fold that reaches some readers and not others changes what the model computes."""
    block = block_class()
    before = {name: param.detach().clone() for name, param in block.named_parameters()}
    assert has_batched_experts(block)

    assert fold_scale_into_moe_block(block, torch.full((4,), 2.0)) is False

    assert "Not folding" in caplog.text
    for name, param in block.named_parameters():
        torch.testing.assert_close(param, before[name], atol=0.0, rtol=0.0, msg=name)


class _FlaglessTransposedExperts(nn.Module):
    """Llama4-style experts: transposed layout, no is_transposed attribute."""

    def __init__(self, d_mlp: int) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.ones(2, 4, 2 * d_mlp))
        self.down_proj = nn.Parameter(torch.ones(2, d_mlp, 4))


def test_flagless_transposed_layout_folds_on_the_d_model_axis() -> None:
    """Experts classes that predate transformers' layout flag fall back to shape."""
    block = _FlaglessTransposedExperts(d_mlp=3)
    scale = torch.arange(1.0, 5.0)

    assert fold_scale_into_moe_block(block, scale) is True

    torch.testing.assert_close(block.gate_up_proj, torch.ones(2, 4, 6) * scale[None, :, None])
    torch.testing.assert_close(block.down_proj, torch.ones(2, 3, 4))


def test_flagless_ambiguous_shape_declines_the_fold(caplog: Any) -> None:
    """2 * d_mlp == d_model puts d_model on both axes, and guessing could double-count."""
    block = _FlaglessTransposedExperts(d_mlp=2)

    assert fold_scale_into_moe_block(block, torch.full((4,), 2.0)) is False

    assert "does not pin d_model to one axis" in caplog.text
    torch.testing.assert_close(block.gate_up_proj, torch.ones(2, 4, 4))
